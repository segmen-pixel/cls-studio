# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Per-patch metadata for time-aware critical / negative memory banks.

Each row in ``critical/<label>.npy`` (or ``negative/<label>.npy``) is paired
with one row in this metadata array. The bank stores the *features*; this
module stores the bookkeeping needed to make memory time-aware:

    - severity:                  human-tagged 1 (light) / 2 (medium) / 3 (heavy)
    - registered_at_inspection:  inspection_count when the row was first added
    - last_hit_at_inspection:    inspection_count of the most recent re-hit
    - hit_count:                 number of inferences that matched this row
    - tier:                      memory consolidation tier (0=short, 1=mid, 2=long)

freshness is *not* persisted: it's computed lazily from
``inspection_count - last_hit_at_inspection`` so the decay schedule can be
changed without rewriting the bank.

The on-disk format is one ``.meta.npz`` per label, written next to the
``.npy`` feature file. Loading a legacy bank with no metadata files
synthesises an array of defaults (severity=2, hit_count=0, tier=short,
registered/last_hit = 0) so AUROC stays bit-exact when scoring still
ignores the metadata (Phase 1a).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fsio import replace_with_retry

# Severity 3-step scale. Likert reproducibility past 5 steps is poor and
# 10-step is unusable on the line; 3 is the coarsest scale that still lets
# operators communicate "trivial / normal / critical" without forcing
# artificial precision they can't sustain across shifts.
SEVERITY_LIGHT: int = 1
SEVERITY_MEDIUM: int = 2
SEVERITY_HEAVY: int = 3
DEFAULT_SEVERITY: int = SEVERITY_MEDIUM

# Memory consolidation tiers (short-term -> mid-term -> long-term). New
# entries start in short; promotion to mid/long happens in the (Phase 1c)
# decay/promote batch based on hit_count thresholds.
TIER_SHORT: int = 0
TIER_MID: int = 1
TIER_LONG: int = 2
DEFAULT_TIER: int = TIER_SHORT

# Decay time constants in *inspections* (not wall-clock). freshness is
# computed as ``exp(-delta / tau[tier])`` where ``delta = inspection_count
# - last_hit_at_inspection``. Picking inspections rather than seconds
# means a 10 Hz line and a 0.01 Hz line forget at the same operational
# pace; only the actual exposure of the row to inferences matters.
#
# Initial values are first-pass guesses tied to typical line cadence:
#   short: ~100 inspections (a few hours on a slow line, minutes on a fast)
#   mid:   ~1000 inspections
#   long:  ~10000 inspections
# Phase 1c's decay/promote batch will let us tune these from real logs.
TAU_SHORT: int = 100
TAU_MID: int = 1_000
TAU_LONG: int = 10_000
TAU_BY_TIER: tuple[int, int, int] = (TAU_SHORT, TAU_MID, TAU_LONG)

# Phase 1c: promotion / death thresholds for the decay batch.
#
# A row is promoted short -> mid -> long once it has been re-hit enough
# times to qualify as "this incident keeps coming back". Numbers are
# first-pass guesses; a real ablation needs ~weeks of HITL telemetry,
# which we don't have yet.
PROMOTE_TO_MID_HITS: int = 3
PROMOTE_TO_LONG_HITS: int = 10

# A short-tier row whose freshness drops below DEATH_FRESHNESS is
# retired by the decay batch. 0.05 corresponds to delta ~= 3 * tau, i.e.
# ~300 inspections on the default short tau — long enough that a single
# noisy day doesn't kill a real incident, short enough that "I appended
# this once and never saw it again" cleans itself up.
#
# mid / long tiers are protected from death entirely; they only ever get
# promoted, never removed, until Phase 2's reconsolidation overwrite path.
DEATH_FRESHNESS: float = 0.05
DEATH_PROTECTED_TIERS: tuple[int, ...] = (TIER_MID, TIER_LONG)


def severity_weight(severity: np.ndarray) -> np.ndarray:
    """Linear severity weight in {0.5, 1.0, 1.5} for {1, 2, 3}.

    Anchored at ``DEFAULT_SEVERITY=2 -> 1.0`` so a default-severity entry
    contributes the same as a Phase 1a unweighted entry; this is what
    keeps the multiplier neutral on a freshly-loaded legacy bank and
    prevents an accidental AUROC drift when ``weighted=True`` is first
    flipped on.
    """
    return severity.astype(np.float32) / float(DEFAULT_SEVERITY)


def freshness(
    last_hit_at_inspection: np.ndarray,
    tier: np.ndarray,
    inspection_count: int,
) -> np.ndarray:
    """Per-row freshness in [0, 1] from ``exp(-delta / tau[tier])``.

    ``delta`` is clipped to >=0 to defend against (rare but possible)
    metadata that records a ``last_hit_at`` ahead of the current counter
    — e.g. after a partial restore from a stale backup. Without the clip
    such rows would erroneously show freshness > 1.
    """
    tau = np.asarray(TAU_BY_TIER, dtype=np.float32)[tier.astype(np.int64)]
    delta = np.maximum(
        inspection_count - last_hit_at_inspection.astype(np.int64),
        0,
    ).astype(np.float32)
    return np.exp(-delta / tau)


def multiplier_for(meta: IncidentMetaArray, inspection_count: int) -> np.ndarray:
    """Per-row weighting in scoring: ``severity_weight * freshness``.

    A default-severity (medium=2), never-decayed row produces 1.0 — i.e.
    the same distance the unweighted scoring would compute — which is
    why turning ``weighted`` on is non-disruptive on a fresh bank. As
    severity rises or freshness decays this multiplier drifts away from
    1.0 and the divide-by-multiplier in ``scoring._per_label_min``
    starts shrinking or growing distances.
    """
    return severity_weight(meta.severity) * freshness(
        meta.last_hit_at_inspection,
        meta.tier,
        inspection_count,
    )


@dataclass
class IncidentMetaArray:
    """Parallel metadata for one labelled sub-bank's feature array.

    Every column has length N == features.shape[0]. Append/concat keep the
    invariant that lengths match across this dataclass and the feature
    ndarray; ``Bank`` is responsible for calling ``append()`` exactly when
    it appends rows to the corresponding ``.npy``.
    """

    severity: np.ndarray  # [N] uint8, in {1, 2, 3}
    registered_at_inspection: np.ndarray  # [N] uint64
    last_hit_at_inspection: np.ndarray  # [N] uint64
    hit_count: np.ndarray  # [N] uint32
    tier: np.ndarray  # [N] uint8, in {0, 1, 2}

    @classmethod
    def empty(cls) -> IncidentMetaArray:
        return cls(
            severity=np.zeros(0, dtype=np.uint8),
            registered_at_inspection=np.zeros(0, dtype=np.uint64),
            last_hit_at_inspection=np.zeros(0, dtype=np.uint64),
            hit_count=np.zeros(0, dtype=np.uint32),
            tier=np.zeros(0, dtype=np.uint8),
        )

    @classmethod
    def defaults_for(cls, n: int, registered_at: int = 0) -> IncidentMetaArray:
        """Build N rows of defaults — used when a legacy bank has no metadata file.

        ``last_hit_at_inspection`` defaults to ``registered_at`` so the very
        first freshness computation (Phase 1b) gives delta=0 and treats
        legacy entries as fully fresh, not as already-decayed.
        """
        return cls(
            severity=np.full(n, DEFAULT_SEVERITY, dtype=np.uint8),
            registered_at_inspection=np.full(n, registered_at, dtype=np.uint64),
            last_hit_at_inspection=np.full(n, registered_at, dtype=np.uint64),
            hit_count=np.zeros(n, dtype=np.uint32),
            tier=np.full(n, DEFAULT_TIER, dtype=np.uint8),
        )

    def __len__(self) -> int:
        return int(self.severity.shape[0])

    def append(self, n: int, severity: int, inspection_count: int) -> None:
        """Extend in-place by N rows of identical metadata.

        Called by ``Bank.append`` to keep one row of metadata per row of
        features. ``severity`` is clipped to {1, 2, 3} so a buggy caller
        can't poison the scale.
        """
        if n <= 0:
            return
        sev = int(np.clip(int(severity), SEVERITY_LIGHT, SEVERITY_HEAVY))
        ic = int(inspection_count)
        self.severity = np.concatenate([self.severity, np.full(n, sev, dtype=np.uint8)])
        self.registered_at_inspection = np.concatenate(
            [self.registered_at_inspection, np.full(n, ic, dtype=np.uint64)]
        )
        self.last_hit_at_inspection = np.concatenate(
            [self.last_hit_at_inspection, np.full(n, ic, dtype=np.uint64)]
        )
        self.hit_count = np.concatenate([self.hit_count, np.zeros(n, dtype=np.uint32)])
        self.tier = np.concatenate([self.tier, np.full(n, DEFAULT_TIER, dtype=np.uint8)])

    def take(self, idx: np.ndarray) -> IncidentMetaArray:
        """Reorder/subset by ``idx`` (e.g. after k-Center coreset reduction).

        Mirrors ``features[idx]`` so a coreset that drops 90% of feature
        rows drops the same 90% of metadata rows.
        """
        return IncidentMetaArray(
            severity=self.severity[idx],
            registered_at_inspection=self.registered_at_inspection[idx],
            last_hit_at_inspection=self.last_hit_at_inspection[idx],
            hit_count=self.hit_count[idx],
            tier=self.tier[idx],
        )

    def save(self, path: Path) -> None:
        """Atomic write so a torn save doesn't corrupt the metadata.

        We pass an open file handle to ``np.savez`` rather than a string
        path: numpy auto-appends ``.npz`` to string paths that don't end in
        it, which would silently turn ``foo.meta.npz.tmp`` into
        ``foo.meta.npz.tmp.npz`` and break the subsequent ``replace()``.
        Using a handle keeps the tmp filename literal.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                severity=self.severity,
                registered_at_inspection=self.registered_at_inspection,
                last_hit_at_inspection=self.last_hit_at_inspection,
                hit_count=self.hit_count,
                tier=self.tier,
            )
        replace_with_retry(tmp, path)

    @classmethod
    def load(cls, path: Path) -> IncidentMetaArray:
        with np.load(path) as z:
            return cls(
                severity=z["severity"].astype(np.uint8, copy=False),
                registered_at_inspection=z["registered_at_inspection"].astype(
                    np.uint64, copy=False
                ),
                last_hit_at_inspection=z["last_hit_at_inspection"].astype(
                    np.uint64, copy=False
                ),
                hit_count=z["hit_count"].astype(np.uint32, copy=False),
                tier=z["tier"].astype(np.uint8, copy=False),
            )

    def assert_matches(self, n_features: int, label: str) -> None:
        """Defensive check used at load time: features and metadata lengths agree.

        A mismatch here means the bank has been edited in two places without
        going through ``Bank.append`` — almost always a bug. We refuse to
        load rather than silently truncate / pad, because the wrong choice
        permanently corrupts the bank.
        """
        if len(self) != n_features:
            raise ValueError(
                f"incident metadata length ({len(self)}) does not match "
                f"feature count ({n_features}) for label {label!r}; "
                f"refusing to load to avoid silent corruption"
            )
