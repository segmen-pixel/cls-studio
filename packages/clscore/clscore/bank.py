# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""3-tier memory bank: normal / critical / negative.

Critical and negative are split per-label so HITL annotators can attribute a
detection to a specific defect class (e.g. "scratch", "stain") and so the API
can return per-class distance contributions. ``normal`` stays a single tensor
because it represents the whole nominal manifold.

The bank is a thin wrapper around the feature arrays plus a JSON manifest. It
supports load/save round-trips, append-only updates (the HITL primitive), and
coreset-based reduction of the normal bank (the standard PatchCore-style
greedy k-Center on a sparse-random-projected proxy space).

This module deliberately does *not* depend on any DINOv2 / scoring code; it
only knows how to manage feature arrays.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .feature_extractor import DEFAULT_DINO_DIM, DEFAULT_DINO_NAME
from .fsio import atomic_save_npy, atomic_write_text
from .incident import (
    DEATH_FRESHNESS,
    DEATH_PROTECTED_TIERS,
    DEFAULT_SEVERITY,
    PROMOTE_TO_LONG_HITS,
    PROMOTE_TO_MID_HITS,
    SEVERITY_HEAVY,
    SEVERITY_LIGHT,
    TIER_LONG,
    TIER_MID,
    TIER_SHORT,
    IncidentMetaArray,
    freshness,
)
from .sw import DINO_PATCH, WINDOW_SIZE, WINDOW_STRIDE

logger = logging.getLogger(__name__)

__all__ = [
    "BANK_DTYPE",
    "Bank",
    "BankMeta",
    "Tier",
    "DEFAULT_LABEL",
    "safe_label",
    "kcenter_greedy",
    "coreset_reduce",
]

PROJ_DIM: int = 128
DEFAULT_LABEL: str = "_default"
META_SUFFIX: str = ".meta.npz"

# Feature arrays are held and persisted as float16. GPU scoring already casts
# every bank row to fp16 before the distance computation (see
# ``ClsStudioState.get_*_tensor*`` / ``to_tensors``), so storing fp16 is the
# same quantisation applied earlier — CUDA verdicts are bit-identical — while
# RAM, disk and export packages halve. CPU-only scoring (fp32 compute) sees
# the quantised rows, which is the documented trade-off. Analytical paths
# that need fp32 (PCA, coreset projection) cast up at their boundary.
BANK_DTYPE = np.float16

# Key under which a row-range index entry records the store entry it came
# from. Present only on banks assembled after 2026-08; readers must fall back
# to matching on ``name`` for anything older.
INDEX_ENTRY_ID_KEY = "entry_id"

# The per-tier source-image copies the app writes next to a bank. clscore does
# not create them, but ``_carve`` mints refs into this directory when it
# migrates an append-era bank into a store, so the name has to live somewhere
# both sides can see.
SOURCE_IMAGES_SUBDIR = "_images"

Tier = Literal["normal", "critical", "negative"]


def safe_label(label: str | None) -> str:
    """A filename-safe stem that is UNIQUE per distinct label.

    Letters/digits/underscore/hyphen are kept verbatim; anything else
    collapses to a single underscore. Empty / whitespace-only labels fall back
    to ``DEFAULT_LABEL`` so the on-disk layout is always well-defined.

    The stem is not just a filename — it is the label's PRIMARY KEY. It keys
    ``bank.critical[label]``, ``BankMeta.*_image_index[label]``, the per-row
    metadata, the eval-cache namespace and the ``<label>.npy`` file itself. So
    the sanitiser has to be injective, and the collapsing rule alone is not:

        safe_label("傷")    -> ""    -> DEFAULT_LABEL
        safe_label("汚れ")  -> ""    -> DEFAULT_LABEL   ← same bucket
        safe_label("傷A")   -> "_A"  -> "A"
        safe_label("汚れA") -> "_A"  -> "A"             ← same bucket

    This is a Japanese shop and the UI actively invites Japanese defect names
    (`label.newKind` = "新しい種類"), so those are the ordinary cases, not
    edge cases: two defect classes merged into one tensor, one row index and
    one metadata array, and the chip picker showed a single bucket.

    A short digest of the ORIGINAL string is appended whenever the collapse
    lost information, which makes the mapping injective while leaving every
    already-legal lowercase label — "scratch", "dent", "lot-3", "_default" —
    byte-identical. That matters: those banks keep their filenames, their
    assembly fingerprint and their eval cache, so only projects that actually
    hit the bug are asked to re-assemble.

    CASE COUNTS AS LOST INFORMATION, because the stem is a FILENAME and this
    is a Windows desktop app. "Scratch" and "scratch" are two keys in Python
    and one directory entry on NTFS (and on a default macOS volume): both were
    written to critical/<label>.npy, the second replaced the first, the stale
    sweep saw a stem that WAS in its keep-set and removed nothing, and
    _load_tier rebuilt the label dict from the surviving filename — one label,
    while bank_meta.json still named two. A whole defect class vanished with
    nothing raised or logged, on the ordinary tab switch that re-loads the
    bank. The UI's own comments name the scenario ("how a bank ends up with
    'scratch', 'Scratch' and 'scrach'") and the input that produces it is free
    text with no normalisation.

    So the fast path requires an already-LOWERCASE legal stem, and every other
    output is lowercased before the digest is appended. Since sha256 hex is
    lowercase, the entire output namespace is lowercase and therefore closed
    under case folding: no two distinct labels can share a directory entry.
    Every other filesystem slug in this codebase already folds case
    (``slug_labelset_id``, ``slug_bank_id``); this one was the exception.

    The digest is over the stripped original, so it is stable across runs,
    machines and Python versions (``hash()`` is not).
    """
    if not label or not str(label).strip():
        return DEFAULT_LABEL
    raw = str(label).strip()
    subbed = re.sub(r"[^A-Za-z0-9_\-]+", "_", raw)
    if subbed == raw and raw == raw.lower():
        # Already a legal, already-lowercase stem, so it IS the identity --
        # returned verbatim, underscores and all. Testing the substitution
        # rather than the stripped result matters: ``.strip("_")`` would
        # otherwise turn ``_default`` into ``default-3bf30573`` and split the
        # bucket the whole codebase special-cases by name.
        return raw
    stem = subbed.strip("_").lower()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}" if stem else f"c-{digest}"


@dataclass
class BankMeta:
    """Persisted alongside the .npy files; describes how the bank was built.

    ``critical_images`` / ``negative_images`` are now ``dict[label, list[str]]``
    so each defect class keeps its own provenance log. Legacy meta files where
    these were plain ``list[str]`` are migrated to ``{"_default": [...]}`` on
    load.
    """

    model: str = DEFAULT_DINO_NAME
    dim: int = DEFAULT_DINO_DIM
    layers: list[int] | None = None
    window: int = WINDOW_SIZE
    stride: int = WINDOW_STRIDE
    patch: int = DINO_PATCH
    coreset_ratio: float = 1.0
    n_patches_pre_coreset: int = 0
    n_patches: int = 0
    bank_images: list[str] = field(default_factory=list)
    critical_images: dict[str, list[str]] = field(default_factory=dict)
    negative_images: dict[str, list[str]] = field(default_factory=dict)
    # Row-range index: for every image appended to the tier we record
    # ``{"name": str, "start": int, "count": int}`` so a later
    # per-image delete can prune the corresponding rows out of
    # ``bank.<tier>[label][start:start+count]`` without rebuilding the
    # whole bank. Legacy banks (built before 0.2) have these empty —
    # the per-image delete then just skips the patch pruning step and
    # treats the request as "drop the filename log + capture file only".
    normal_image_index: list[dict] = field(default_factory=list)
    critical_image_index: dict[str, list[dict]] = field(default_factory=dict)
    negative_image_index: dict[str, list[dict]] = field(default_factory=dict)
    # ``safe_label`` stem -> the text the operator actually typed. The stem is
    # injective but not readable: a Japanese defect class lands on disk as
    # "A-3f2a1b9c", and every surface that shows a label to a human needs the
    # original back. Absent on banks assembled before 2026-08, so readers fall
    # back to the stem.
    label_display: dict[str, str] = field(default_factory=dict)
    # Operator-machine path the bank was originally built from. Useful for
    # local provenance but intentionally **not persisted** (see ``to_json``)
    # so a shared bank directory can never leak the original absolute path.
    project_path: str = ""
    description: str = ""
    # Monotonic counter advanced once per inference. The unit of "time" for
    # the time-aware critical/negative metadata (registered_at_inspection,
    # last_hit_at_inspection): wall-clock would mean a high-throughput line
    # forgets faster than a slow line for no real reason; counting
    # inspections instead makes decay invariant to line speed.
    inspection_count: int = 0

    def to_json(self) -> str:
        """Serialise the manifest to JSON, scrubbing operator-local paths.

        ``project_path`` is dropped on serialisation: it carries the absolute
        path on the operator's machine where the bank was first built and
        leaking it into a shared bank directory would expose the original
        username and folder layout. The field stays available on the
        in-memory ``BankMeta`` for callers who want to log it locally.
        """
        data = asdict(self)
        data.pop("project_path", None)
        return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_path(cls, path: Path) -> BankMeta:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Migrate legacy ``list[str]`` representation to per-label ``dict``.
        for key in ("critical_images", "negative_images"):
            v = data.get(key)
            if isinstance(v, list):
                data[key] = {DEFAULT_LABEL: v} if v else {}
        # Legacy banks may have a persisted ``project_path``; we drop it on
        # load so the in-memory copy doesn't carry over a stranger's path
        # (the field is internal-only, see ``to_json``).
        data.pop("project_path", None)
        return cls(**{k: data[k] for k in data if k in cls.__dataclass_fields__})


def kcenter_greedy(
    features: torch.Tensor,
    n_select: int,
    seed: int = 42,
    batch_size: int = 64,
) -> np.ndarray:
    """Approximate greedy k-Center sampling on `features`.

    The naive single-pick loop CPU-syncs once per iteration via ``.item()``,
    which makes 100k iterations take minutes. Here we pick the ``batch_size``
    farthest points in one shot, update distances against the whole batch,
    and repeat — about ``n_select / batch_size`` GPU steps. The approximation
    is well-known to lose only a small amount of coverage compared to true
    sequential greedy and is the standard PatchCore-scale recipe.
    """
    n = features.shape[0]
    if n_select >= n:
        return np.arange(n)
    device = features.device
    g = torch.Generator(device=device).manual_seed(seed)
    init = int(torch.randint(0, n, (1,), generator=g, device=device).item())
    selected_t = torch.tensor([init], device=device, dtype=torch.long)
    dists = torch.linalg.norm(features - features[init], dim=1)
    while selected_t.shape[0] < n_select:
        k = min(batch_size, n_select - selected_t.shape[0])
        # Top-k farthest under current distances; this is the batch
        # approximation. After updating dists with the new batch's
        # contributions, the next iteration is correct again.
        topk_idx = torch.topk(dists, k, largest=True).indices
        selected_t = torch.cat([selected_t, topk_idx], dim=0)
        # Refresh distances: each query gets ``min(prev_dist, dist_to_any_new_pick)``.
        # MM mode: the default heuristic's brute kernel has no fp16 support
        # when both operands are tiny (see clscore.scoring._per_label_reduce).
        new_d = torch.cdist(
            features, features[topk_idx], compute_mode="use_mm_for_euclid_dist",
        ).min(dim=1).values
        dists = torch.minimum(dists, new_d)
    return selected_t.cpu().numpy().astype(np.int64)


def coreset_reduce(
    features: np.ndarray,
    ratio: float,
    device: str,
    seed: int = 42,
    proj_dim: int = PROJ_DIM,
) -> np.ndarray:
    """Sparse random projection -> greedy k-Center on the projection. Returns subset."""
    return coreset_reduce_indexed(features, ratio, device, seed, proj_dim)[0]


def coreset_reduce_indexed(
    features: np.ndarray,
    ratio: float,
    device: str,
    seed: int = 42,
    proj_dim: int = PROJ_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`coreset_reduce`, but also returns which input rows were kept.

    ``kept[i]`` is the index — into the image's original patch grid — of the
    row that ended up at position ``i`` of the subset. The labelled tiers need
    that map: an NG image's bank rows are addressed geometrically by
    :func:`clscore.sw.rows_for_rects`, and once the image is reduced, position
    ``i`` in the bank is no longer patch ``i`` of the grid.

    Selection ORDER is preserved, not sorted: callers truncate with
    ``feats[:headroom]`` and rely on the greedy farthest-point prefix staying
    the diverse one.
    """
    from sklearn.random_projection import SparseRandomProjection

    n_select = max(1, int(features.shape[0] * ratio))
    if n_select >= features.shape[0]:
        return features, np.arange(features.shape[0], dtype=np.int64)
    logger.info(
        "coreset: %d -> %d patches (random-projection %d->%d, k-center greedy)",
        features.shape[0],
        n_select,
        features.shape[1],
        proj_dim,
    )
    rp = SparseRandomProjection(n_components=proj_dim, random_state=seed)
    proj = rp.fit_transform(features.astype(np.float32))
    proj_t = torch.from_numpy(proj).to(device)
    idx = np.asarray(kcenter_greedy(proj_t, n_select, seed=seed), dtype=np.int64)
    return features[idx], idx


class Bank:
    """3-tier memory bank with per-label critical/negative sub-banks.

    Disk layout (all relative to `directory`):
        bank.npy                       - normal patches            [N0, D]
        critical/<label>.npy           - kept-NG patches per class [Nl, D]
        negative/<label>.npy           - FP-suppression per class  [Nl, D]
        bank_meta.json                 - BankMeta serialised

    Backwards compatibility: legacy single-file layouts
    (``critical_bank.npy`` / ``negative_bank.npy``) are read on load as the
    sole entry under the ``_default`` label and removed on the next save.
    """

    NORMAL_FILE = "bank.npy"
    CRITICAL_DIR = "critical"
    NEGATIVE_DIR = "negative"
    LEGACY_CRITICAL_FILE = "critical_bank.npy"
    LEGACY_NEGATIVE_FILE = "negative_bank.npy"
    META_FILE = "bank_meta.json"

    def __init__(
        self,
        normal: np.ndarray,
        critical: dict[str, np.ndarray] | None = None,
        negative: dict[str, np.ndarray] | None = None,
        meta: BankMeta | None = None,
        critical_meta: dict[str, IncidentMetaArray] | None = None,
        negative_meta: dict[str, IncidentMetaArray] | None = None,
    ) -> None:
        self.normal = normal.astype(BANK_DTYPE, copy=False)
        self.critical: dict[str, np.ndarray] = _as_bank_dtype_dict(critical)
        self.negative: dict[str, np.ndarray] = _as_bank_dtype_dict(negative)
        self.meta = meta or BankMeta(dim=int(normal.shape[1]) if normal.size else DEFAULT_DINO_DIM)
        # Per-row metadata for the labelled tiers. Synthesised from defaults
        # when the caller (or load()) didn't pass any, so a callsite never
        # has to special-case "no metadata yet". The Bank is responsible for
        # keeping len(meta) == len(features) for every label going forward.
        self.critical_meta: dict[str, IncidentMetaArray] = _ensure_meta_dict(
            self.critical, critical_meta, registered_at=self.meta.inspection_count
        )
        self.negative_meta: dict[str, IncidentMetaArray] = _ensure_meta_dict(
            self.negative, negative_meta, registered_at=self.meta.inspection_count
        )

    # ---- load / save ------------------------------------------------------

    @classmethod
    def _load_tier(cls, directory: Path, subdir: str, legacy_file: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        d = directory / subdir
        if d.is_dir():
            for f in sorted(d.glob("*.npy")):
                out[f.stem] = np.load(f)
        legacy = directory / legacy_file
        if legacy.exists() and DEFAULT_LABEL not in out:
            out[DEFAULT_LABEL] = np.load(legacy)
        return out

    @classmethod
    def _load_tier_meta(
        cls,
        directory: Path,
        subdir: str,
        features: dict[str, np.ndarray],
        registered_at: int,
    ) -> dict[str, IncidentMetaArray]:
        """Load ``<label>.meta.npz`` siblings of each ``<label>.npy``.

        For any label whose ``.meta.npz`` is missing (the legacy case for
        banks built before time-aware metadata existed) we synthesise an
        N-row defaults array so downstream code never has to branch on
        "this label has no metadata". A length mismatch between metadata
        and features is treated as corruption and raised, since silently
        truncating or padding would invalidate every per-row decision.
        """
        out: dict[str, IncidentMetaArray] = {}
        d = directory / subdir
        for label, arr in features.items():
            meta_path = d / f"{label}{META_SUFFIX}"
            if meta_path.exists():
                m = IncidentMetaArray.load(meta_path)
                try:
                    m.assert_matches(int(arr.shape[0]), label)
                except ValueError:
                    # A crash between the .npy and .meta.npz saves (the pair
                    # is not atomic together) used to brick the whole bank:
                    # every activate 422'd with no repair path. Rebuilding
                    # defaults loses this label's severity marks / freshness
                    # — recoverable by re-marking — while refusing to load
                    # would strand every taught feature row.
                    logger.warning(
                        "%s/%s: metadata rows (%d) != feature rows (%d) — "
                        "rebuilding default metadata (severity marks for "
                        "this label are reset)",
                        subdir, label, len(m), int(arr.shape[0]),
                    )
                    m = IncidentMetaArray.defaults_for(
                        int(arr.shape[0]), registered_at=registered_at
                    )
            else:
                m = IncidentMetaArray.defaults_for(int(arr.shape[0]), registered_at=registered_at)
            out[label] = m
        return out

    @classmethod
    def load(cls, directory: Path) -> Bank:
        """Load a bank and its metadata from disk.

        Args:
            directory: Bank directory containing ``bank.npy``,
                ``bank_meta.json`` (optional), and per-label tier subdirs.

        Returns:
            A ``Bank`` populated with normal / critical / negative arrays
            and per-row incident metadata.

        Raises:
            FileNotFoundError: If ``bank.npy`` is missing.
            OSError: If required files cannot be read.
            ValueError: If metadata and feature shapes disagree.
        """
        directory = Path(directory)
        normal = np.load(directory / cls.NORMAL_FILE)
        critical = cls._load_tier(directory, cls.CRITICAL_DIR, cls.LEGACY_CRITICAL_FILE)
        negative = cls._load_tier(directory, cls.NEGATIVE_DIR, cls.LEGACY_NEGATIVE_FILE)
        meta_path = directory / cls.META_FILE
        meta = BankMeta.from_path(meta_path) if meta_path.exists() else None
        # Default registered_at for legacy entries: current inspection_count
        # (0 for a brand-new bank, or whatever the previous run got to).
        # See incident.IncidentMetaArray.defaults_for for why this matters.
        registered_at = meta.inspection_count if meta is not None else 0
        critical_meta = cls._load_tier_meta(directory, cls.CRITICAL_DIR, critical, registered_at)
        negative_meta = cls._load_tier_meta(directory, cls.NEGATIVE_DIR, negative, registered_at)
        return cls(
            normal=normal,
            critical=critical,
            negative=negative,
            meta=meta,
            critical_meta=critical_meta,
            negative_meta=negative_meta,
        )

    @staticmethod
    def _save_tier(
        tier_dir: Path,
        items: dict[str, np.ndarray],
        meta_items: dict[str, IncidentMetaArray] | None = None,
    ) -> None:
        """Write each label's .npy (+ .meta.npz) and remove any stale ones on disk.

        A label that was cleared in-memory must disappear from disk; otherwise
        a subsequent load would resurrect it. We also drop the matching
        ``.meta.npz`` so the per-row metadata can never linger past the
        feature file it's supposed to describe.
        """
        tier_dir.mkdir(parents=True, exist_ok=True)
        # Leftover tmps from a crash between tmp-write and replace: harmless
        # (the real file is intact) but they'd accumulate forever.
        for f in tier_dir.glob("*.tmp"):
            f.unlink(missing_ok=True)
        kept: set[str] = set()
        for label, arr in items.items():
            if arr is None or arr.size == 0:
                continue
            atomic_save_npy(tier_dir / f"{label}.npy", arr)
            if meta_items is not None and label in meta_items:
                meta_items[label].save(tier_dir / f"{label}{META_SUFFIX}")
            kept.add(label)
        for f in tier_dir.glob("*.npy"):
            if f.stem not in kept:
                f.unlink()
        for f in tier_dir.glob(f"*{META_SUFFIX}"):
            stem = f.name[: -len(META_SUFFIX)]
            if stem not in kept:
                f.unlink()

    def save(self, directory: Path, parts: tuple[Tier, ...] | None = None) -> None:
        """Persist the bank to ``directory``.

        ``parts`` lets callers skip tiers that haven't changed: e.g. an
        ``append("critical", ...)`` only needs ``parts=("critical",)``,
        avoiding a 500MB rewrite of the (untouched) normal bank on every
        single-image teach. Default = save everything.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = set(parts or ("normal", "critical", "negative"))
        if "normal" in target:
            atomic_save_npy(directory / self.NORMAL_FILE, self.normal)
        if "critical" in target:
            self._save_tier(directory / self.CRITICAL_DIR, self.critical, self.critical_meta)
        if "negative" in target:
            self._save_tier(directory / self.NEGATIVE_DIR, self.negative, self.negative_meta)
        # Legacy single-file layout: drop on first save so we never have two
        # sources of truth for the same tier. Cheap (file presence check).
        for legacy in (self.LEGACY_CRITICAL_FILE, self.LEGACY_NEGATIVE_FILE):
            p = directory / legacy
            if p.exists():
                p.unlink()
        # Meta is tiny (a few KB) — always rewrite so ``n_patches`` etc. stay
        # in sync with whatever tier(s) the caller just modified.
        self.meta.n_patches = int(self.normal.shape[0])
        self.meta.dim = int(self.normal.shape[1]) if self.normal.size else self.meta.dim
        atomic_write_text(directory / self.META_FILE, self.meta.to_json())

    # ---- HITL primitive: append-only -------------------------------------

    def append(
        self,
        tier: Tier,
        features: np.ndarray,
        label: str | None = None,
        image_name: str | None = None,
        severity: int = DEFAULT_SEVERITY,
        max_patches: int = 0,
        device: str = "cpu",
        kept_idx: np.ndarray | None = None,
    ) -> str:
        """Append features to the chosen tier; returns the resolved label.

        ``label`` is ignored for the ``normal`` tier and sanitised to a
        filesystem-safe name for ``critical`` / ``negative``. ``severity``
        is recorded per row of metadata for the labelled tiers (and clipped
        to {1, 2, 3} downstream); it has no effect on the ``normal`` tier
        because nominal samples don't carry an "incident strength". This is
        the 6-second HITL primitive.

        ``max_patches`` (>0) coreset-reduces THIS image's patches to at most
        that many representative rows before adding them — a per-image cap
        that bounds bank growth without ever touching earlier images' rows,
        so the per-image row index stays exact (leave-one-out eval, per-image
        delete and NG marks all keep working). ``device`` is used for the
        k-center selection.

        It applies to the labelled tiers too. Those were exempt on the grounds
        that they are "small", which stopped being true: banks on the dev box
        reached 12.8 M labelled rows — 20 GB resident — against a capped normal
        tier of 219 k (2026-07-31). Their rows really are addressed
        geometrically by the NG marks, so a reduced image records ``kept``
        alongside its row range and :meth:`set_image_annotation` maps through
        it. ``kept_idx`` lets a caller that reduced the features itself supply
        the same map.
        """
        if features.size == 0:
            return safe_label(label)
        features = features.astype(BANK_DTYPE, copy=False)
        if max_patches > 0 and features.shape[0] > max_patches:
            features, idx = coreset_reduce_indexed(
                features, max_patches / float(features.shape[0]), device
            )
            features = features.astype(BANK_DTYPE, copy=False)
            # Compose with any reduction the caller already did.
            kept_idx = idx if kept_idx is None else np.asarray(kept_idx, dtype=np.int64)[idx]
        if tier == "normal":
            start = int(self.normal.shape[0]) if self.normal.size else 0
            count = int(features.shape[0])
            self.normal = (
                np.concatenate([self.normal, features], axis=0) if self.normal.size else features
            )
            # Mirror the critical / negative paths: log the filename so the
            # Images tab can list every OK image the operator dropped in
            # without re-walking the bank's patch features. The
            # ``(name, start, count)`` index lets the Images tab's delete
            # action prune exactly the matching rows back out later.
            if image_name:
                if image_name not in self.meta.bank_images:
                    self.meta.bank_images.append(image_name)
                self.meta.normal_image_index.append(
                    {"name": image_name, "start": start, "count": count}
                )
            return ""
        if tier in ("critical", "negative"):
            tgt = self.critical if tier == "critical" else self.negative
            tgt_meta = self.critical_meta if tier == "critical" else self.negative_meta
            log = self.meta.critical_images if tier == "critical" else self.meta.negative_images
            row_index = (
                self.meta.critical_image_index if tier == "critical"
                else self.meta.negative_image_index
            )
            lab = safe_label(label)
            existing = tgt.get(lab)
            start = int(existing.shape[0]) if existing is not None and existing.size else 0
            count = int(features.shape[0])
            tgt[lab] = (
                np.concatenate([existing, features], axis=0)
                if existing is not None and existing.size
                else features
            )
            # Keep parallel metadata in lockstep with the feature array; if
            # this label is brand-new, start with an empty metadata array.
            tgt_meta.setdefault(lab, IncidentMetaArray.empty()).append(
                int(features.shape[0]),
                severity=severity,
                inspection_count=self.meta.inspection_count,
            )
            if image_name:
                names = log.setdefault(lab, [])
                if image_name not in names:
                    names.append(image_name)
                # Same row-range index as the normal tier; lets the
                # Images tab's delete prune exactly these rows later.
                entry: dict = {"name": image_name, "start": start, "count": count}
                # Only recorded when the image was actually reduced. Its absence
                # means "row i is patch i of the grid", which is what every bank
                # written before the labelled tiers were capped relies on.
                if kept_idx is not None and len(kept_idx) == count:
                    entry["kept"] = [int(v) for v in kept_idx]
                row_index.setdefault(lab, []).append(entry)
            return lab
        raise ValueError(f"unknown tier: {tier!r}")

    def set_image_annotation(
        self,
        tier: Tier,
        label: str | None,
        name: str,
        rows: list[int],
        severity: int = SEVERITY_HEAVY,
        rects: list[dict] | None = None,
    ) -> int:
        """Mark ``rows`` (local, 0-based within the image's slice) as defect exemplars.

        Replace semantics per image: the image's whole severity slice is
        reset to ``DEFAULT_SEVERITY`` first, then the given rows are set to
        ``severity`` — re-annotating never leaves stale marks behind. The
        source rectangles are stored on the image's row-range index entry
        (``annotations`` key) purely so a UI can re-display and edit them;
        the rows themselves are the scoring-facing artifact. An empty
        ``rows``/``rects`` pair clears the annotation entirely.

        Returns the number of rows actually marked. Raises ``KeyError`` when
        the image has no row-range index entry (legacy bank) and
        ``ValueError`` for the normal tier, which has no per-row metadata.

        ``label`` is the *resolved on-disk* label (e.g. ``"_default"``), as
        reported by the images listing — not re-sanitised here, because
        ``safe_label`` strips leading underscores and would turn
        ``"_default"`` into a key that doesn't exist.
        """
        if tier == "normal":
            raise ValueError("normal tier has no per-row metadata to annotate")
        lab = label or DEFAULT_LABEL
        row_index = (
            self.meta.critical_image_index if tier == "critical"
            else self.meta.negative_image_index
        )
        metas_d = self.critical_meta if tier == "critical" else self.negative_meta
        entry = next(
            (e for e in row_index.get(lab, []) if str(e.get("name", "")) == name), None,
        )
        meta_arr = metas_d.get(lab)
        if entry is None or meta_arr is None:
            raise KeyError(f"image {name!r} has no indexed rows under {tier}/{lab}")
        start, count = int(entry.get("start", -1)), int(entry.get("count", 0))
        if start < 0 or count <= 0 or start + count > len(meta_arr):
            raise KeyError(f"row range of {name!r} is out of bounds — legacy bank entry")
        sev = int(np.clip(int(severity), SEVERITY_LIGHT, SEVERITY_HEAVY))
        meta_arr.severity[start : start + count] = DEFAULT_SEVERITY
        # `rows` addresses the image's PATCH GRID. That is the same thing as a
        # position in the bank slice only while the image kept every patch; a
        # coreset-reduced image records which grid patches survived, and the
        # marks have to be translated through it or they land on unrelated
        # rows. No `kept` means the image was not reduced (or predates the
        # labelled-tier cap) and the two are identical.
        kept = entry.get("kept")
        if kept:
            where = {int(g): i for i, g in enumerate(kept)}
            valid = sorted({where[int(r)] for r in rows if int(r) in where})
        else:
            valid = sorted({int(r) for r in rows if 0 <= int(r) < count})
        if valid:
            meta_arr.severity[np.asarray(valid, dtype=np.int64) + start] = sev
            entry["annotations"] = list(rects or [])
        else:
            entry.pop("annotations", None)
        return len(valid)

    def save_meta_only(self, directory: Path, tier: Tier) -> None:
        """Persist one labelled tier's ``.meta.npz`` files plus the manifest.

        Annotation only touches per-row severity and the JSON index — the
        feature arrays are untouched, and ``save()`` would rewrite them
        (multi-GB for a large tier). This writes just the small sidecars.
        """
        if tier == "normal":
            raise ValueError("normal tier has no per-row metadata to save")
        directory = Path(directory)
        subdir = self.CRITICAL_DIR if tier == "critical" else self.NEGATIVE_DIR
        metas_d = self.critical_meta if tier == "critical" else self.negative_meta
        for label, meta_arr in metas_d.items():
            if len(meta_arr):
                meta_arr.save(directory / subdir / f"{label}{META_SUFFIX}")
        atomic_write_text(directory / self.META_FILE, self.meta.to_json())

    def tick(self) -> int:
        """Advance the inspection counter by one and return the new value.

        Called once per inference (by API/CLI, not by ``scoring``) so the
        scoring functions stay pure and deterministic for benchmarks. The
        new value becomes the timestamp for any subsequent ``append`` and
        the reference point for freshness decay (Phase 1b/c).
        """
        self.meta.inspection_count = int(self.meta.inspection_count) + 1
        return self.meta.inspection_count

    def hit(self, tier: Tier, label: str, indices: np.ndarray) -> int:
        """Mark the given bank rows as freshly re-encountered.

        ``indices`` is an array of row positions inside
        ``self.{critical,negative}[label]`` whose ``last_hit_at_inspection``
        should be set to the current ``inspection_count`` and whose
        ``hit_count`` should be incremented by one. Returns the number of
        rows actually updated (``0`` if the label is unknown or empty).

        Called per inference by the API after collecting argmin from
        scoring; scoring itself stays pure. Indices that are out of range
        are silently dropped — this keeps the call site simple in the
        face of label clears that race against an in-flight score.
        """
        if tier == "normal":
            raise ValueError("normal tier has no per-row metadata to hit")
        meta = (self.critical_meta if tier == "critical" else self.negative_meta).get(label)
        if meta is None or len(meta) == 0 or indices.size == 0:
            return 0
        idx = np.asarray(indices, dtype=np.int64).ravel()
        idx = idx[(idx >= 0) & (idx < len(meta))]
        if idx.size == 0:
            return 0
        # ``np.unique`` is the ergonomic way to dedupe an argmin vector
        # before incrementing hit_count; without it the same row could
        # be counted multiple times per call when several queries map
        # to the same nearest-neighbour bank row.
        idx = np.unique(idx)
        meta.last_hit_at_inspection[idx] = self.meta.inspection_count
        meta.hit_count[idx] += 1
        return int(idx.size)

    def decay(
        self,
        dry_run: bool = False,
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Promote / retire per-row metadata based on hit_count + freshness.

        Pure offline maintenance call (typically nightly): walks every
        labelled tier and applies the consolidation rules from
        ``incident``:

            - hit_count >= PROMOTE_TO_MID_HITS  -> tier short  becomes mid
            - hit_count >= PROMOTE_TO_LONG_HITS -> tier mid    becomes long
            - tier == short AND freshness < DEATH_FRESHNESS    -> retired
            - tier in DEATH_PROTECTED_TIERS                    -> never retired

        With ``dry_run=True`` the bank is left untouched and the same
        per-label counters are returned so an operator can preview the
        effect of the batch before applying it.

        Returns a nested summary keyed as
        ``{tier_name: {label: {"promoted_to_mid": n, "promoted_to_long": n,
        "retired": n}}}`` so the API can surface a one-line "decay run:
        +5 mid, +1 long, -12 retired" status to the UI.
        """
        now = int(self.meta.inspection_count)
        summary: dict[str, dict[str, dict[str, int]]] = {"critical": {}, "negative": {}}

        for tier_name in ("critical", "negative"):
            features_d = self.critical if tier_name == "critical" else self.negative
            metas_d = self.critical_meta if tier_name == "critical" else self.negative_meta
            for label in list(features_d.keys()):
                meta = metas_d.get(label)
                if meta is None or len(meta) == 0:
                    continue

                # Promotion is a pure tier rewrite — no rows are removed,
                # so it can be applied in-place without touching features.
                promote_mid_mask = (meta.tier == TIER_SHORT) & (
                    meta.hit_count >= PROMOTE_TO_MID_HITS
                )
                promote_long_mask = (meta.tier == TIER_MID) & (
                    meta.hit_count >= PROMOTE_TO_LONG_HITS
                )
                # Retirement is short-tier-only; mid/long are protected.
                fresh = freshness(meta.last_hit_at_inspection, meta.tier, now)
                retire_mask = (meta.tier == TIER_SHORT) & (fresh < DEATH_FRESHNESS)
                # Defensive: never retire anything in a protected tier even
                # if a future bug somehow flips a protected row into ``short``.
                for protected in DEATH_PROTECTED_TIERS:
                    retire_mask &= meta.tier != protected

                summary[tier_name][label] = {
                    "promoted_to_mid": int(promote_mid_mask.sum()),
                    "promoted_to_long": int(promote_long_mask.sum()),
                    "retired": int(retire_mask.sum()),
                }

                if dry_run:
                    continue

                # Apply promotions first (cheap), then drop retired rows.
                if promote_mid_mask.any():
                    meta.tier[promote_mid_mask] = TIER_MID
                if promote_long_mask.any():
                    meta.tier[promote_long_mask] = TIER_LONG
                if retire_mask.any():
                    keep = ~retire_mask
                    features_d[label] = features_d[label][keep]
                    metas_d[label] = meta.take(keep)
                # An empty array after retirement should disappear so a
                # subsequent ``save`` removes the on-disk files; otherwise
                # the bank carries dead labels around forever.
                if features_d[label].shape[0] == 0:
                    features_d.pop(label, None)
                    metas_d.pop(label, None)
                    log = (
                        self.meta.critical_images
                        if tier_name == "critical"
                        else self.meta.negative_images
                    )
                    log.pop(label, None)

        return summary

    def remove_images(
        self, tier: Tier, label: str | None, names: list[str],
    ) -> dict[str, int]:
        """Prune all patches that came from ``names`` out of the bank.

        Uses ``BankMeta.*_image_index`` to map each filename to its row
        range (``[start, start+count)``) inside the underlying tier
        array. Per-image deletes have to:

        1. Slice those rows out of the feature array.
        2. Slice the matching rows out of the IncidentMetaArray for
           labelled tiers (severity / freshness / tier / hit_count).
        3. Shift the start of every later index entry down by the
           removed count (because the rows above moved up).
        4. Drop the entry from the filename log (``bank_images`` /
           ``*_images[label]``) so the Images tab forgets it too.

        Filenames with no matching index entry are silently skipped
        (legacy bank rows pre-0.2 don't have an index — for those the
        caller falls back to "remove file + drop log entry only" and
        accepts that the patches stay in the bank).

        ``label`` is the *resolved on-disk* label (e.g. ``"_default"``), as
        reported by the images listing — not re-sanitised here, because
        ``safe_label`` strips leading underscores and would turn
        ``"_default"`` into a key that doesn't exist (same contract as
        ``set_image_annotation``).

        Returns ``{"rows_removed": N, "names_removed": M}`` so the
        caller can log the action.
        """
        if tier == "normal":
            entries = self.meta.normal_image_index
            arr = self.normal
            meta_arr = None
            log = self.meta.bank_images
        else:
            row_index = (
                self.meta.critical_image_index if tier == "critical"
                else self.meta.negative_image_index
            )
            features_d = self.critical if tier == "critical" else self.negative
            metas_d = self.critical_meta if tier == "critical" else self.negative_meta
            log_d = (
                self.meta.critical_images if tier == "critical"
                else self.meta.negative_images
            )
            lab = label or DEFAULT_LABEL
            entries = row_index.get(lab, [])
            arr = features_d.get(lab, np.zeros((0, 0), dtype=np.float32))
            meta_arr = metas_d.get(lab)
            log = log_d.get(lab, [])

        target_names = set(names)
        # Collect the row slices we're removing, keep the kept entries.
        to_drop_indices: list[int] = []
        for i, e in enumerate(entries):
            if e.get("name") in target_names:
                to_drop_indices.append(i)
        if not to_drop_indices:
            return {"rows_removed": 0, "names_removed": 0}

        # Build a boolean mask of rows-to-keep for the feature array.
        keep_mask = np.ones(int(arr.shape[0]), dtype=bool) if arr.size else np.zeros(0, dtype=bool)
        rows_removed = 0
        for i in to_drop_indices:
            e = entries[i]
            s, c = int(e.get("start", 0)), int(e.get("count", 0))
            if s + c <= keep_mask.shape[0]:
                keep_mask[s : s + c] = False
                rows_removed += c

        # Apply to feature array + (for labelled tiers) IncidentMetaArray.
        new_arr = arr[keep_mask] if arr.size else arr
        if meta_arr is not None and len(meta_arr) == arr.shape[0]:
            new_meta_arr = meta_arr.take(keep_mask)
        else:
            new_meta_arr = meta_arr  # length mismatch: legacy, leave alone

        # Rebuild the index with kept entries, shifting ``start`` to
        # reflect the new row positions in the compacted array.
        #
        # Copy the WHOLE entry. Rebuilding it from three keys silently threw
        # away every other image's ``kept`` map (which is what makes a capped
        # labelled tier able to place a row back on the source grid) and its
        # ``annotations`` rectangles — so deleting one image erased the
        # operator's defect marks on all the survivors. Only ``start`` moves;
        # ``kept`` and ``annotations`` are local to their own image and are
        # unaffected by rows removed above them.
        new_entries: list[dict] = []
        cursor = 0
        for i, e in enumerate(entries):
            if i in to_drop_indices:
                continue
            count = int(e.get("count", 0))
            new_entries.append({**e, "start": cursor, "count": count})
            cursor += count

        # Write everything back.
        if tier == "normal":
            self.normal = new_arr.astype(BANK_DTYPE, copy=False) if new_arr.size else np.zeros(
                (0, int(self.meta.dim) if self.meta.dim else 0), dtype=BANK_DTYPE,
            )
            self.meta.normal_image_index = new_entries
            self.meta.bank_images = [n for n in log if n not in target_names]
        else:
            features_d[lab] = new_arr.astype(BANK_DTYPE, copy=False) if new_arr.size else np.zeros(
                (0, int(self.meta.dim) if self.meta.dim else 0), dtype=BANK_DTYPE,
            )
            if new_meta_arr is not None:
                metas_d[lab] = new_meta_arr
            row_index[lab] = new_entries
            log_d[lab] = [n for n in log if n not in target_names]
            # If the label is now completely empty, drop it everywhere.
            if features_d[lab].shape[0] == 0:
                features_d.pop(lab, None)
                metas_d.pop(lab, None)
                row_index.pop(lab, None)
                log_d.pop(lab, None)

        return {"rows_removed": rows_removed, "names_removed": len(to_drop_indices)}

    def clear_label(self, tier: Tier, label: str) -> None:
        """Drop one label from a tier. Use ``clear_tier`` to wipe everything."""
        if tier == "normal":
            raise ValueError("normal tier is not labeled")
        tgt = self.critical if tier == "critical" else self.negative
        tgt_meta = self.critical_meta if tier == "critical" else self.negative_meta
        log = self.meta.critical_images if tier == "critical" else self.meta.negative_images
        row_index = (
            self.meta.critical_image_index if tier == "critical"
            else self.meta.negative_image_index
        )
        tgt.pop(label, None)
        tgt_meta.pop(label, None)
        log.pop(label, None)
        row_index.pop(label, None)

    def clear_tier(self, tier: Tier) -> None:
        # The row index goes with the rows. Wiping the arrays and the filename
        # log while leaving ``*_image_index`` populated left one phantom per
        # cleared image: /bank/images lists them (it reads the index, not the
        # log), the viewer 404s on files the clear had already unlinked, and a
        # later per-image delete would slice a row range out of an array that
        # no longer has those rows.
        if tier == "normal":
            raise ValueError("normal tier is protected")
        if tier == "critical":
            self.critical.clear()
            self.critical_meta.clear()
            self.meta.critical_images.clear()
            self.meta.critical_image_index.clear()
        else:
            self.negative.clear()
            self.negative_meta.clear()
            self.meta.negative_images.clear()
            self.meta.negative_image_index.clear()

    # ---- size summaries --------------------------------------------------

    def tier_size(self, tier: Tier) -> int:
        if tier == "normal":
            return int(self.normal.shape[0])
        items = self.critical if tier == "critical" else self.negative
        return sum(int(arr.shape[0]) for arr in items.values() if arr is not None)

    def label_sizes(self, tier: Tier) -> dict[str, int]:
        if tier == "normal":
            return {}
        items = self.critical if tier == "critical" else self.negative
        return {
            lab: int(arr.shape[0])
            for lab, arr in items.items()
            if arr is not None and arr.size
        }

    # ---- tensor view for inference ---------------------------------------

    def to_tensors(
        self,
        device: str,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Returns (normal, critical_by_label, negative_by_label) on ``device``.

        Empty labels are dropped so callers can iterate without size checks.
        """
        n_t = torch.from_numpy(self.normal).to(device, dtype=dtype)
        c_d = {
            lab: torch.from_numpy(arr).to(device, dtype=dtype)
            for lab, arr in self.critical.items()
            if arr is not None and arr.size
        }
        g_d = {
            lab: torch.from_numpy(arr).to(device, dtype=dtype)
            for lab, arr in self.negative.items()
            if arr is not None and arr.size
        }
        return n_t, c_d, g_d

    # ---- size summary ----------------------------------------------------

    def __repr__(self) -> str:
        n0 = int(self.normal.shape[0])
        n1 = self.tier_size("critical")
        n2 = self.tier_size("negative")
        d = int(self.normal.shape[1]) if self.normal.size else 0
        ic = int(self.meta.inspection_count)
        return f"Bank(normal={n0}, critical={n1}, negative={n2}, dim={d}, inspections={ic})"


def _as_bank_dtype_dict(d: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    if not d:
        return {}
    return {lab: arr.astype(BANK_DTYPE, copy=False) for lab, arr in d.items() if arr is not None}


def _ensure_meta_dict(
    features: dict[str, np.ndarray],
    provided: dict[str, IncidentMetaArray] | None,
    registered_at: int,
) -> dict[str, IncidentMetaArray]:
    """Validate caller-provided metadata against features, or synthesise defaults.

    The Bank's invariant is that every feature label has a metadata entry
    of the same length. Callers can pass ``None`` (we synthesise defaults
    sized to each feature array) or a partial dict (we fill in the missing
    labels with defaults). A length mismatch between a provided entry and
    its feature array is fatal — see ``IncidentMetaArray.assert_matches``.
    """
    out: dict[str, IncidentMetaArray] = {}
    for label, arr in features.items():
        n = int(arr.shape[0])
        if provided is not None and label in provided:
            m = provided[label]
            m.assert_matches(n, label)
            out[label] = m
        else:
            out[label] = IncidentMetaArray.defaults_for(n, registered_at=registered_at)
    return out
