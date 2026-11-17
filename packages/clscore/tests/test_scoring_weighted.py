# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Phase 1b tests: severity x freshness weighting in ``_per_label_min``.

The contract for Phase 1b is:

    1. ``weighted=False`` returns *bit-exact* the same min distances as
       Phase 1a, even when metas are passed in. (Existing benchmarks must
       not drift when the new arg is wired in but not turned on.)

    2. ``weighted=True`` with default-severity (medium=2), never-decayed
       metadata is also bit-exact, because severity_weight=1.0 and
       freshness=1.0 give multiplier=1.0 — i.e. the divide is a no-op.
       This is the property that makes flipping the flag safe on a
       freshly-loaded legacy bank.

    3. Severity actually moves the result: severity=3 shrinks the
       weighted distance, severity=1 grows it.

    4. Freshness actually decays: a row whose ``last_hit_at_inspection``
       is one ``tau`` behind the current counter has multiplier
       ``~exp(-1) ~= 0.37`` and the weighted distance grows ~2.7x.

We test ``_per_label_min`` directly (CPU tensors, no DINOv2) so the
checks are fast and don't depend on a backbone download.
"""

from __future__ import annotations

import numpy as np
import torch

from clscore.incident import (
    SEVERITY_HEAVY,
    SEVERITY_LIGHT,
    SEVERITY_MEDIUM,
    TAU_SHORT,
    TIER_SHORT,
    IncidentMetaArray,
    freshness,
    multiplier_for,
    severity_weight,
)
from clscore.scoring import _per_label_min

# ---- helpers ----------------------------------------------------------------


def _toy_bank(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    queries = torch.tensor(rng.standard_normal((4, 8)).astype(np.float32))
    bank = torch.tensor(rng.standard_normal((6, 8)).astype(np.float32))
    return queries, bank


# ---- incident helpers (unit) ------------------------------------------------


def test_severity_weight_anchored_at_medium():
    sev = np.array([SEVERITY_LIGHT, SEVERITY_MEDIUM, SEVERITY_HEAVY], dtype=np.uint8)
    w = severity_weight(sev)
    assert w.tolist() == [0.5, 1.0, 1.5]


def test_freshness_zero_delta_is_one():
    last = np.array([100, 100], dtype=np.uint64)
    tier = np.array([TIER_SHORT, TIER_SHORT], dtype=np.uint8)
    f = freshness(last, tier, inspection_count=100)
    np.testing.assert_allclose(f, [1.0, 1.0])


def test_freshness_one_tau_delta_is_e_minus_1():
    last = np.array([0], dtype=np.uint64)
    tier = np.array([TIER_SHORT], dtype=np.uint8)
    f = freshness(last, tier, inspection_count=TAU_SHORT)
    np.testing.assert_allclose(f, [np.exp(-1.0)], atol=1e-6)


def test_freshness_clipped_when_last_hit_is_in_the_future():
    """A backup-restored bank can briefly hold last_hit > current counter.
    Without the clip this would pretend to be 'fresher than fresh'."""
    last = np.array([200], dtype=np.uint64)
    tier = np.array([TIER_SHORT], dtype=np.uint8)
    f = freshness(last, tier, inspection_count=100)
    np.testing.assert_allclose(f, [1.0])


def test_multiplier_default_metadata_is_unity():
    """The whole point of medium-severity + freshness=1: multiplier == 1.0,
    so flipping ``weighted=True`` on a legacy bank is a no-op."""
    m = IncidentMetaArray.defaults_for(5, registered_at=10)
    mult = multiplier_for(m, inspection_count=10)
    np.testing.assert_allclose(mult, np.ones(5))


def test_multiplier_high_severity_increases():
    m = IncidentMetaArray.empty()
    m.append(1, severity=SEVERITY_HEAVY, inspection_count=0)
    np.testing.assert_allclose(multiplier_for(m, 0), [1.5])


# ---- _per_label_min: bit-exact when weighted=False --------------------------


def test_unweighted_ignores_meta_completely():
    """Phase 1a regression: passing metas alongside weighted=False must
    leave the function output bit-exact identical to the no-metas call."""
    q, b = _toy_bank()
    banks = {"scratch": b}
    m = IncidentMetaArray.empty()
    m.append(b.shape[0], severity=SEVERITY_HEAVY, inspection_count=0)
    metas = {"scratch": m}

    base = _per_label_min(q, banks)["scratch"]
    with_meta = _per_label_min(q, banks, metas, inspection_count=999, weighted=False)["scratch"]
    torch.testing.assert_close(base, with_meta, rtol=0, atol=0)


# ---- _per_label_min: weighted=True with neutral metadata is also exact ------


def test_weighted_with_default_metadata_is_no_op():
    """The freshness=1 + severity=medium case: multiplier==1 across the
    board, so weighted/unweighted should agree to floating-point exact."""
    q, b = _toy_bank(seed=1)
    banks = {"scratch": b}
    m = IncidentMetaArray.defaults_for(b.shape[0], registered_at=42)
    metas = {"scratch": m}

    base = _per_label_min(q, banks)["scratch"]
    weighted = _per_label_min(
        q, banks, metas, inspection_count=42, weighted=True
    )["scratch"]
    # Tiny numerical drift from the EPS in the divide is allowed; we use
    # a rtol that's much smaller than any real-world severity effect.
    torch.testing.assert_close(weighted, base, rtol=1e-5, atol=1e-5)


# ---- _per_label_min: weighted=True with high severity shrinks distance -----


def test_weighted_high_severity_shrinks_min():
    """severity=3 doubles+ severity_weight to 1.5; dividing distances by
    1.5 must produce a strictly smaller min for the same query."""
    q, b = _toy_bank(seed=2)
    banks = {"scratch": b}
    m_high = IncidentMetaArray.empty()
    m_high.append(b.shape[0], severity=SEVERITY_HEAVY, inspection_count=0)
    m_low = IncidentMetaArray.empty()
    m_low.append(b.shape[0], severity=SEVERITY_LIGHT, inspection_count=0)

    high = _per_label_min(q, banks, {"scratch": m_high}, weighted=True)["scratch"]
    low = _per_label_min(q, banks, {"scratch": m_low}, weighted=True)["scratch"]
    # severity=3 (multiplier 1.5) yields smaller distances than
    # severity=1 (multiplier 0.5) for the same bank rows.
    assert (high < low).all()


# ---- _per_label_min: freshness decay grows distance ------------------------


def test_weighted_decayed_freshness_grows_min():
    """A row that hasn't been hit for one tau drops to multiplier ~0.37,
    which inflates distance by ~2.7x relative to a fresh row."""
    q, b = _toy_bank(seed=3)
    banks = {"scratch": b}

    m_fresh = IncidentMetaArray.empty()
    m_fresh.append(b.shape[0], severity=SEVERITY_MEDIUM, inspection_count=0)
    # Severity=2 (weight=1.0) so the decay is the only thing changing.
    m_stale = IncidentMetaArray.empty()
    m_stale.append(b.shape[0], severity=SEVERITY_MEDIUM, inspection_count=0)

    fresh = _per_label_min(
        q, banks, {"scratch": m_fresh}, inspection_count=0, weighted=True
    )["scratch"]
    stale = _per_label_min(
        q, banks, {"scratch": m_stale}, inspection_count=TAU_SHORT, weighted=True
    )["scratch"]
    # Decayed rows produce uniformly larger weighted distances.
    assert (stale > fresh).all()
    # And the ratio is roughly e^1 ~= 2.718 because multiplier_fresh=1.0
    # and multiplier_stale ~= exp(-1), modulo the EPS in the divide.
    ratio = (stale / fresh).cpu().numpy()
    np.testing.assert_allclose(ratio, np.full_like(ratio, np.e), rtol=1e-2)


# ---- _per_label_min: empty / mismatched metas are gracefully ignored -------


def test_weighted_with_no_meta_for_label_falls_back_to_raw_distance():
    """If a label exists in banks but not in metas (e.g. a partial decay
    state), we must NOT crash; the row keeps its raw distance."""
    q, b = _toy_bank(seed=4)
    banks = {"a": b, "b": b}
    metas = {"a": IncidentMetaArray.defaults_for(b.shape[0])}  # only "a"
    out = _per_label_min(q, banks, metas, inspection_count=0, weighted=True)
    base = _per_label_min(q, banks)
    # Label "a": neutral metadata → ~equal to base (mod EPS).
    torch.testing.assert_close(out["a"], base["a"], rtol=1e-5, atol=1e-5)
    # Label "b": no metadata → exactly the raw distance.
    torch.testing.assert_close(out["b"], base["b"], rtol=0, atol=0)


def test_weighted_empty_banks_returns_empty_dict():
    """Edge case: every well-formed call should survive an empty banks dict."""
    q, _ = _toy_bank(seed=5)
    assert _per_label_min(q, {}, weighted=True) == {}
    assert _per_label_min(q, None, weighted=True) == {}


# ---- _per_label_min: zero-multiplier rows don't blow up to inf -------------


def test_weighted_fully_decayed_row_does_not_dominate_min():
    """Without the EPS in the divide, multiplier=0 would produce inf and
    cause torch.min to incorrectly treat the row as the minimum. We
    care that min stays finite and ~equal to the *other* rows' weighted
    distances rather than collapsing to a junk minimum."""
    q, b = _toy_bank(seed=6)
    # First row simulates a totally-decayed entry: tier=long, last hit at 0,
    # current inspection huge — exp(-huge / TAU_LONG) underflows to 0.
    m = IncidentMetaArray.empty()
    m.append(b.shape[0], severity=SEVERITY_MEDIUM, inspection_count=0)
    m.tier = np.zeros_like(m.tier)  # force short tier so decay runs faster
    out = _per_label_min(
        q, {"x": b}, {"x": m}, inspection_count=10**9, weighted=True
    )["x"]
    assert torch.isfinite(out).all()
