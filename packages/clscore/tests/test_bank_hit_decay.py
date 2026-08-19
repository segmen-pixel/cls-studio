# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Phase 1c tests: ``Bank.hit`` and ``Bank.decay``.

These exercise the consolidation lifecycle without touching scoring:
the API plumbs scoring's argmin into ``Bank.hit`` between Phase 1b and
this layer, but the bank-side bookkeeping (last_hit, hit_count,
promotions, retirements) is independently testable on numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from clscore.bank import Bank
from clscore.incident import (
    DEATH_FRESHNESS,
    PROMOTE_TO_LONG_HITS,
    PROMOTE_TO_MID_HITS,
    SEVERITY_HEAVY,
    TAU_LONG,
    TAU_SHORT,
    TIER_LONG,
    TIER_MID,
    TIER_SHORT,
)


def _rand(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


# ---- hit() basics -----------------------------------------------------------


def test_hit_increments_count_and_refreshes_last_hit():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(3), label="x")
    b.tick()  # inspection_count = 1
    n = b.hit("critical", "x", np.array([0, 2]))
    assert n == 2
    m = b.critical_meta["x"]
    assert m.hit_count.tolist() == [1, 0, 1]
    assert m.last_hit_at_inspection.tolist() == [1, 0, 1]


def test_hit_dedupes_repeated_indices():
    """A single inference can map several queries to the same nearest
    bank row; we count that as one re-encounter, not N."""
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(3), label="x")
    n = b.hit("critical", "x", np.array([1, 1, 1, 1]))
    assert n == 1
    assert b.critical_meta["x"].hit_count.tolist() == [0, 1, 0]


def test_hit_silently_drops_out_of_range_indices():
    """clear_label can race with an in-flight score, so the API may
    submit indices past the current bank size. Ignoring beats raising."""
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(2), label="x")
    n = b.hit("critical", "x", np.array([0, 99, -1]))
    assert n == 1


def test_hit_empty_indices_is_noop():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(2), label="x")
    assert b.hit("critical", "x", np.array([], dtype=np.int64)) == 0


def test_hit_unknown_label_is_noop():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(2), label="x")
    assert b.hit("critical", "nope", np.array([0])) == 0


def test_hit_on_normal_tier_raises():
    """``normal`` carries no per-row metadata; trying to refresh it is
    almost certainly a logic bug, so we raise instead of silently
    dropping the call."""
    b = Bank(normal=_rand(3))
    with pytest.raises(ValueError, match="normal"):
        b.hit("normal", "_default", np.array([0]))


# ---- decay() promotion ------------------------------------------------------


def test_decay_promotes_short_to_mid_at_threshold():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(4), label="x", severity=SEVERITY_HEAVY)
    m = b.critical_meta["x"]
    m.hit_count[:] = [PROMOTE_TO_MID_HITS - 1, PROMOTE_TO_MID_HITS, PROMOTE_TO_MID_HITS + 5, 0]
    summary = b.decay()
    # Indices 1 and 2 met the threshold; 0 and 3 did not.
    assert m.tier.tolist() == [TIER_SHORT, TIER_MID, TIER_MID, TIER_SHORT]
    assert summary["critical"]["x"]["promoted_to_mid"] == 2


def test_decay_promotes_mid_to_long_at_threshold():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(2), label="x")
    m = b.critical_meta["x"]
    m.tier[:] = TIER_MID
    m.hit_count[:] = [PROMOTE_TO_LONG_HITS - 1, PROMOTE_TO_LONG_HITS]
    summary = b.decay()
    assert m.tier.tolist() == [TIER_MID, TIER_LONG]
    assert summary["critical"]["x"]["promoted_to_long"] == 1


def test_decay_dry_run_reports_but_does_not_modify():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(3), label="x")
    m = b.critical_meta["x"]
    m.hit_count[:] = [PROMOTE_TO_MID_HITS, 0, PROMOTE_TO_MID_HITS]
    summary = b.decay(dry_run=True)
    assert summary["critical"]["x"]["promoted_to_mid"] == 2
    # Bank stays untouched.
    assert m.tier.tolist() == [TIER_SHORT, TIER_SHORT, TIER_SHORT]


# ---- decay() retirement -----------------------------------------------------


def test_decay_retires_decayed_short_rows():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(3), label="x")
    # Push two of three rows so far back they fall under the threshold.
    m = b.critical_meta["x"]
    m.last_hit_at_inspection[:] = [0, 0, 1_000_000]  # third row hasn't decayed
    b.meta.inspection_count = TAU_SHORT * 5  # delta = 5*tau -> freshness ~0.0067
    summary = b.decay()
    assert summary["critical"]["x"]["retired"] == 2
    # One survivor (the row with future-ish last_hit).
    assert b.critical["x"].shape[0] == 1
    assert len(b.critical_meta["x"]) == 1


def test_decay_protects_mid_and_long_from_retirement():
    """Even if a long-tier row's freshness falls through the floor it
    shouldn't be retired — long is the safety vault for chronic incidents
    that may not appear for years."""
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(2), label="x")
    m = b.critical_meta["x"]
    m.tier[:] = [TIER_MID, TIER_LONG]
    m.last_hit_at_inspection[:] = 0
    b.meta.inspection_count = TAU_LONG * 100  # absurdly old
    summary = b.decay()
    assert summary["critical"]["x"]["retired"] == 0
    assert b.critical["x"].shape[0] == 2


def test_decay_drops_label_when_all_rows_retired():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(2), label="goner", severity=SEVERITY_HEAVY)
    b.meta.critical_images["goner"] = ["img1.png", "img2.png"]
    b.critical_meta["goner"].last_hit_at_inspection[:] = 0
    b.meta.inspection_count = TAU_SHORT * 10
    summary = b.decay()
    assert summary["critical"]["goner"]["retired"] == 2
    assert "goner" not in b.critical
    assert "goner" not in b.critical_meta
    assert "goner" not in b.meta.critical_images


def test_decay_keeps_features_and_meta_in_lockstep():
    """The retired-mask must be applied identically to features and
    metadata — a mismatch would corrupt the bank such that load() fails
    on the next round-trip."""
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("critical", _rand(5, seed=11), label="x")
    m = b.critical_meta["x"]
    # Set up so exactly indices [1, 3] retire: pin the others to "now"
    # and push 1, 3 far enough into the past that freshness < threshold.
    b.meta.inspection_count = TAU_SHORT * 10
    m.last_hit_at_inspection[:] = b.meta.inspection_count  # all fresh
    m.last_hit_at_inspection[[1, 3]] = 0  # decayed -> retired
    feats_before = b.critical["x"].copy()
    b.decay()
    assert b.critical["x"].shape[0] == 3
    assert len(b.critical_meta["x"]) == 3
    # Surviving rows are exactly indices [0, 2, 4] of the original.
    np.testing.assert_array_equal(b.critical["x"], feats_before[[0, 2, 4]])


# ---- decay() also walks negative tier --------------------------------------


def test_decay_walks_negative_tier_too():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("negative", _rand(2), label="fp")
    b.negative_meta["fp"].hit_count[:] = PROMOTE_TO_MID_HITS
    summary = b.decay()
    assert summary["negative"]["fp"]["promoted_to_mid"] == 2
    assert b.negative_meta["fp"].tier.tolist() == [TIER_MID, TIER_MID]


# ---- end-to-end: full lifecycle round-trip preserves invariants ------------


def test_save_load_after_decay_round_trips(tmp_path):
    b = Bank(normal=_rand(4))
    b.append("critical", _rand(3), label="x", severity=SEVERITY_HEAVY)
    b.critical_meta["x"].hit_count[:] = [PROMOTE_TO_MID_HITS, 0, PROMOTE_TO_MID_HITS]
    b.decay()
    b.save(tmp_path)
    b2 = Bank.load(tmp_path)
    np.testing.assert_array_equal(b2.critical_meta["x"].tier, b.critical_meta["x"].tier)
    np.testing.assert_array_equal(b2.critical_meta["x"].hit_count, b.critical_meta["x"].hit_count)


# ---- _per_label_min_argmin --------------------------------------------------


def test_per_label_min_argmin_matches_per_label_min():
    """The new joint min/argmin path must agree numerically with the
    bit-exact ``_per_label_min`` so we can switch the API over without
    score drift."""
    import torch

    from clscore.scoring import _per_label_min, _per_label_min_argmin

    rng = np.random.default_rng(3)
    q = torch.tensor(rng.standard_normal((4, 8)).astype(np.float32))
    b = torch.tensor(rng.standard_normal((6, 8)).astype(np.float32))
    base = _per_label_min(q, {"x": b})["x"]
    mins, args = _per_label_min_argmin(q, {"x": b})
    torch.testing.assert_close(mins["x"], base, rtol=0, atol=0)
    # Argmin matches what numpy says.
    diffs = q.numpy()[:, None, :] - b.numpy()[None, :, :]
    expected = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)
    np.testing.assert_array_equal(args["x"].numpy(), expected)


def test_per_label_min_argmin_threshold_for_death_freshness():
    """Sanity: DEATH_FRESHNESS=0.05 corresponds to about 3*tau decay."""
    # exp(-3) ~= 0.0498 < 0.05; exp(-2.9) ~= 0.0550 > 0.05
    assert np.exp(-3.0) < DEATH_FRESHNESS < np.exp(-2.9)
