# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Phase 1a tests: time-aware metadata is plumbed through Bank without
changing scoring behaviour. The contract for Phase 1a is:

    1. Existing legacy banks (no .meta.npz files) load and round-trip
       without losing or altering a single feature byte.
    2. ``Bank.append`` keeps per-row metadata in lockstep with the feature
       array for every label.
    3. ``Bank.tick`` advances the inspection counter that ``append`` stamps
       onto new rows.
    4. ``Bank.clear_label`` / ``clear_tier`` drop the matching metadata
       (both in-memory and on disk after save) so a stale ``.meta.npz``
       can never resurrect a cleared label.

Phase 1a deliberately does NOT modify scoring.py, so AUROC stays bit-exact
on existing benchmarks; we don't need a separate scoring regression test
here as long as the feature arrays themselves are preserved.
"""

from __future__ import annotations

import numpy as np

from clscore.bank import Bank, BankMeta
from clscore.incident import (
    DEFAULT_SEVERITY,
    DEFAULT_TIER,
    SEVERITY_HEAVY,
    SEVERITY_LIGHT,
    IncidentMetaArray,
)

# ---- helpers ----------------------------------------------------------------


def _rand_features(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def _legacy_bank_on_disk(directory, normal=10, critical_rows=3, negative_rows=2):
    """Write a bank that mimics the pre-time-aware layout (no .meta.npz).

    Used to verify that legacy banks load cleanly with synthesised default
    metadata and can be saved back without surprises.
    """
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "bank.npy", _rand_features(normal, seed=1))
    if critical_rows:
        (directory / "critical").mkdir(exist_ok=True)
        np.save(directory / "critical" / "scratch.npy", _rand_features(critical_rows, seed=2))
    if negative_rows:
        (directory / "negative").mkdir(exist_ok=True)
        np.save(directory / "negative" / "_default.npy", _rand_features(negative_rows, seed=3))
    meta = BankMeta(dim=16, n_patches=normal)
    (directory / "bank_meta.json").write_text(meta.to_json(), encoding="utf-8")


# ---- legacy load: features preserved, defaults synthesised ------------------


def test_legacy_bank_loads_with_default_metadata(tmp_path):
    _legacy_bank_on_disk(tmp_path)
    b = Bank.load(tmp_path)
    # Features unchanged on the round-trip.
    assert b.normal.shape == (10, 16)
    assert b.tier_size("critical") == 3
    assert b.tier_size("negative") == 2
    # Metadata synthesised at default severity / tier, lengths match features.
    assert (b.critical_meta["scratch"].severity == DEFAULT_SEVERITY).all()
    assert (b.critical_meta["scratch"].tier == DEFAULT_TIER).all()
    assert len(b.critical_meta["scratch"]) == 3
    assert len(b.negative_meta["_default"]) == 2


def test_legacy_bank_save_round_trip_preserves_features(tmp_path):
    _legacy_bank_on_disk(tmp_path)
    b = Bank.load(tmp_path)
    crit_before = b.critical["scratch"].copy()
    neg_before = b.negative["_default"].copy()
    normal_before = b.normal.copy()
    b.save(tmp_path)
    b2 = Bank.load(tmp_path)
    np.testing.assert_array_equal(b2.normal, normal_before)
    np.testing.assert_array_equal(b2.critical["scratch"], crit_before)
    np.testing.assert_array_equal(b2.negative["_default"], neg_before)


def test_legacy_save_writes_meta_npz_siblings(tmp_path):
    _legacy_bank_on_disk(tmp_path)
    b = Bank.load(tmp_path)
    b.save(tmp_path)
    assert (tmp_path / "critical" / "scratch.meta.npz").exists()
    assert (tmp_path / "negative" / "_default.meta.npz").exists()


# ---- append plumbs severity through to metadata -----------------------------


def test_append_records_severity_per_row(tmp_path):
    b = Bank(normal=np.zeros((0, 16), dtype=np.float32))
    feats = _rand_features(4, dim=16, seed=10)
    b.append("critical", feats, label="dent", severity=SEVERITY_HEAVY)
    m = b.critical_meta["dent"]
    assert len(m) == 4
    assert (m.severity == SEVERITY_HEAVY).all()


def test_append_uses_default_severity_when_unspecified():
    b = Bank(normal=np.zeros((0, 16), dtype=np.float32))
    b.append("critical", _rand_features(2), label="x")
    assert (b.critical_meta["x"].severity == DEFAULT_SEVERITY).all()


def test_append_stamps_current_inspection_count_on_new_rows():
    b = Bank(normal=np.zeros((0, 16), dtype=np.float32))
    # Advance the counter twice so the next append should record 2.
    b.tick()
    b.tick()
    b.append("critical", _rand_features(3), label="x")
    m = b.critical_meta["x"]
    assert (m.registered_at_inspection == 2).all()
    assert (m.last_hit_at_inspection == 2).all()


def test_normal_tier_is_not_metadata_tracked():
    b = Bank(normal=np.zeros((0, 16), dtype=np.float32))
    feats = _rand_features(5)
    label = b.append("normal", feats, severity=SEVERITY_HEAVY)
    # Normal is a single tensor with no per-row metadata; severity is ignored.
    assert label == ""
    assert b.normal.shape == (5, 16)


# ---- tick advances and is persisted ----------------------------------------


def test_tick_advances_inspection_count_monotonically():
    b = Bank(normal=np.zeros((0, 16), dtype=np.float32))
    assert b.meta.inspection_count == 0
    assert b.tick() == 1
    assert b.tick() == 2
    assert b.meta.inspection_count == 2


def test_inspection_count_persists_across_save_load(tmp_path):
    b = Bank(normal=_rand_features(4))
    for _ in range(7):
        b.tick()
    b.save(tmp_path)
    b2 = Bank.load(tmp_path)
    assert b2.meta.inspection_count == 7


# ---- clear drops metadata in lockstep --------------------------------------


def test_clear_label_drops_metadata(tmp_path):
    b = Bank(normal=np.zeros((0, 16), dtype=np.float32))
    b.append("critical", _rand_features(3), label="a", severity=SEVERITY_HEAVY)
    b.append("critical", _rand_features(2), label="b", severity=SEVERITY_LIGHT)
    b.clear_label("critical", "a")
    assert "a" not in b.critical
    assert "a" not in b.critical_meta
    assert "b" in b.critical_meta


def test_clear_tier_clears_all_metadata():
    b = Bank(normal=np.zeros((0, 16), dtype=np.float32))
    b.append("critical", _rand_features(3), label="a")
    b.append("critical", _rand_features(2), label="b")
    b.clear_tier("critical")
    assert b.critical == {}
    assert b.critical_meta == {}


def test_save_after_clear_drops_stale_meta_npz(tmp_path):
    b = Bank(normal=_rand_features(4))
    b.append("critical", _rand_features(3), label="oops")
    b.save(tmp_path)
    assert (tmp_path / "critical" / "oops.meta.npz").exists()
    b.clear_label("critical", "oops")
    b.save(tmp_path)
    assert not (tmp_path / "critical" / "oops.npy").exists()
    assert not (tmp_path / "critical" / "oops.meta.npz").exists()


# ---- corruption defence: metadata length must match features ---------------


def test_load_repairs_meta_when_length_disagrees(tmp_path):
    # Build a legitimate bank, then surgically corrupt a meta file by
    # truncating the severity column (a crash between the .npy and .meta.npz
    # saves produces exactly this). Load used to refuse — bricking the whole
    # bank with no repair path — so it now rebuilds default metadata for the
    # damaged label (severity marks reset, warned loudly) and keeps every
    # taught feature row loadable.
    b = Bank(normal=_rand_features(4))
    b.append("critical", _rand_features(3), label="x")
    b.save(tmp_path)
    bad = IncidentMetaArray.defaults_for(2)  # wrong length: 2 instead of 3
    bad.save(tmp_path / "critical" / "x.meta.npz")
    loaded = Bank.load(tmp_path)
    assert int(loaded.critical["x"].shape[0]) == 3, "feature rows must survive"
    assert len(loaded.critical_meta["x"]) == 3, "metadata rebuilt to match features"


# ---- repr exposes inspection counter for debugging --------------------------


def test_repr_includes_inspection_count():
    b = Bank(normal=_rand_features(4))
    b.tick()
    b.tick()
    assert "inspections=2" in repr(b)
