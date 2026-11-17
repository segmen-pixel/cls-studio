# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Migrating an existing bank into store + label set, without re-extracting.

The production bank is 20 GB of features that cost hours of GPU time. The
whole migration is therefore a *carve*: rows are copied out through the row
index the bank already keeps, and the proof that it worked is that
re-assembling reproduces the original arrays byte for byte.
"""

from __future__ import annotations

import numpy as np

from clscore.assemble import assemble_bank, migrate_bank_to_store, roundtrip_diff
from clscore.bank import Bank, BankMeta
from clscore.incident import SEVERITY_HEAVY
from clscore.store import FeatureStore

DIM = 160


def _feats(rng, n):
    return rng.random((n, DIM), dtype=np.float32).astype(np.float16)


def _taught_bank(seed: int = 1) -> Bank:
    """A bank built the way the API builds one: append, then annotate."""
    rng = np.random.default_rng(seed)
    bank = Bank(normal=np.zeros((0, DIM), dtype=np.float16), meta=BankMeta(dim=DIM))
    for i in range(3):
        bank.append("normal", _feats(rng, 20 + i), image_name=f"ok{i}.png")
    bank.append("critical", _feats(rng, 12), label="scratch", image_name="ng0.png")
    bank.append("critical", _feats(rng, 9), label="scratch", image_name="ng1.png")
    bank.append("critical", _feats(rng, 7), label="stain", image_name="ng2.png")
    bank.append("negative", _feats(rng, 5), label="glare", image_name="fp0.png")
    bank.set_image_annotation("critical", "scratch", "ng0.png", rows=[1, 4, 11])
    return bank


def _migrated(tmp_path, bank: Bank):
    store = FeatureStore(tmp_path / "store")
    return migrate_bank_to_store(bank, store, bank_dir=None, read_images=False)


def test_migration_reproduces_the_bank_byte_for_byte(tmp_path):
    bank = _taught_bank()
    store, ls = _migrated(tmp_path, bank)
    # prev_bank is deliberately None: the point is that the LABEL SET alone
    # reconstructs the bank. Passing the original would copy its metadata over
    # and hide any information the migration failed to capture.
    rebuilt = assemble_bank(store, ls, prev_bank=None)
    assert roundtrip_diff(bank, rebuilt) == []


def test_migration_extracts_nothing_and_indexes_every_image(tmp_path):
    bank = _taught_bank()
    store, ls = _migrated(tmp_path, bank)
    assert sorted(e.name for e in store) == [
        "fp0.png", "ng0.png", "ng1.png", "ng2.png", "ok0.png", "ok1.png", "ok2.png",
    ]
    assert store.total_rows() == 20 + 21 + 22 + 12 + 9 + 7 + 5
    assert ls.counts() == {"normal": 3, "critical": 3, "negative": 1}
    assert store.meta.dim == DIM


def test_migration_recovers_defect_marks_as_grid_indices(tmp_path):
    bank = _taught_bank()
    store, ls = _migrated(tmp_path, bank)
    eid = next(e.id for e in store if e.name == "ng0.png")
    assert ls.assignments[eid].marks == [1, 4, 11]
    assert ls.assignments[eid].tier == "critical"
    assert ls.assignments[eid].label == "scratch"
    # And they come back out on the same rows.
    rebuilt = assemble_bank(store, ls)
    sev = rebuilt.critical_meta["scratch"].severity
    assert list(np.flatnonzero(sev == SEVERITY_HEAVY)) == [1, 4, 11]


def test_migrated_store_survives_a_reload(tmp_path):
    bank = _taught_bank()
    store, ls = _migrated(tmp_path, bank)
    ls.save(tmp_path / "labelsets")
    from clscore.labelset import LabelSet

    reloaded_store = FeatureStore.load(tmp_path / "store")
    reloaded_ls = LabelSet.load(tmp_path / "labelsets" / f"{ls.id}.json")
    assert roundtrip_diff(bank, assemble_bank(reloaded_store, reloaded_ls)) == []


def test_capped_image_keeps_its_kept_map_through_migration(tmp_path):
    """A bank whose NG image was coreset-reduced at teach time."""
    rng = np.random.default_rng(5)
    bank = Bank(normal=np.zeros((0, DIM), dtype=np.float16), meta=BankMeta(dim=DIM))
    bank.append("normal", _feats(rng, 10), image_name="ok.png")
    kept = np.asarray([0, 2, 5, 9, 14, 20, 22, 31], dtype=np.int64)
    bank.append(
        "critical", _feats(rng, len(kept)), label="scratch",
        image_name="ng.png", kept_idx=kept,
    )
    # Grid patch 20 is stored row 5.
    bank.set_image_annotation("critical", "scratch", "ng.png", rows=[20])

    store, ls = _migrated(tmp_path, bank)
    eid = next(e.id for e in store if e.name == "ng.png")
    assert store.by_id(eid).kept == [int(v) for v in kept]
    assert ls.assignments[eid].marks == [20]
    assert roundtrip_diff(bank, assemble_bank(store, ls)) == []


def test_rows_with_no_index_entry_are_kept_not_dropped(tmp_path):
    """Banks built before the row index existed still hold real taught data."""
    rng = np.random.default_rng(7)
    legacy = Bank(normal=_feats(rng, 15), meta=BankMeta(dim=DIM))
    legacy.meta.normal_image_index = []  # pre-0.2 layout
    store, ls = _migrated(tmp_path, legacy)
    assert store.total_rows() == 15
    assert [e.name for e in store] == ["__unindexed__normal"]
    rebuilt = assemble_bank(store, ls)
    assert np.array_equal(rebuilt.normal, legacy.normal)


def test_partially_indexed_bank_keeps_the_gap(tmp_path):
    rng = np.random.default_rng(8)
    bank = Bank(normal=np.zeros((0, DIM), dtype=np.float16), meta=BankMeta(dim=DIM))
    bank.append("normal", _feats(rng, 6), image_name="ok.png")
    # Rows appended without going through the index (a legacy remnant).
    bank.normal = np.concatenate([bank.normal, _feats(rng, 4)], axis=0)
    store, ls = _migrated(tmp_path, bank)
    assert sorted(e.name for e in store) == ["__unindexed__normal", "ok.png"]
    assert store.total_rows() == 10
    rebuilt = assemble_bank(store, ls)
    assert rebuilt.normal.shape[0] == 10


def test_incident_history_is_carried_over_when_the_tier_is_unchanged(tmp_path):
    bank = _taught_bank()
    bank.tick()
    bank.hit("critical", "scratch", np.asarray([0, 1, 2]))
    store, ls = _migrated(tmp_path, bank)

    carried = assemble_bank(store, ls, prev_bank=bank)
    assert list(carried.critical_meta["scratch"].hit_count[:3]) == [1, 1, 1]
    assert carried.meta.inspection_count == 1

    # Moving the image to another tier resets it — the history was about
    # being a scratch, and it no longer is one.
    eid = next(e.id for e in store if e.name == "ng0.png")
    ls.assign(eid, "negative", label="scratch")
    moved = assemble_bank(store, ls, prev_bank=bank)
    assert int(moved.negative_meta["scratch"].hit_count.sum()) == 0


def test_roundtrip_diff_reports_a_real_mismatch(tmp_path):
    bank = _taught_bank()
    store, ls = _migrated(tmp_path, bank)
    eid = next(e.id for e in store if e.name == "ok0.png")
    ls.unassign(eid)
    diff = roundtrip_diff(bank, assemble_bank(store, ls))
    assert diff and "normal shape" in diff[0]
