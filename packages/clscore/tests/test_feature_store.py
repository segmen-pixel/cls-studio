# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Feature store + label set: persistence, and the assembly they feed."""

from __future__ import annotations

import numpy as np
import pytest

from clscore.assemble import assemble_bank, per_image_budget
from clscore.bank import Bank
from clscore.incident import DEFAULT_SEVERITY, SEVERITY_HEAVY, SEVERITY_LIGHT
from clscore.labelset import Assignment, LabelSet, list_labelsets, read_active_id, write_active_id
from clscore.store import FeatureStore, StoreMeta

DIM = 160  # > the coreset projection's 128 components, so no sklearn warning


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _store(tmp_path, images: list[tuple[str, int]], seed: int = 0) -> FeatureStore:
    """Store holding one array per ``(name, rows)`` pair, values keyed to the name."""
    st = FeatureStore(tmp_path / "store", meta=StoreMeta(dim=DIM))
    rng = _rng(seed)
    for name, rows in images:
        st.add(rng.random((rows, DIM), dtype=np.float32), name=name, grid_rows=rows)
    st.save_index()
    return st


# ---- store persistence -----------------------------------------------------


def test_store_roundtrips_through_disk(tmp_path):
    st = _store(tmp_path, [("a.png", 7), ("b.png", 3)])
    again = FeatureStore.load(tmp_path / "store")
    assert [e.name for e in again] == ["a.png", "b.png"]
    assert [e.rows for e in again] == [7, 3]
    assert again.meta.dim == DIM
    for e in again:
        assert np.array_equal(again.features_of(e), st.features_of(e.id))


def test_store_ids_are_unique_across_equal_names(tmp_path):
    """Two tiers may hold the same filename; the store must not merge them."""
    st = FeatureStore(tmp_path / "store", meta=StoreMeta(dim=DIM))
    a = st.add(np.ones((2, DIM), dtype=np.float32), name="dup.png")
    b = st.add(np.zeros((2, DIM), dtype=np.float32), name="dup.png")
    assert a.id != b.id
    assert st.features_of(a).sum() == 2 * DIM
    assert st.features_of(b).sum() == 0


def test_store_rejects_mismatched_dim(tmp_path):
    st = FeatureStore(tmp_path / "store", meta=StoreMeta(dim=DIM))
    with pytest.raises(ValueError, match="does not match store dim"):
        st.add(np.ones((2, DIM + 1), dtype=np.float32), name="x.png")


def test_store_rejects_kept_of_wrong_length(tmp_path):
    st = FeatureStore(tmp_path / "store", meta=StoreMeta(dim=DIM))
    with pytest.raises(ValueError, match="kept map"):
        st.add(np.ones((4, DIM), dtype=np.float32), name="x.png", kept=[0, 1])


def test_store_remove_drops_files_and_entries(tmp_path):
    st = _store(tmp_path, [("a.png", 4), ("b.png", 4)])
    victim = st.entries[0].id
    path = st.feature_path(victim)
    assert path.exists()
    assert st.remove([victim, "nope"]) == 1
    assert not path.exists()
    assert [e.name for e in st] == ["b.png"]


def test_missing_array_is_reported_not_silently_empty(tmp_path):
    st = _store(tmp_path, [("a.png", 4)])
    st.feature_path(st.entries[0].id).unlink()
    assert st.missing_arrays() == [st.entries[0].id]
    with pytest.raises(FileNotFoundError):
        assemble_bank(st, LabelSet())


# ---- label set -------------------------------------------------------------


def test_labelset_roundtrips_with_marks(tmp_path):
    ls = LabelSet(id="ls1", name="実験A")
    ls.assign("000000", "critical", label="scratch", severity=SEVERITY_LIGHT)
    ls.mark("000000", [3, 1, 1], rects=[{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}])
    ls.assign("000001", "normal")
    ls.save(tmp_path / "labelsets")

    again = LabelSet.load(tmp_path / "labelsets" / "ls1.json")
    assert again.name == "実験A"
    assert again.assignments["000000"].marks == [1, 3]
    assert again.assignments["000000"].severity == SEVERITY_LIGHT
    assert again.assignments["000000"].rects[0]["w"] == 0.3
    assert again.counts() == {"normal": 1, "critical": 1, "negative": 0}
    assert [x.id for x in list_labelsets(tmp_path / "labelsets")] == ["ls1"]


def test_active_marker_roundtrips(tmp_path):
    assert read_active_id(tmp_path / "labelsets") is None
    write_active_id(tmp_path / "labelsets", "ls9")
    assert read_active_id(tmp_path / "labelsets") == "ls9"


def test_marks_follow_a_move_between_labelled_tiers_but_not_to_normal():
    """The operator circled the same pixels; only 'this is nominal' erases that."""
    ls = LabelSet()
    ls.assign("000000", "critical", label="scratch")
    ls.mark("000000", [2, 5])
    ls.assign("000000", "negative", label="scratch")
    assert ls.assignments["000000"].marks == [2, 5]
    ls.assign("000000", "normal")
    assert ls.assignments["000000"].marks == []


def test_marking_an_unassigned_image_is_an_error():
    with pytest.raises(KeyError):
        LabelSet().mark("000000", [1])


# ---- assembly --------------------------------------------------------------


def test_unassigned_images_contribute_nothing(tmp_path):
    st = _store(tmp_path, [("a.png", 5), ("b.png", 5)])
    ls = LabelSet()
    ls.assign(st.entries[0].id, "normal")
    bank = assemble_bank(st, ls)
    assert bank.normal.shape[0] == 5
    assert bank.meta.bank_images == ["a.png"]


def test_assembly_concatenates_in_store_order_with_exact_row_index(tmp_path):
    st = _store(tmp_path, [("a.png", 4), ("b.png", 6), ("c.png", 5)])
    ls = LabelSet()
    for e in st:
        ls.assign(e.id, "normal")
    bank = assemble_bank(st, ls)
    assert bank.normal.shape == (15, DIM)
    assert bank.meta.normal_image_index == [
        {"name": "a.png", "start": 0, "count": 4},
        {"name": "b.png", "start": 4, "count": 6},
        {"name": "c.png", "start": 10, "count": 5},
    ]
    for e, entry in zip(st.entries, bank.meta.normal_image_index):
        s, c = entry["start"], entry["count"]
        assert np.array_equal(bank.normal[s : s + c], st.features_of(e).astype(bank.normal.dtype))


def test_relabelling_moves_rows_without_touching_the_store(tmp_path):
    st = _store(tmp_path, [("a.png", 4), ("b.png", 6)])
    ls = LabelSet()
    ls.assign(st.entries[0].id, "normal")
    ls.assign(st.entries[1].id, "normal")
    before = assemble_bank(st, ls)
    assert before.normal.shape[0] == 10

    ls.assign(st.entries[1].id, "critical", label="scratch")
    after = assemble_bank(st, ls)
    assert after.normal.shape[0] == 4
    assert after.critical["scratch"].shape[0] == 6
    # The expensive artifact is untouched — that is the point of the split.
    assert st.total_rows() == 10
    assert np.array_equal(
        after.critical["scratch"], st.features_of(st.entries[1]).astype(after.normal.dtype)
    )


def test_severity_and_marks_land_on_the_right_rows(tmp_path):
    st = _store(tmp_path, [("ng.png", 8)])
    ls = LabelSet()
    ls.assign(st.entries[0].id, "critical", label="scratch", severity=SEVERITY_LIGHT)
    ls.mark(st.entries[0].id, [2, 5])
    bank = assemble_bank(st, ls)
    sev = bank.critical_meta["scratch"].severity
    assert list(np.flatnonzero(sev == SEVERITY_HEAVY)) == [2, 5]
    assert set(sev[[0, 1, 3, 4, 6, 7]].tolist()) == {SEVERITY_LIGHT}


def test_default_label_survives_assembly(tmp_path):
    """``_default`` must not be re-sanitised into a different bucket."""
    st = _store(tmp_path, [("ng.png", 3)])
    ls = LabelSet()
    ls.assign(st.entries[0].id, "critical", label="_default")
    bank = assemble_bank(st, ls)
    assert list(bank.critical) == ["_default"]


# ---- capacity --------------------------------------------------------------


@pytest.mark.parametrize(
    "rows,ceiling",
    [([100, 100, 100], 150), ([10, 500], 100), ([5, 5, 5], 100), ([1000], 3)],
)
def test_per_image_budget_never_exceeds_the_ceiling(rows, ceiling):
    b = per_image_budget(rows, ceiling)
    assert sum(b) <= ceiling
    assert all(1 <= x <= r for x, r in zip(b, rows))


def test_per_image_budget_gives_slack_back_to_the_hungry():
    # A straight ceiling//n split would cut the 500-row image to 50 while the
    # 10-row one left 40 of its share unused.
    assert per_image_budget([10, 500], 100) == [10, 90]


def test_per_image_budget_is_a_no_op_under_the_ceiling():
    assert per_image_budget([4, 5], 100) == [4, 5]


def test_ceiling_shrinks_images_individually_and_keeps_them_contiguous(tmp_path):
    st = _store(tmp_path, [("a.png", 60), ("b.png", 60), ("c.png", 60)])
    ls = LabelSet()
    for e in st:
        ls.assign(e.id, "normal")
    bank = assemble_bank(st, ls, normal_ceiling=90)
    assert bank.normal.shape[0] <= 90
    idx = bank.meta.normal_image_index
    assert [e["name"] for e in idx] == ["a.png", "b.png", "c.png"]
    # Contiguous, gapless, and covering the whole array.
    cursor = 0
    for e in idx:
        assert e["start"] == cursor
        cursor += e["count"]
        assert len(e["kept"]) == e["count"]  # reduced -> map recorded
    assert cursor == bank.normal.shape[0]


def test_reduced_rows_are_the_ones_the_kept_map_names(tmp_path):
    st = _store(tmp_path, [("a.png", 40)])
    ls = LabelSet()
    ls.assign(st.entries[0].id, "normal")
    bank = assemble_bank(st, ls, normal_ceiling=10)
    kept = bank.meta.normal_image_index[0]["kept"]
    src = st.features_of(st.entries[0]).astype(bank.normal.dtype)
    assert np.array_equal(bank.normal, src[np.asarray(kept)])


def test_marks_survive_reassembly_at_a_lower_capacity(tmp_path):
    """Marks address the patch grid, so a coreset must not move them."""
    st = _store(tmp_path, [("ng.png", 40)])
    eid = st.entries[0].id
    ls = LabelSet()
    ls.assign(eid, "critical", label="scratch")
    ls.mark(eid, [7, 21])
    full = assemble_bank(st, ls)
    heavy_full = set(np.flatnonzero(full.critical_meta["scratch"].severity == SEVERITY_HEAVY))
    assert heavy_full == {7, 21}

    # Re-ingesting the same image capped to 12 rows: the marks are still the
    # same two patches of the grid, and only the surviving ones are heavy.
    st2 = FeatureStore(tmp_path / "store2", meta=StoreMeta(dim=DIM))
    src = st.features_of(eid)
    kept = [0, 3, 7, 9, 12, 15, 18, 21, 25, 30, 33, 39]
    e2 = st2.add(src[np.asarray(kept)], name="ng.png", grid_rows=40, kept=kept)
    ls2 = LabelSet()
    ls2.assign(e2.id, "critical", label="scratch")
    ls2.mark(e2.id, [7, 21])
    capped = assemble_bank(st2, ls2)
    heavy = np.flatnonzero(capped.critical_meta["scratch"].severity == SEVERITY_HEAVY)
    assert [kept[i] for i in heavy] == [7, 21]


def test_assembled_bank_saves_and_reloads(tmp_path):
    st = _store(tmp_path, [("a.png", 6), ("ng.png", 4)])
    ls = LabelSet()
    ls.assign(st.entries[0].id, "normal")
    ls.assign(st.entries[1].id, "critical", label="scratch")
    ls.mark(st.entries[1].id, [1])
    bank = assemble_bank(st, ls)
    bank.save(tmp_path / "bank")

    again = Bank.load(tmp_path / "bank")
    assert again.normal.shape == (6, DIM)
    assert again.critical["scratch"].shape == (4, DIM)
    assert list(np.flatnonzero(again.critical_meta["scratch"].severity == SEVERITY_HEAVY)) == [1]
    assert again.meta.dim == DIM


def test_empty_labelset_assembles_to_an_empty_bank(tmp_path):
    st = _store(tmp_path, [("a.png", 4)])
    bank = assemble_bank(st, LabelSet())
    assert bank.normal.shape == (0, DIM)
    assert bank.critical == {}


def test_assignment_default_severity_is_the_bank_default():
    assert Assignment().clamped_severity() == DEFAULT_SEVERITY
    assert Assignment(severity=99).clamped_severity() == SEVERITY_HEAVY
    assert Assignment(severity=-4).clamped_severity() == SEVERITY_LIGHT


# ---- assembly state --------------------------------------------------------


def test_assembly_state_roundtrips(tmp_path):
    from clscore.assemble import assembly_fingerprint, read_assembly_state, write_assembly_state

    st = _store(tmp_path, [("a.png", 4)])
    ls = LabelSet(id="ls1")
    ls.assign(st.entries[0].id, "normal")
    assert read_assembly_state(tmp_path / "bank") == {}
    fp = assembly_fingerprint(st, ls)
    write_assembly_state(tmp_path / "bank", ls, fp)
    assert read_assembly_state(tmp_path / "bank") == {"labelset_id": "ls1", "fingerprint": fp}


def test_fingerprint_tracks_assignments_not_cosmetics(tmp_path):
    """Renaming a label set must not make a current bank look out of date."""
    from clscore.assemble import assembly_fingerprint

    st = _store(tmp_path, [("a.png", 4), ("b.png", 4)])
    ls = LabelSet(id="ls1", name="standard")
    ls.assign(st.entries[0].id, "normal")
    ls.assign(st.entries[1].id, "critical", label="scratch")
    base = assembly_fingerprint(st, ls)

    ls.name = "renamed"
    ls.description = "notes"
    assert assembly_fingerprint(st, ls) == base

    ls.assign(st.entries[1].id, "negative", label="scratch")
    assert assembly_fingerprint(st, ls) != base


def test_fingerprint_tracks_marks(tmp_path):
    from clscore.assemble import assembly_fingerprint

    st = _store(tmp_path, [("ng.png", 8)])
    ls = LabelSet()
    ls.assign(st.entries[0].id, "critical", label="scratch")
    before = assembly_fingerprint(st, ls)
    ls.mark(st.entries[0].id, [2, 5])
    assert assembly_fingerprint(st, ls) != before


def test_unreadable_assembly_state_reads_as_absent(tmp_path):
    """A torn write must degrade to "never assembled", not raise."""
    from clscore.assemble import ASSEMBLY_STATE_FILE, read_assembly_state

    d = tmp_path / "bank"
    d.mkdir()
    (d / ASSEMBLY_STATE_FILE).write_text("{ not json", encoding="utf-8")
    assert read_assembly_state(d) == {}
