# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The bank row index records WHICH store entry each row range came from.

Names cannot do that job: the store deliberately allows two entries to share
a filename (two folders in one zip, both holding img001.png), so every
name-keyed reader collapses them onto whichever it finds first. The id is the
only stable link back to the photograph.
"""

from __future__ import annotations

import numpy as np

from clscore.assemble import assemble_bank
from clscore.bank import INDEX_ENTRY_ID_KEY
from clscore.labelset import LabelSet
from clscore.store import FeatureStore, StoreMeta

DIM = 16


def _store(tmp_path, images: list[str]) -> FeatureStore:
    st = FeatureStore(tmp_path / "store", meta=StoreMeta(dim=DIM))
    rng = np.random.default_rng(0)
    for name in images:
        st.add(rng.random((5, DIM), dtype=np.float32), name=name, grid_rows=5)
    st.save_index()
    return st


def test_assemble_stamps_the_store_entry_id(tmp_path):
    st = _store(tmp_path, ["a.png", "b.png"])
    ls = LabelSet(id="standard", name="standard")
    for e in st.entries:
        ls.assign(e.id, tier="normal")

    bank = assemble_bank(st, ls)

    ids = [e[INDEX_ENTRY_ID_KEY] for e in bank.meta.normal_image_index]
    assert ids == [e.id for e in st.entries]


def test_duplicate_filenames_keep_separate_identities(tmp_path):
    """Two photographs, one filename: the index must still tell them apart."""
    st = _store(tmp_path, ["img001.png", "img001.png"])
    ls = LabelSet(id="standard", name="standard")
    for e in st.entries:
        ls.assign(e.id, tier="normal")

    bank = assemble_bank(st, ls)

    index = bank.meta.normal_image_index
    assert len(index) == 2, "one row range per photograph"
    assert index[0]["name"] == index[1]["name"] == "img001.png"
    assert index[0][INDEX_ENTRY_ID_KEY] != index[1][INDEX_ENTRY_ID_KEY]
    # The name log still de-duplicates; that is what made every name-keyed
    # counter under-report, and why the id had to exist.
    assert bank.meta.bank_images == ["img001.png"]


def test_the_stamp_survives_a_delete(tmp_path):
    """remove_images rebuilds surviving entries — the id must not be stripped."""
    st = _store(tmp_path, ["a.png", "b.png", "c.png"])
    ls = LabelSet(id="standard", name="standard")
    for e in st.entries:
        ls.assign(e.id, tier="normal")
    bank = assemble_bank(st, ls)
    wanted = {
        e["name"]: e[INDEX_ENTRY_ID_KEY]
        for e in bank.meta.normal_image_index
        if e["name"] != "a.png"
    }

    bank.remove_images("normal", None, ["a.png"])

    got = {e["name"]: e[INDEX_ENTRY_ID_KEY] for e in bank.meta.normal_image_index}
    assert got == wanted


def test_a_labelled_image_is_stamped_too(tmp_path):
    st = _store(tmp_path, ["ng.png"])
    ls = LabelSet(id="standard", name="standard")
    ls.assign(st.entries[0].id, tier="critical", label="scratch")

    bank = assemble_bank(st, ls)

    entry = bank.meta.critical_image_index["scratch"][0]
    assert entry[INDEX_ENTRY_ID_KEY] == st.entries[0].id


# ---- incident history follows the image, not the filename -------------------


def _hit(bank, label: str, rows: int) -> None:
    """Give every row of ``label`` some operational history."""
    m = bank.critical_meta[label]
    m.hit_count[:] = 7
    assert len(m) == rows


def test_a_retake_under_a_dead_images_filename_inherits_no_history(tmp_path):
    """Delete lot1_003.png, shoot the part again, import under the same name.

    The carry-over matched (tier, label, name, rows). The row count always
    matches -- every frame comes off one camera at one resolution -- so the
    deleted part's hit counts, severities and consolidation tier were donated
    onto a photograph they were never measured on.
    """
    st = _store(tmp_path, ["lot1_003.png"])
    dead_id = st.entries[0].id
    ls = LabelSet(id="standard", name="standard")
    ls.assign(dead_id, tier="critical", label="kizu")
    first = assemble_bank(st, ls)
    _hit(first, "kizu", 5)

    # The retake, through the route the operator actually takes: delete the
    # entry, import the new photograph under the same filename. StoreMeta's
    # next_id is monotonic and a remove never gives it back, so the retake is
    # a different entry -- which is the whole basis of this fix.
    st.remove([dead_id])
    rng = np.random.default_rng(9)
    retake = st.add(rng.random((5, DIM), dtype=np.float32), name="lot1_003.png", grid_rows=5)
    assert retake.id != dead_id, "a delete must not hand the id back"
    ls.unassign(dead_id)
    ls.assign(retake.id, tier="critical", label="kizu")

    rebuilt = assemble_bank(st, ls, prev_bank=first)

    carried = rebuilt.critical_meta["kizu"].hit_count
    assert not (carried == 7).any(), "history followed a filename onto another photo"


def test_the_same_image_still_keeps_its_history_across_a_reassemble(tmp_path):
    """The whole point of the carry-over must survive the identity change."""
    st = _store(tmp_path, ["lot1_003.png"])
    ls = LabelSet(id="standard", name="standard")
    ls.assign(st.entries[0].id, tier="critical", label="kizu")
    first = assemble_bank(st, ls)
    _hit(first, "kizu", 5)

    again = assemble_bank(st, ls, prev_bank=first)

    assert (again.critical_meta["kizu"].hit_count == 7).all(), (
        "re-assembling the same bank must not reset operational history"
    )


def test_a_legacy_bank_without_stamps_still_carries_by_name(tmp_path):
    """Banks assembled before the stamp have only the filename to go on."""
    from clscore.bank import INDEX_ENTRY_ID_KEY

    st = _store(tmp_path, ["a.png"])
    ls = LabelSet(id="standard", name="standard")
    ls.assign(st.entries[0].id, tier="critical", label="kizu")
    first = assemble_bank(st, ls)
    _hit(first, "kizu", 5)
    # Strip the stamp: this is what an older bank looks like on disk.
    for e in first.meta.critical_image_index["kizu"]:
        e.pop(INDEX_ENTRY_ID_KEY, None)

    again = assemble_bank(st, ls, prev_bank=first)
    assert (again.critical_meta["kizu"].hit_count == 7).all()


def test_moving_an_image_to_another_tier_still_starts_fresh(tmp_path):
    st = _store(tmp_path, ["a.png"])
    ls = LabelSet(id="standard", name="standard")
    ls.assign(st.entries[0].id, tier="critical", label="kizu")
    first = assemble_bank(st, ls)
    _hit(first, "kizu", 5)

    ls.assign(st.entries[0].id, tier="negative", label="kizu")
    moved = assemble_bank(st, ls, prev_bank=first)
    assert not (moved.negative_meta["kizu"].hit_count == 7).any()
