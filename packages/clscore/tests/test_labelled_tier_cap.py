# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Capping the labelled tiers must not move the NG marks.

The critical / negative tiers used to be exempt from the per-image patch cap
"because they are small". They are not: banks on the dev box reached 12.8 M
labelled rows (~20 GB resident) against a capped normal tier of 219 k, and one
project held 7.7 M labelled rows with an empty normal tier, so the capacity
gauge read 0% while the bank cost 11.8 GB.

They were exempt for a real reason though: an NG image's bank rows are
addressed geometrically (`rows_for_rects` returns patch-grid indices), and a
coreset-reduced image no longer has row i == patch i. So a reduced image stores
which grid patches survived, and the marks map through it. These pin that map.
"""
from __future__ import annotations

import numpy as np

from clscore.bank import Bank, coreset_reduce, coreset_reduce_indexed
from clscore.incident import DEFAULT_SEVERITY, SEVERITY_HEAVY

DIM = 16


def _feats(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, DIM)).astype(np.float32)


def test_indexed_reduce_agrees_with_the_plain_one():
    f = _feats(200, 1)
    sub, idx = coreset_reduce_indexed(f, 0.25, "cpu")
    assert sub.shape == (50, DIM)
    assert idx.shape == (50,)
    assert len(set(idx.tolist())) == 50, "no row selected twice"
    assert idx.max() < 200
    # The subset really is those rows of the input, in that order.
    assert np.allclose(sub, f[idx])
    # And it matches what the plain helper returns for the same inputs.
    assert np.allclose(sub, coreset_reduce(f, 0.25, "cpu"))


def test_no_reduction_returns_an_identity_map():
    f = _feats(10, 2)
    sub, idx = coreset_reduce_indexed(f, 1.0, "cpu")
    assert sub is f
    assert idx.tolist() == list(range(10))


def test_capped_critical_image_records_its_kept_rows():
    bank = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    bank.append("critical", _feats(300, 3), label="ng", image_name="a.png",
                max_patches=60, device="cpu")
    entry = bank.meta.critical_image_index["ng"][0]
    assert entry["count"] == 60
    assert "kept" in entry, "a reduced image must record which patches survived"
    assert len(entry["kept"]) == 60
    assert max(entry["kept"]) < 300
    assert bank.critical["ng"].shape[0] == 60


def test_uncapped_image_records_no_map():
    """Absence of `kept` is what every pre-cap bank on disk relies on."""
    bank = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    bank.append("critical", _feats(40, 4), label="ng", image_name="small.png",
                max_patches=100, device="cpu")
    entry = bank.meta.critical_image_index["ng"][0]
    assert entry["count"] == 40
    assert "kept" not in entry


def test_marks_land_on_the_rows_the_grid_indices_named():
    """The whole point: mark by patch-grid index, hit the right bank rows."""
    bank = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    bank.append("critical", _feats(300, 5), label="ng", image_name="a.png",
                max_patches=60, device="cpu")
    entry = bank.meta.critical_image_index["ng"][0]
    kept = list(entry["kept"])

    # Ask for three grid patches that survived and one that did not.
    survived = [kept[7], kept[20], kept[41]]
    dropped = next(g for g in range(300) if g not in set(kept))
    n = bank.set_image_annotation("critical", "ng", "a.png", [*survived, dropped],
                                  severity=SEVERITY_HEAVY)
    assert n == 3, "the patch that was not kept cannot be marked"

    sev = bank.critical_meta["ng"].severity
    marked = sorted(np.flatnonzero(sev == SEVERITY_HEAVY).tolist())
    assert marked == sorted([7, 20, 41]), "marks must sit at the kept rows' positions"
    assert (sev[[i for i in range(60) if i not in (7, 20, 41)]] == DEFAULT_SEVERITY).all()


def test_marks_still_work_without_a_map():
    """Legacy path: no `kept`, so grid index == row index."""
    bank = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    bank.append("critical", _feats(30, 6), label="ng", image_name="legacy.png",
                max_patches=0, device="cpu")
    n = bank.set_image_annotation("critical", "ng", "legacy.png", [2, 5],
                                  severity=SEVERITY_HEAVY)
    assert n == 2
    sev = bank.critical_meta["ng"].severity
    assert sorted(np.flatnonzero(sev == SEVERITY_HEAVY).tolist()) == [2, 5]


def test_second_image_marks_are_offset_by_the_first():
    """Two capped images in one label: the map is per image, `start` is not."""
    bank = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    bank.append("critical", _feats(300, 7), label="ng", image_name="a.png",
                max_patches=60, device="cpu")
    bank.append("critical", _feats(300, 8), label="ng", image_name="b.png",
                max_patches=60, device="cpu")
    e_a, e_b = bank.meta.critical_image_index["ng"]
    assert (e_a["start"], e_a["count"]) == (0, 60)
    assert (e_b["start"], e_b["count"]) == (60, 60)

    bank.set_image_annotation("critical", "ng", "b.png", [e_b["kept"][3]],
                              severity=SEVERITY_HEAVY)
    sev = bank.critical_meta["ng"].severity
    assert np.flatnonzero(sev == SEVERITY_HEAVY).tolist() == [63], "60 + 3"


def test_caller_supplied_map_composes_with_a_second_reduction():
    """append() may reduce again on top of a caller's reduction."""
    f = _feats(400, 9)
    once, idx1 = coreset_reduce_indexed(f, 0.5, "cpu")  # 400 -> 200
    bank = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    bank.append("critical", once, label="ng", image_name="a.png",
                max_patches=50, device="cpu", kept_idx=idx1)
    entry = bank.meta.critical_image_index["ng"][0]
    assert entry["count"] == 50
    # Every recorded index must still address the ORIGINAL 400-patch grid.
    assert len(entry["kept"]) == 50
    assert max(entry["kept"]) < 400
    assert set(entry["kept"]).issubset(set(idx1.tolist()))
