# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""scripts/reduce_bank.py — shrinking a bank that is already on disk.

The per-image cap only bounds new teaches, so banks in the field stay as large
as they grew (12.8 M labelled rows, ~20 GB, on the box this was written for).
The tool reduces them in place, and the things it must not get wrong are:

  * an annotated row is an alpha exemplar — dropping one silently changes
    detection, so they are all kept regardless of the budget;
  * the per-row metadata has to travel with its own feature row, which is not
    the same as "in ascending order" once marks are pulled to the front;
  * the kept-index map has to keep addressing the ORIGINAL patch grid, even
    when the image was already reduced once at teach time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from clscore.bank import Bank
from clscore.incident import DEFAULT_SEVERITY, SEVERITY_HEAVY

_TOOL = Path(__file__).resolve().parents[3] / "scripts" / "reduce_bank.py"
_spec = importlib.util.spec_from_file_location("reduce_bank", _TOOL)
reduce_bank = importlib.util.module_from_spec(_spec)
sys.modules["reduce_bank"] = reduce_bank
_spec.loader.exec_module(reduce_bank)

DIM = 16


def _feats(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, DIM)).astype(np.float32)


def _bank_with(n_rows: int, name: str = "a.png", seed: int = 0) -> Bank:
    b = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    b.append("critical", _feats(n_rows, seed), label="ng", image_name=name,
             max_patches=0, device="cpu")
    return b


def _reduce(b: Bank, cap: int):
    return reduce_bank._reduce_label(
        b.critical["ng"], b.critical_meta["ng"], b.meta.critical_image_index["ng"],
        cap, "cpu", apply=True,
    )


def test_image_under_the_cap_is_untouched():
    b = _bank_with(100)
    arr, meta, entries, saved = _reduce(b, 500)
    assert saved == 0
    assert arr is b.critical["ng"]
    assert entries[0]["count"] == 100 and "kept" not in entries[0]


def test_reduces_to_the_cap_and_records_the_grid_map():
    b = _bank_with(1000, seed=1)
    arr, meta, entries, saved = _reduce(b, 200)
    assert arr.shape[0] == 200 and saved == 800
    assert len(meta) == 200
    e = entries[0]
    assert (e["start"], e["count"]) == (0, 200)
    assert len(e["kept"]) == 200 and max(e["kept"]) < 1000
    assert len(set(e["kept"])) == 200


def test_every_annotated_row_survives():
    b = _bank_with(1000, seed=2)
    marks = [3, 77, 512, 999]
    b.set_image_annotation("critical", "ng", "a.png", marks, severity=SEVERITY_HEAVY)
    arr, meta, entries, _saved = _reduce(b, 100)
    kept = list(entries[0]["kept"])
    assert set(marks).issubset(set(kept)), "an exemplar was dropped"
    # ...and they are still flagged, at their new positions.
    still = {kept[i] for i in np.flatnonzero(meta.severity == SEVERITY_HEAVY).tolist()}
    assert still == set(marks)


def test_metadata_travels_with_its_own_row():
    """Marks are pulled to the front, so a boolean mask would mis-align."""
    b = _bank_with(600, seed=3)
    src = b.critical["ng"].copy()
    b.set_image_annotation("critical", "ng", "a.png", [5, 400], severity=SEVERITY_HEAVY)
    arr, meta, entries, _ = _reduce(b, 60)
    kept = list(entries[0]["kept"])
    assert len(meta) == arr.shape[0] == len(kept)
    # Row i of the rebuilt array must be original row kept[i] ...
    for i in (0, 1, len(kept) // 2, len(kept) - 1):
        assert np.array_equal(arr[i], src[kept[i]])
    # ... and its severity must be the one that original row carried.
    for i, g in enumerate(kept):
        want = SEVERITY_HEAVY if g in (5, 400) else DEFAULT_SEVERITY
        assert meta.severity[i] == want


def test_more_marks_than_the_cap_keeps_them_all():
    b = _bank_with(500, seed=4)
    marks = list(range(0, 300, 2))  # 150 exemplars, cap will be 50
    b.set_image_annotation("critical", "ng", "a.png", marks, severity=SEVERITY_HEAVY)
    arr, meta, entries, _ = _reduce(b, 50)
    assert arr.shape[0] == len(marks), "the cap must never cost an exemplar"
    assert sorted(entries[0]["kept"]) == marks


def test_map_composes_with_a_teach_time_reduction():
    b = Bank(normal=np.empty((0, DIM), dtype=np.float16))
    b.append("critical", _feats(4000, 5), label="ng", image_name="a.png",
             max_patches=1000, device="cpu")
    first = list(b.meta.critical_image_index["ng"][0]["kept"])
    _arr, _meta, entries, _ = _reduce(b, 100)
    kept = entries[0]["kept"]
    assert len(kept) == 100
    assert set(kept).issubset(set(first)), "indices must still address the 4000-patch grid"
    assert max(kept) < 4000


def test_two_images_are_repacked_contiguously():
    b = _bank_with(800, name="a.png", seed=6)
    b.append("critical", _feats(800, 7), label="ng", image_name="b.png",
             max_patches=0, device="cpu")
    arr, meta, entries, saved = _reduce(b, 100)
    assert [(e["start"], e["count"]) for e in entries] == [(0, 100), (100, 100)]
    assert arr.shape[0] == 200 and len(meta) == 200 and saved == 1400


def test_refuses_a_row_index_that_does_not_tile():
    b = _bank_with(300, seed=8)
    b.meta.critical_image_index["ng"][0]["count"] = 200  # now covers 200 of 300
    try:
        _reduce(b, 50)
    except ValueError as exc:
        assert "refusing to rewrite" in str(exc)
    else:
        raise AssertionError("a partial row index must not be rewritten")
