# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""NG defect marks: exemplar-row flagging + meta-only persistence.

``Bank.set_image_annotation`` records which of an image's rows are defect
exemplars (severity=heavy) with replace semantics, storing the source
rectangles on the image's row-range index entry so a UI can re-edit them.
``Bank.save_meta_only`` persists just the metadata sidecars — annotating one
image must never rewrite a multi-GB feature array.
"""

from __future__ import annotations

import numpy as np
import pytest

from clscore.bank import Bank
from clscore.incident import DEFAULT_SEVERITY, SEVERITY_HEAVY

RECT = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}


def _rand(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def _bank_with_two_images() -> Bank:
    """critical/scratch holds a.png (rows 0..5) then b.png (rows 6..10)."""
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(6, seed=2), label="scratch", image_name="a.png")
    b.append("critical", _rand(5, seed=3), label="scratch", image_name="b.png")
    return b


def _entry(b: Bank, name: str) -> dict:
    return next(e for e in b.meta.critical_image_index["scratch"] if e["name"] == name)


def test_annotation_sets_heavy_severity_on_selected_rows_only():
    b = _bank_with_two_images()
    n = b.set_image_annotation("critical", "scratch", "b.png", [0, 2], rects=[RECT])
    assert n == 2
    sev = b.critical_meta["scratch"].severity
    # a.png's rows are untouched; only b.png's local rows {0, 2} are heavy.
    assert (sev[:6] == DEFAULT_SEVERITY).all()
    assert sev[6] == SEVERITY_HEAVY and sev[8] == SEVERITY_HEAVY
    assert sev[7] == DEFAULT_SEVERITY and (sev[9:] == DEFAULT_SEVERITY).all()


def test_annotation_replaces_previous_marks():
    b = _bank_with_two_images()
    b.set_image_annotation("critical", "scratch", "b.png", [0, 1, 2], rects=[RECT])
    b.set_image_annotation("critical", "scratch", "b.png", [4], rects=[RECT])
    sev = b.critical_meta["scratch"].severity
    assert (sev[6:9] == DEFAULT_SEVERITY).all()
    assert sev[10] == SEVERITY_HEAVY


def test_annotation_stores_rects_and_empty_rows_clears_everything():
    b = _bank_with_two_images()
    b.set_image_annotation("critical", "scratch", "b.png", [1], rects=[RECT])
    assert _entry(b, "b.png")["annotations"] == [RECT]
    n = b.set_image_annotation("critical", "scratch", "b.png", [], rects=[])
    assert n == 0
    assert "annotations" not in _entry(b, "b.png")
    assert (b.critical_meta["scratch"].severity == DEFAULT_SEVERITY).all()


def test_annotation_drops_out_of_range_rows():
    b = _bank_with_two_images()
    # a.png has 6 rows: 0 and 5 are valid, 99 and -1 must be dropped.
    assert b.set_image_annotation("critical", "scratch", "a.png", [0, 5, 99, -1], rects=[RECT]) == 2


def test_annotation_accepts_resolved_default_label_verbatim():
    """Unlabelled teaches land under "_default" on disk; annotating with that
    resolved label must hit the same key (safe_label would strip it to
    "default" and miss)."""
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(3, seed=5), label="", image_name="c.png")
    assert "_default" in b.meta.critical_image_index
    assert b.set_image_annotation("critical", "_default", "c.png", [1], rects=[RECT]) == 1
    assert b.critical_meta["_default"].severity[1] == SEVERITY_HEAVY


def test_annotation_unknown_image_raises_keyerror():
    b = _bank_with_two_images()
    with pytest.raises(KeyError):
        b.set_image_annotation("critical", "scratch", "nope.png", [0])


def test_annotation_normal_tier_raises_valueerror():
    b = _bank_with_two_images()
    with pytest.raises(ValueError):
        b.set_image_annotation("normal", "", "a.png", [0])


def test_save_meta_only_persists_marks_without_touching_features(tmp_path):
    b = _bank_with_two_images()
    b.save(tmp_path)
    feat_path = tmp_path / "critical" / "scratch.npy"
    mtime_before = feat_path.stat().st_mtime_ns
    b.set_image_annotation("critical", "scratch", "b.png", [0, 2], rects=[RECT])
    b.save_meta_only(tmp_path, "critical")
    assert feat_path.stat().st_mtime_ns == mtime_before
    b2 = Bank.load(tmp_path)
    sev = b2.critical_meta["scratch"].severity
    assert sev[6] == SEVERITY_HEAVY and sev[8] == SEVERITY_HEAVY
    assert _entry(b2, "b.png")["annotations"] == [RECT]
