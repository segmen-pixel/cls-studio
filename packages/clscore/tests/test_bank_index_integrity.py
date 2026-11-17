# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The row-range index has to survive the operations that edit it.

``BankMeta.*_image_index`` is the only link from a slice of the feature array
back to the photograph it came from, and it carries more than the slice: the
``kept`` map that places a reduced row back on the source grid, and the
``annotations`` rectangles the operator drew. Two edits used to lose them.
"""

from __future__ import annotations

import numpy as np
import pytest

from clscore.bank import Bank

RECT = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}


def _rand(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def _entry(bank: Bank, label: str, name: str) -> dict:
    return next(e for e in bank.meta.critical_image_index[label] if e["name"] == name)


def test_removing_one_image_keeps_every_survivors_annotations():
    """Deleting a.png used to erase the marks the operator drew on b.png."""
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(6, seed=2), label="scratch", image_name="a.png")
    b.append("critical", _rand(5, seed=3), label="scratch", image_name="b.png")
    b.set_image_annotation("critical", "scratch", "b.png", [0, 2], rects=[RECT])
    assert _entry(b, "scratch", "b.png")["annotations"] == [RECT]

    b.remove_images("critical", "scratch", ["a.png"])

    survivor = _entry(b, "scratch", "b.png")
    assert survivor["annotations"] == [RECT], "the survivor's marks must not move"
    assert survivor["start"] == 0, "but its rows do shift down"
    assert survivor["count"] == 5


def test_removing_one_image_keeps_every_survivors_kept_map():
    """``kept`` is what places a reduced row back on the source grid."""
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(4, seed=2), label="scratch", image_name="a.png",
             kept_idx=[0, 3, 7, 9])
    b.append("critical", _rand(3, seed=3), label="scratch", image_name="b.png",
             kept_idx=[1, 4, 8])
    assert _entry(b, "scratch", "b.png")["kept"] == [1, 4, 8]

    b.remove_images("critical", "scratch", ["a.png"])
    assert _entry(b, "scratch", "b.png")["kept"] == [1, 4, 8]


def test_removing_an_image_still_prunes_exactly_its_rows():
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(6, seed=2), label="scratch", image_name="a.png")
    b.append("critical", _rand(5, seed=3), label="scratch", image_name="b.png")
    out = b.remove_images("critical", "scratch", ["a.png"])
    assert out == {"rows_removed": 6, "names_removed": 1}
    assert b.critical["scratch"].shape[0] == 5
    assert b.meta.critical_images["scratch"] == ["b.png"]


def test_clearing_a_tier_also_clears_its_row_index():
    """Phantoms: /bank/images reads the index, not the filename log.

    Leaving the index behind made a cleared tier keep listing images whose
    files the route had already unlinked.
    """
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(6, seed=2), label="scratch", image_name="a.png")
    b.append("negative", _rand(4, seed=3), label="dust", image_name="n.png")

    b.clear_tier("critical")
    assert b.meta.critical_images == {}
    assert b.meta.critical_image_index == {}, "no phantom images"
    assert b.meta.negative_image_index != {}, "the other tier is untouched"

    b.clear_tier("negative")
    assert b.meta.negative_image_index == {}


def test_clearing_one_label_also_clears_its_row_index():
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(6, seed=2), label="scratch", image_name="a.png")
    b.append("critical", _rand(4, seed=3), label="dent", image_name="d.png")

    b.clear_label("critical", "scratch")
    assert "scratch" not in b.meta.critical_image_index
    assert "dent" in b.meta.critical_image_index, "the other label survives"


def test_the_normal_tier_is_still_protected():
    b = Bank(normal=_rand(4, seed=1))
    with pytest.raises(ValueError):
        b.clear_tier("normal")
    with pytest.raises(ValueError):
        b.clear_label("normal", "whatever")
