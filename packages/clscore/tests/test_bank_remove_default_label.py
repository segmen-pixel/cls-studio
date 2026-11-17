# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Per-image delete must accept resolved on-disk labels verbatim.

``Bank.remove_images`` receives the label exactly as the images listing
reported it (e.g. ``"_default"``). Re-sanitising through ``safe_label``
strips the leading underscore into a key that doesn't exist, so deleting a
default-label image silently removed nothing — same bug class that was
fixed for ``set_image_annotation``.
"""

from __future__ import annotations

import numpy as np

from clscore.bank import DEFAULT_LABEL, Bank


def _rand(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def _bank_with_default_label_images() -> Bank:
    """critical/_default holds a.png (6 rows) then b.png (5 rows)."""
    b = Bank(normal=_rand(4, seed=1))
    # Empty label resolves to DEFAULT_LABEL ("_default") on append.
    b.append("critical", _rand(6, seed=2), label="", image_name="a.png")
    b.append("critical", _rand(5, seed=3), label="", image_name="b.png")
    assert DEFAULT_LABEL in b.critical
    return b


def test_remove_images_accepts_resolved_default_label():
    b = _bank_with_default_label_images()
    r = b.remove_images("critical", DEFAULT_LABEL, ["a.png"])
    assert r["rows_removed"] == 6
    assert b.critical[DEFAULT_LABEL].shape[0] == 5
    names = [e["name"] for e in b.meta.critical_image_index[DEFAULT_LABEL]]
    assert names == ["b.png"]
    # b.png's rows shifted down to the front of the compacted array.
    assert b.meta.critical_image_index[DEFAULT_LABEL][0]["start"] == 0


def test_remove_images_none_label_still_falls_back_to_default():
    b = _bank_with_default_label_images()
    r = b.remove_images("critical", None, ["b.png"])
    assert r["rows_removed"] == 5
    names = [e["name"] for e in b.meta.critical_image_index[DEFAULT_LABEL]]
    assert names == ["a.png"]


def test_remove_images_custom_label_passed_verbatim():
    b = Bank(normal=_rand(4, seed=1))
    b.append("critical", _rand(3, seed=4), label="scratch", image_name="c.png")
    r = b.remove_images("critical", "scratch", ["c.png"])
    assert r["rows_removed"] == 3
    # Last image under the label gone → the label is dropped everywhere.
    assert "scratch" not in b.critical
    assert "scratch" not in b.meta.critical_image_index
