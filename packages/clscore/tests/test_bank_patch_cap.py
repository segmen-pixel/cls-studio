# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Per-image patch cap on the normal tier.

A too-large OK library grows the bank unbounded (plain concatenate). The cap
coreset-reduces each image's patches to a bound BEFORE appending, which must
keep the per-image row index exact — leave-one-out eval, per-image delete and
NG marks all read those (name, start, count) ranges.
"""
from __future__ import annotations

import numpy as np

from clscore.bank import Bank


def _rand(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def test_normal_image_capped_to_bound():
    b = Bank(normal=np.empty((0, 16), dtype=np.float16))
    b.append("normal", _rand(5000, seed=1), image_name="a.png", max_patches=1000, device="cpu")
    assert b.normal.shape[0] == 1000
    idx = b.meta.normal_image_index
    assert len(idx) == 1
    assert idx[0]["name"] == "a.png"
    assert idx[0]["start"] == 0
    assert idx[0]["count"] == 1000


def test_index_stays_exact_and_contiguous_across_appends():
    b = Bank(normal=np.empty((0, 16), dtype=np.float16))
    b.append("normal", _rand(5000, seed=1), image_name="a.png", max_patches=1000, device="cpu")
    b.append("normal", _rand(300, seed=2), image_name="b.png", max_patches=1000, device="cpu")  # below cap
    b.append("normal", _rand(4000, seed=3), image_name="c.png", max_patches=1000, device="cpu")
    idx = b.meta.normal_image_index
    counts = [e["count"] for e in idx]
    assert counts == [1000, 300, 1000]
    # contiguous, non-overlapping, covering exactly the whole array
    cursor = 0
    for e in idx:
        assert e["start"] == cursor
        cursor += e["count"]
    assert cursor == b.normal.shape[0] == 2300


def test_below_cap_is_not_reduced():
    b = Bank(normal=np.empty((0, 16), dtype=np.float16))
    b.append("normal", _rand(500, seed=1), image_name="a.png", max_patches=1000, device="cpu")
    assert b.normal.shape[0] == 500


def test_cap_zero_disables():
    b = Bank(normal=np.empty((0, 16), dtype=np.float16))
    b.append("normal", _rand(5000, seed=1), image_name="a.png", max_patches=0, device="cpu")
    assert b.normal.shape[0] == 5000


def test_labelled_tiers_are_capped_too_and_record_their_map():
    # These used to be exempt "because they are small". They were not: real
    # banks reached 12.8 M labelled rows against a 219 k capped normal tier.
    # The reason for the exemption was real though — NG/FP rows back the alpha
    # exemplars and the marks address the patch grid — so a reduced image
    # records which grid patches survived. See test_labelled_tier_cap.py.
    b = Bank(normal=_rand(4, seed=9))
    b.append("critical", _rand(5000, seed=1), label="scratch", image_name="x.png",
             max_patches=1000, device="cpu")
    assert b.critical["scratch"].shape[0] == 1000
    entry = b.meta.critical_image_index["scratch"][0]
    assert entry["count"] == 1000
    assert len(entry["kept"]) == 1000 and max(entry["kept"]) < 5000


def test_capped_rows_are_a_real_subset():
    feats = _rand(3000, seed=7).astype(np.float16)
    b = Bank(normal=np.empty((0, 16), dtype=np.float16))
    b.append("normal", feats.copy(), image_name="a.png", max_patches=800, device="cpu")
    kept = b.normal
    assert kept.shape[0] == 800
    # every kept row must be one of the original rows (coreset selects, never invents)
    orig = {r.tobytes() for r in feats}
    assert all(r.tobytes() in orig for r in kept)
