# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Group derivation, and the multi-range exclusion that leave-own-group-out needs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from clscore.compress import IvfIndex
from clscore.grouping import (
    derive_groups,
    exclusion_ranges,
    group_of,
    group_summary,
)
from clscore.scoring import _merge_ranges, score_stored_features

# ---- group derivation ------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("OK_170104_094937_cam1.png", "170104"),   # the convention on this line
        ("NG_170104_101122.png", "170104"),
        ("IMG_20260804_094937.jpg", "20260804"),
        ("2026-08-04_094937.png", "20260804"),
        ("2026_08_04-094937.png", "20260804"),
        ("shot_20260804.png", "20260804"),
        ("2017-10-16_12-46-58-7650.jpg", "20171016"),
        ("ok_Image__2017-05-16__09-33-06.png", "20170516"),
        # YYMMDDHHMMSS + milliseconds, run together with no separator.
        ("OK_a02_02_170302164558368.png", "170302"),
        ("plate_a.png", ""),                        # no date at all
        # Serials shaped like dates. Real filenames on these lines; before the
        # calendar check the last one produced one group per image.
        ("serial_19283746_a.png", ""),              # month 37
        ("run_12345678_a.png", ""),                 # month 56
        ("OK_0002-08-02-0008.jpg", ""),             # year 0002
        ("OK__IMG_6839.JPG", ""),
        ("000057681_OK_AAA.png", ""),
    ],
)
def test_datetime_group_reads_the_filename(name, expected):
    assert group_of(name, "datetime") == expected


def test_datetime_ignores_a_bare_six_digit_run():
    """Six digits alone are a counter far more often than a date."""
    assert group_of("part_170104.png", "datetime") == ""


@pytest.mark.parametrize(
    "name,sep,fields,expected",
    [
        ("LOTA_170104_1.png", "_", 1, "LOTA"),
        ("LOTA_170104_1.png", "_", 2, "LOTA_170104"),
        ("LOTA-170104-1.png", "-", 1, "LOTA"),
        ("noseparator.png", "_", 1, "noseparator"),
        ("LOTA_1.png", "_", 9, "LOTA_1"),  # more fields than the name has
    ],
)
def test_prefix_group_splits_the_stem(name, sep, fields, expected):
    assert group_of(name, "prefix", sep=sep, fields=fields) == expected


def test_manual_group_is_an_explicit_map():
    manual = {"a.png": "lot-1", "b.png": " lot-1 "}
    assert group_of("a.png", "manual", manual=manual) == "lot-1"
    assert group_of("b.png", "manual", manual=manual) == "lot-1"
    assert group_of("c.png", "manual", manual=manual) == ""


def test_none_mode_groups_nothing():
    assert derive_groups(["a.png", "b.png"], "none") == {}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown group mode"):
        derive_groups(["a.png"], "byvibes")


def test_group_summary_counts_what_the_rule_failed_to_place():
    groups = derive_groups(
        [
            "OK_170104_094937_1.png",
            "OK_170104_101122_2.png",
            "OK_170105_083010_1.png",
            "plate.png",
        ],
        "datetime",
    )
    summary = group_summary(groups)
    assert summary["170104"] == ["OK_170104_094937_1.png", "OK_170104_101122_2.png"]
    assert summary["170105"] == ["OK_170105_083010_1.png"]
    assert summary[""] == ["plate.png"]


# ---- exclusion ranges ------------------------------------------------------


INDEX = [
    {"name": "a.png", "start": 0, "count": 10},
    {"name": "b.png", "start": 10, "count": 5},
    {"name": "c.png", "start": 15, "count": 7},
]


def test_ungrouped_image_excludes_only_itself():
    assert exclusion_ranges(INDEX, {}, "b.png") == [(10, 5)]


def test_grouped_image_excludes_the_whole_lot():
    groups = {"a.png": "lot1", "b.png": "lot1", "c.png": "lot2"}
    assert exclusion_ranges(INDEX, groups, "b.png") == [(0, 10), (10, 5)]
    assert exclusion_ranges(INDEX, groups, "c.png") == [(15, 7)]


def test_an_empty_group_key_does_not_pool_the_unplaced_together():
    """Two images the rule could not place are not therefore the same lot."""
    groups = {"a.png": "", "b.png": "", "c.png": ""}
    assert exclusion_ranges(INDEX, groups, "a.png") == [(0, 10)]


# ---- range merging ---------------------------------------------------------


def test_merge_ranges_clips_drops_and_joins():
    assert _merge_ranges([(0, 5), (3, 4), (20, 5)], 100) == [(0, 7), (20, 5)]
    assert _merge_ranges([(90, 50)], 100) == [(90, 10)]      # clipped to the bank
    assert _merge_ranges([(5, 0), (-3, 2)], 100) == []       # empty / fully negative
    assert _merge_ranges([], 100) == []


def test_merge_ranges_touching_spans_become_one():
    assert _merge_ranges([(0, 5), (5, 5)], 100) == [(0, 10)]


# ---- multi-range exclusion in scoring --------------------------------------


def _bank(n=200, dim=8, seed=0):
    return np.random.default_rng(seed).random((n, dim)).astype(np.float32)


def _reference(q, bank, k, excluded_rows):
    """Ground truth: physically remove the rows, then score."""
    keep = np.setdiff1d(np.arange(bank.shape[0]), np.asarray(excluded_rows, dtype=np.int64))
    return score_stored_features(q, bank[keep], k=k)


def test_a_single_range_is_unchanged_by_the_generalisation():
    bank = _bank()
    q = bank[10:15]
    old = score_stored_features(q, bank, k=3, exclude_start=10, exclude_count=5)
    new = score_stored_features(q, bank, k=3, exclude_ranges=[(10, 5)])
    assert np.array_equal(old, new)
    assert np.allclose(old, _reference(q, bank, 3, range(10, 15)), rtol=1e-5, atol=1e-6)


def test_multiple_ranges_match_physically_removing_the_rows():
    bank = _bank()
    q = bank[10:15]
    ranges = [(0, 10), (10, 5), (120, 20)]
    got = score_stored_features(q, bank, k=4, exclude_ranges=ranges)
    excluded = list(range(0, 15)) + list(range(120, 140))
    assert np.allclose(got, _reference(q, bank, 4, excluded), rtol=1e-5, atol=1e-6)


def test_start_count_and_ranges_compose():
    bank = _bank()
    q = bank[10:15]
    got = score_stored_features(
        q, bank, k=4, exclude_start=10, exclude_count=5, exclude_ranges=[(0, 10)]
    )
    assert np.allclose(got, _reference(q, bank, 4, range(0, 15)), rtol=1e-5, atol=1e-6)


def test_overlapping_ranges_do_not_starve_k():
    """Double-counting the overlap would ask for more neighbours than remain."""
    bank = _bank(n=20)
    q = bank[:2]
    got = score_stored_features(q, bank, k=5, exclude_ranges=[(0, 15), (10, 8)])
    assert np.isfinite(got).all()
    assert np.allclose(got, _reference(q, bank, 5, range(0, 18)), rtol=1e-5, atol=1e-6)


def test_excluding_almost_everything_still_returns_finite_scores():
    bank = _bank(n=30)
    q = bank[:3]
    got = score_stored_features(q, bank, k=5, exclude_ranges=[(0, 29)])
    assert np.isfinite(got).all()


# ---- multi-range exclusion through the IVF paths ---------------------------


def _blobs(n_per=100, centers=3, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    out = [
        rng.normal(loc=float(c) * 8.0, scale=0.4, size=(n_per, dim))
        for c in range(centers)
    ]
    return np.concatenate(out).astype(np.float32)


def test_ivf_mask_path_honours_every_range():
    bank = _blobs()
    bank_t = torch.from_numpy(bank)
    idx = IvfIndex.build(bank_t, n_clusters=3, seed=1)
    q = bank[100:105]
    ranges = [(0, 100), (100, 5)]
    base = score_stored_features(q, bank_t, k=3, exclude_ranges=ranges)
    routed = score_stored_features(
        q, bank_t, k=3, exclude_ranges=ranges, ivf=idx, ivf_nprobe=3
    )
    assert np.isfinite(routed).all()
    assert np.allclose(base, routed, rtol=1e-5, atol=1e-6)


def test_ivf_resident_storage_path_honours_every_range():
    """The gather path re-derives the mask from permuted row ids."""
    bank = _blobs()
    bank_t = torch.from_numpy(bank)
    idx = IvfIndex.build(bank_t, n_clusters=3, seed=1)
    idx.set_storage(bank)
    q = bank[100:105]
    ranges = [(0, 100), (100, 5)]
    base = score_stored_features(q, bank_t, k=3, exclude_ranges=ranges)
    routed = score_stored_features(
        q, None, k=3, exclude_ranges=ranges, ivf=idx, ivf_nprobe=3
    )
    assert np.isfinite(routed).all()
    assert np.allclose(base, routed, rtol=1e-4, atol=1e-5)


def test_ivf_full_scan_fallback_honours_every_range():
    """A narrow probe whose candidates are entirely excluded falls back."""
    bank = _blobs(n_per=100, centers=2)
    bank_t = torch.from_numpy(bank)
    idx = IvfIndex.build(bank_t, n_clusters=2, seed=1)
    idx.set_storage(bank)
    q = bank[:5]
    ranges = [(0, 50), (50, 50)]  # all of the query's own blob
    base = score_stored_features(q, bank_t, k=3, exclude_ranges=ranges)
    routed = score_stored_features(
        q, None, k=3, exclude_ranges=ranges, ivf=idx, ivf_nprobe=1
    )
    assert np.isfinite(routed).all()
    assert np.allclose(base, routed, rtol=1e-4, atol=1e-5)
