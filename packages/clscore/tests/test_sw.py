# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Sliding-window geometry: window placement and reflect-padding."""

from __future__ import annotations

import numpy as np

from clscore.sw import WINDOW_SIZE, expected_rows, pad_to_min, rows_for_rects, sw_offsets

# ---- sw_offsets -------------------------------------------------------------


def test_sw_offsets_image_smaller_than_window_returns_origin_only():
    """Anything smaller than one window should anchor at (0, 0); we rely
    on pad_to_min to grow the image up to size rather than producing
    multiple weirdly-overlapped windows on a tiny image."""
    assert sw_offsets(100, 200, window_size=518, stride=388) == [(0, 0)]


def test_sw_offsets_image_exactly_one_window_returns_one_offset():
    assert sw_offsets(518, 518, window_size=518, stride=388) == [(0, 0)]


def test_sw_offsets_anchors_last_row_and_column_to_image_edge():
    """If the stride doesn't tile the image evenly, the last row/column
    offset must be pinned to ``size - window`` so the bottom and right
    edges still get scored."""
    offsets = sw_offsets(900, 900, window_size=518, stride=388)
    ys = sorted({y for y, _ in offsets})
    xs = sorted({x for _, x in offsets})
    assert ys[-1] == 900 - 518
    assert xs[-1] == 900 - 518


def test_sw_offsets_strided_tiling_is_unique():
    """No duplicate (y, x) entries even when the edge anchor coincides
    with the strided position."""
    offsets = sw_offsets(518, 518 + 388, window_size=518, stride=388)
    assert len(offsets) == len(set(offsets))


def test_sw_offsets_default_arguments_match_module_constants():
    a = sw_offsets(800, 800)
    b = sw_offsets(800, 800, window_size=WINDOW_SIZE, stride=388)
    assert a == b


# ---- pad_to_min -------------------------------------------------------------


def test_pad_to_min_no_op_when_already_large():
    img = np.zeros((600, 700, 3), dtype=np.uint8)
    out, orig = pad_to_min(img, window_size=518)
    assert out.shape == img.shape
    assert orig == (600, 700)


def test_pad_to_min_pads_short_dimensions_only():
    img = np.zeros((100, 700, 3), dtype=np.uint8)
    out, orig = pad_to_min(img, window_size=518)
    assert out.shape[0] >= 518
    assert out.shape[1] == 700  # untouched
    assert orig == (100, 700)


def test_pad_to_min_records_original_size_for_unpadding():
    """The returned (H, W) must reflect the *pre-pad* size so callers
    can crop the heatmap back to the original frame at the end of
    inference."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, orig = pad_to_min(img, window_size=518)
    assert orig == (50, 50)


# ---- rows_for_rects / expected_rows ------------------------------------------
# Toy geometry small enough to enumerate by hand: window 4, stride 2, patch 2
# -> win_p = 2, and a 6x6 image tiles into 4 windows at (0,0)/(0,2)/(2,0)/(2,2)
# contributing 4 rows each (row = win_idx*4 + py*2 + px).

_GEO = {"window_size": 4, "stride": 2, "patch": 2}


def test_expected_rows_matches_toy_and_production_geometry():
    assert expected_rows(6, 6, **_GEO) == 16
    # The real bank layout: 1360x1024 -> 3x4 windows x 37^2 patches.
    assert expected_rows(1024, 1360) == 12 * (WINDOW_SIZE // 14) ** 2


def test_rows_for_rects_full_image_selects_every_row():
    assert rows_for_rects(6, 6, [(0.0, 0.0, 1.0, 1.0)], **_GEO) == list(range(16))


def test_rows_for_rects_corner_pixel_hits_single_window_row():
    """A rect inside pixel (0,0) touches only patch (0,0) of window 0 —
    the other three windows don't cover that pixel at all."""
    assert rows_for_rects(6, 6, [(0.0, 0.0, 0.1, 0.1)], **_GEO) == [0]


def test_rows_for_rects_center_pixel_selects_one_row_per_covering_window():
    """Pixel (3,3) lies in the overlap of all four windows, so it must map
    to exactly one row in each: the same defect is a valid exemplar in
    every window that saw it."""
    rect = (3 / 6, 3 / 6, 1 / 6, 1 / 6)
    assert rows_for_rects(6, 6, [rect], **_GEO) == [3, 6, 9, 12]


def test_rows_for_rects_degenerate_rect_is_skipped():
    assert rows_for_rects(6, 6, [(0.5, 0.5, 0.0, 0.1)], **_GEO) == []


def test_rows_for_rects_dedupes_overlapping_rects():
    once = rows_for_rects(6, 6, [(0.0, 0.0, 1.0, 1.0)], **_GEO)
    twice = rows_for_rects(6, 6, [(0.0, 0.0, 1.0, 1.0), (0.2, 0.2, 0.5, 0.5)], **_GEO)
    assert once == twice


def test_rows_for_rects_padded_small_image_uses_padded_offsets():
    """A 3x3 image pads up to one 4x4 window; a full-image rect covers the
    3-pixel extent, which still touches all 4 patches of that window."""
    assert expected_rows(3, 3, **_GEO) == 4
    assert rows_for_rects(3, 3, [(0.0, 0.0, 1.0, 1.0)], **_GEO) == [0, 1, 2, 3]


def test_pad_to_min_uses_reflect_padding_not_zeros():
    """Zero-padding would inject high-frequency edges that DINOv2 reads
    as anomaly. Reflect padding ensures the padded region is statistically
    similar to the original."""
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    out, _ = pad_to_min(img, window_size=518)
    # Bottom-right padded region should not be all zeros if reflect was used.
    pad_corner = out[-1, -1]
    assert (pad_corner > 0).any()
