# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Heatmap alignment: a hot patch must peak where that patch actually is.

The grid is indexed in padded pixel space while the returned map is at the
original image's resolution, so the conversion has to account for the padding.
It did not: on a 400x300 image (which pads out to 518x518) the peak landed
62 px above and 47 px left of the defect. Small images are exactly the ones
where an operator is most likely to notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from clscore.scoring import compose_score_grid
from clscore.sw import DINO_PATCH, WINDOW_SIZE, pad_to_min

WIN_P = WINDOW_SIZE // DINO_PATCH

# One patch of slack: a patch centre sits at x.5 pixels, and argmax has to
# pick one of the two neighbours. Anything beyond this is a real shift.
TOLERANCE = DINO_PATCH


def _peak_offset(orig_h: int, orig_w: int, frac_y: float, frac_x: float) -> tuple[float, float]:
    """Distance between where a hot patch is and where the map peaks, in px."""
    padded, (oh, ow) = pad_to_min(np.zeros((orig_h, orig_w, 3), np.uint8))
    grid_h = padded.shape[0] // DINO_PATCH
    grid_w = padded.shape[1] // DINO_PATCH

    ty = int(frac_y * orig_h) // DINO_PATCH
    tx = int(frac_x * orig_w) // DINO_PATCH
    assert ty < WIN_P and tx < WIN_P, "test only places peaks inside the first window"

    flat = np.zeros(WIN_P * WIN_P, np.float32)
    flat[ty * WIN_P + tx] = 100.0
    window = {
        "y": 0, "x": 0, "win_p": WIN_P, "bank_topk_mean": flat,
        "critical_min": None, "negative_min": None,
    }

    full, grid = compose_score_grid([window], (grid_h, grid_w), (oh, ow), 0.0, 0.0)
    assert full.shape == (orig_h, orig_w), f"map must be at image resolution, got {full.shape}"
    py, px = np.unravel_index(int(np.argmax(full)), full.shape)
    want_y = ty * DINO_PATCH + DINO_PATCH / 2
    want_x = tx * DINO_PATCH + DINO_PATCH / 2
    return abs(py - want_y), abs(px - want_x)


@pytest.mark.parametrize(
    ("h", "w", "label"),
    [
        (300, 400, "both sides under the window: pads to 518x518"),
        (300, 900, "one side under the window"),
        (800, 1000, "no padding, not a multiple of the patch size"),
        (798, 994, "no padding, exact multiple of the patch size"),
        (518, 518, "exactly one window"),
    ],
)
def test_peak_lands_on_the_hot_patch(h: int, w: int, label: str) -> None:
    dy, dx = _peak_offset(h, w, 0.5, 0.5)
    assert dy <= TOLERANCE, f"{label}: vertical shift {dy:.0f}px (> {TOLERANCE})"
    assert dx <= TOLERANCE, f"{label}: horizontal shift {dx:.0f}px (> {TOLERANCE})"


@pytest.mark.parametrize(("frac_y", "frac_x"), [(0.1, 0.1), (0.5, 0.5), (0.8, 0.8)])
def test_alignment_holds_across_the_frame_on_a_padded_image(frac_y: float, frac_x: float) -> None:
    # A uniform scale error shows up as an offset that grows with distance from
    # the origin, so checking one point is not enough to catch it.
    dy, dx = _peak_offset(300, 400, frac_y, frac_x)
    assert dy <= TOLERANCE and dx <= TOLERANCE, f"({frac_y}, {frac_x}): dy={dy:.0f} dx={dx:.0f}"


def test_map_covers_the_whole_image_including_truncated_edges() -> None:
    # 1000 // 14 leaves 6 px the grid never reaches; those pixels still need a
    # finite score rather than a hole.
    padded, (oh, ow) = pad_to_min(np.zeros((800, 1000, 3), np.uint8))
    gh, gw = padded.shape[0] // DINO_PATCH, padded.shape[1] // DINO_PATCH
    flat = np.full(WIN_P * WIN_P, 5.0, np.float32)
    window = {"y": 0, "x": 0, "win_p": WIN_P, "bank_topk_mean": flat,
              "critical_min": None, "negative_min": None}
    full, _ = compose_score_grid([window], (gh, gw), (oh, ow), 0.0, 0.0)
    assert full.shape == (800, 1000)
    assert np.isfinite(full).all()
    # The right and bottom edges come from replication, so they must match the
    # last covered column/row rather than being zero.
    assert full[:, -1].min() > 0.0
    assert full[-1, :].min() > 0.0
