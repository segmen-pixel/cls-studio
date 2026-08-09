# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Sliding-window geometry: window placement and reflect-padding."""

from __future__ import annotations

import cv2
import numpy as np

WINDOW_SIZE: int = 518
WINDOW_STRIDE: int = 388
DINO_PATCH: int = 14


def sw_offsets(
    height: int,
    width: int,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> list[tuple[int, int]]:
    """Top-left positions covering an image; the last row/column is anchored to the edge."""
    if height < window_size or width < window_size:
        return [(0, 0)]
    ys = list(range(0, height - window_size + 1, stride))
    if ys[-1] != height - window_size:
        ys.append(height - window_size)
    xs = list(range(0, width - window_size + 1, stride))
    if xs[-1] != width - window_size:
        xs.append(width - window_size)
    return [(y, x) for y in ys for x in xs]


def pad_to_min(
    image: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad so that both sides are at least window_size; returns (padded, (H, W))."""
    h, w = image.shape[:2]
    pad_h = max(0, window_size - h)
    pad_w = max(0, window_size - w)
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return image, (h, w)


def expected_rows(
    height: int,
    width: int,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    patch: int = DINO_PATCH,
) -> int:
    """Number of bank rows one image of this size contributes.

    Mirrors ``extract_image_features_for_bank``: windows over the padded
    image, ``(window_size // patch)**2`` patch tokens each. Callers use this
    to reject row-index math against a bank entry whose stored row count was
    produced by a different geometry (legacy window/stride settings).
    """
    win_p = window_size // patch
    offsets = sw_offsets(
        max(height, window_size), max(width, window_size),
        window_size=window_size, stride=stride,
    )
    return len(offsets) * win_p * win_p


def rows_for_rects(
    height: int,
    width: int,
    rects: list[tuple[float, float, float, float]],
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    patch: int = DINO_PATCH,
) -> list[int]:
    """Map normalized ``(x, y, w, h)`` rectangles to flat bank-row indices.

    Row order mirrors ``extract_image_features_for_bank``: windows in
    ``sw_offsets`` order over the padded image, each contributing
    ``(window_size // patch)**2`` rows row-major, i.e.
    ``row = win_idx * win_p**2 + py * win_p + px``. A row is selected when
    its patch's pixel square intersects any rectangle; a pixel covered by
    several overlapping windows therefore selects one row per window, which
    is exactly the set of bank rows that pixel produced.

    Rectangles are normalized to the *original* (pre-pad) image size, so a
    UI can send fractions of the displayed image without knowing the SW
    geometry. Degenerate rectangles are skipped; the result is sorted and
    de-duplicated.
    """
    win_p = window_size // patch
    offsets = sw_offsets(
        max(height, window_size), max(width, window_size),
        window_size=window_size, stride=stride,
    )
    out: set[int] = set()
    for rx, ry, rw, rh in rects:
        x0, y0 = rx * width, ry * height
        x1, y1 = x0 + rw * width, y0 + rh * height
        if x1 <= x0 or y1 <= y0:
            continue
        for wi, (wy, wx) in enumerate(offsets):
            py0 = max(0, int(np.floor((y0 - wy) / patch)))
            py1 = min(win_p - 1, int(np.ceil((y1 - wy) / patch)) - 1)
            px0 = max(0, int(np.floor((x0 - wx) / patch)))
            px1 = min(win_p - 1, int(np.ceil((x1 - wx) / patch)) - 1)
            if py0 > py1 or px0 > px1:
                continue
            base = wi * win_p * win_p
            for py in range(py0, py1 + 1):
                out.update(base + py * win_p + px for px in range(px0, px1 + 1))
    return sorted(out)
