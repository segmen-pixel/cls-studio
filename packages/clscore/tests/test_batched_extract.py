# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Batched multi-image extraction must equal per-image extraction.

The bulk-teach path packs every image's sliding windows into shared forwards
to fill the GPU; the resulting per-image features must be identical to running
extract_image_features_for_bank one image at a time, or banks would drift.
Uses a deterministic stub backbone so the equality is exact (real DINOv2 is a
GPU/network dependency, out of scope for a unit test).
"""
from __future__ import annotations

import numpy as np
import torch

from clscore.feature_extractor import (
    WINDOW_SIZE,
    extract_image_features_for_bank,
    extract_images_features_batched,
)


class _StubBackbone:
    """forward_features -> a deterministic function of each window's pixels, so
    a window yields the same tokens whether run alone or inside a big batch."""

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)  # advertise fp32 to the caller

    def forward_features(self, x: torch.Tensor) -> dict:
        b = x.shape[0]
        side = WINDOW_SIZE // 14
        d = 6
        per_window = x.mean(dim=(1, 2, 3))  # [B]
        tok = per_window.view(b, 1, 1).expand(b, side * side, d).contiguous()
        return {"x_norm_patchtokens": tok}


def _img(h: int, w: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_batched_equals_per_image():
    model = _StubBackbone()
    # Different sizes -> different window counts -> exercises the split indexing.
    imgs = [_img(600, 800, 1), _img(1123, 794, 2), _img(520, 520, 3)]
    per_image = [extract_image_features_for_bank(model, im, "cpu") for im in imgs]
    batched = extract_images_features_batched(model, imgs, "cpu", max_batch=8)
    assert len(batched) == len(per_image)
    for a, b in zip(per_image, batched):
        assert a.shape == b.shape
        assert np.array_equal(a, b)


def test_empty_batch_returns_empty():
    assert extract_images_features_batched(_StubBackbone(), [], "cpu") == []
