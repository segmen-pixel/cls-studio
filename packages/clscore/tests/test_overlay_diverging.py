# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Diverging score-map overlay: blue at OK level, neutral at threshold, red above.

``overlay_diverging`` colours the whole image on an absolute scale anchored
to the separation check: ``vmin`` (OK median) → blue, ``vthr`` (raw verdict
threshold) → light neutral, one span above → saturated vermilion. Endpoints
are colourblind-safe and the interpolation passes through neutral, never
purple.
"""

from __future__ import annotations

import numpy as np

from clscore.io import overlay, overlay_diverging


def _flat_image(v: int = 120, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), v, dtype=np.uint8)


def test_ok_level_is_blue_threshold_neutral_and_above_red():
    img = _flat_image()
    hm = np.full((8, 8), 10.0, dtype=np.float32)  # everywhere at vmin (OK level)
    hm[2, 2] = 20.0  # at threshold
    hm[4, 4] = 30.0  # one span above → full red
    out = overlay_diverging(img, hm, vmin=10.0, vthr=20.0)
    b_ok, g_ok, r_ok = out[0, 0].astype(int)
    assert b_ok > r_ok  # OK level → blue dominant
    b_thr, g_thr, r_thr = out[2, 2].astype(int)
    assert abs(int(b_thr) - int(r_thr)) < 12  # threshold → near-neutral
    b_ng, g_ng, r_ng = out[4, 4].astype(int)
    assert r_ng > b_ng  # above threshold → red dominant


def test_scale_is_absolute_not_per_image():
    # Same score value must get the same colour in two different images.
    hm_a = np.full((8, 8), 25.0, dtype=np.float32)
    hm_b = np.full((8, 8), 25.0, dtype=np.float32)
    hm_b[0, 0] = 100.0  # extra outlier must not shift the rest
    a = overlay_diverging(_flat_image(), hm_a, vmin=10.0, vthr=20.0)
    b = overlay_diverging(_flat_image(), hm_b, vmin=10.0, vthr=20.0)
    assert (a[4, 4] == b[4, 4]).all()


def test_saturation_clamps_beyond_one_span():
    img = _flat_image()
    hm = np.full((8, 8), 30.0, dtype=np.float32)
    hm[1, 1] = 300.0  # far beyond the red end
    out = overlay_diverging(img, hm, vmin=10.0, vthr=20.0)
    assert (out[1, 1] == out[5, 5]).all()  # both clamp to full red


def test_jet_overlay_still_relative():
    img = _flat_image()
    hm = np.zeros((8, 8), dtype=np.float32)
    out = overlay(img, hm, vmin=0.0, vmax=1.0)
    assert not (out == img).all()
