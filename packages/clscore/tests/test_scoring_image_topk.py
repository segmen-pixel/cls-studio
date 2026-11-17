# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""image_topk_mean: the image-level verdict statistic for inference.

Must equal the separation check's top-k mean over per-window patch rows —
NO per-pixel max-merge — so a threshold picked on the Develop tab applies
to the Operator verdict without translation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from clscore.scoring import BOOST_FLOOR, image_topk_mean


def _w(raw, crit=None, neg=None):
    return {
        "bank_topk_mean": np.asarray(raw, dtype=np.float32),
        "critical_min": {"d": np.asarray(crit, dtype=np.float32)} if crit is not None else None,
        "negative_min": {"d": np.asarray(neg, dtype=np.float32)} if neg is not None else None,
    }


def test_topk_concatenates_windows_without_max_merge():
    # Top-2 across BOTH windows: 5 and 4 — a per-pixel max-merge would lose
    # values from overlapping windows; concatenation must keep them all.
    assert image_topk_mean([_w([1, 5, 3]), _w([4, 2])], k=2) == pytest.approx((5 + 4) / 2)


def test_topk_clamps_k_to_patch_count():
    assert image_topk_mean([_w([2, 4])], k=10) == pytest.approx(3.0)


def test_topk_alpha_boosts_patch_near_exemplar():
    # Patch 0 sits at distance 0.1 from an exemplar: with a large-enough
    # alpha it must overtake patch 1 (raw 2, exemplar far away). Since the
    # boost was bounded (alpha / (BOOST_FLOOR + d), max contribution alpha),
    # alpha now has to exceed the raw-score gap to flip the ranking — that
    # is the point: alpha is interpretable in score units.
    w = _w([1.0, 2.0], crit=[0.1, 100.0])
    assert image_topk_mean([w], k=1) == pytest.approx(2.0)
    assert image_topk_mean([w], alpha=5.0, k=1) == pytest.approx(
        1.0 + 5.0 / (BOOST_FLOOR + 0.1), rel=1e-4)
    # ...and the boost can never exceed alpha, even at distance zero.
    w0 = _w([1.0, 2.0], crit=[0.0, 100.0])
    assert image_topk_mean([w0], alpha=5.0, k=1) <= 2.0 + 5.0


def test_topk_alpha_zero_ignores_critical():
    w = _w([1.0, 2.0], crit=[0.001, 0.001])
    assert image_topk_mean([w], alpha=0.0, k=2) == pytest.approx(1.5)


def test_topk_beta_suppresses_patch_near_negative_exemplar():
    w = _w([5.0, 5.0], neg=[0.5, 1000.0])
    boosted = image_topk_mean([w], beta=1.0, k=2)
    assert boosted < 5.0  # patch 0 pulled down by the negative term


def test_topk_empty_windows_is_nan():
    assert math.isnan(image_topk_mean([]))
