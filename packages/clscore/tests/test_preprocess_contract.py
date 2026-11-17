# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""The preprocessing chain is a contract with every bank ever written.

A bank row is a point in a space defined entirely by these constants and the
order of the operations that use them. Change any of it and the rows already
on disk stop being comparable to anything computed afterwards -- with no
exception, no warning, and no failing request. Only the scores move.

So this file pins the chain rather than the code that implements it: a
golden vector, the exact arithmetic, and the fact that the extractor and any
exported runtime read the same numbers from one place.
"""

from __future__ import annotations

import hashlib

import numpy as np

from clscore.feature_extractor import _IMAGENET_MEAN, _IMAGENET_STD, _normalize_window
from clscore.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PIXEL_DIVISOR,
    normalize_window,
    preprocess_spec,
)
from clscore.sw import WINDOW_SIZE


def _window(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)


def test_the_chain_still_produces_the_same_numbers():
    """Golden digest. If this moves, every bank on disk has been invalidated."""
    out = normalize_window(_window())
    assert out.shape == (3, WINDOW_SIZE, WINDOW_SIZE)
    assert out.dtype == np.float32
    digest = hashlib.sha256(out.tobytes()).hexdigest()
    assert digest == "5cb03dfc8011e7d2a7ffc2328bfaf94bcc6ce955bb96bb0fd38427c2756e321a"


def test_scaling_divides_and_does_not_multiply_by_a_reciprocal():
    """1/255 is not representable, so the two forms disagree on 126 of the
    256 possible channel values. The banks were built with the division."""
    x = np.arange(256, dtype=np.float32)
    assert not np.array_equal(x / PIXEL_DIVISOR, x * (1.0 / PIXEL_DIVISOR))
    win = np.tile(np.arange(256, dtype=np.uint8)[:2], (WINDOW_SIZE, WINDOW_SIZE // 2, 3))
    win = np.ascontiguousarray(win[:WINDOW_SIZE, :WINDOW_SIZE, :3])
    got = normalize_window(win)
    rgb = win[:, :, ::-1].astype(np.float32) / 255.0
    want = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    assert np.array_equal(got, want)


def test_the_extractor_reads_the_same_numbers_rather_than_a_copy():
    assert _IMAGENET_MEAN is IMAGENET_MEAN
    assert _IMAGENET_STD is IMAGENET_STD
    win = _window(11)
    assert np.array_equal(_normalize_window(win).numpy(), normalize_window(win))


def test_the_exported_spec_states_what_the_code_actually_does():
    """An exported package is checked against this, so it cannot be prose."""
    spec = preprocess_spec()
    assert spec["input_size"] == [WINDOW_SIZE, WINDOW_SIZE]
    assert spec["color_space"] == "RGB"
    assert spec["pixel_divisor"] == PIXEL_DIVISOR
    assert spec["normalize"]["mean"] == [float(v) for v in IMAGENET_MEAN]
    assert spec["normalize"]["std"] == [float(v) for v in IMAGENET_STD]
    assert spec["layout"] == "CHW"
    assert len(set(spec["normalize"]["std"])) == 3
