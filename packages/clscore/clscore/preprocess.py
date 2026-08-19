# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""The feature-space contract, written down once.

A bank row is only comparable to a query row if both were produced by the
same preprocessing. Nothing enforces that at runtime: feed a window through
a slightly different chain and the forward pass still succeeds, the scores
still come out, and every one of them is quietly wrong. The failure has no
exception attached to it, which is exactly why the constants cannot be
restated per call site.

So the server path and any exported runtime -- Core ML on a phone, most
immediately -- both read the numbers from here, and :func:`preprocess_spec`
is what gets written into an export so a device can be checked against the
same values rather than trusting a comment.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "COLOR_SPACE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PIXEL_DIVISOR",
    "normalize_window",
    "preprocess_spec",
]

# ImageNet statistics, per channel in RGB order. The three standard
# deviations differ, which rules out any representation that carries a
# single scalar scale -- Core ML's ImageType among them. A host that folds
# them to one averaged value introduces a per-channel error that a
# classifier would shrug off and a nearest-neighbour bank cannot: it
# displaces every query relative to every taught row.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Images arrive from OpenCV as BGR uint8; the encoder wants RGB in [0, 1].
COLOR_SPACE = "RGB"
# Divide by this; do NOT multiply by its reciprocal. 1/255 is not exactly
# representable in binary floating point, so x * (1/255) and x / 255 differ
# in the last bit for most inputs. Every bank on disk was built with the
# division, and a bank is only comparable to itself.
PIXEL_DIVISOR = 255.0


def normalize_window(window_bgr: np.ndarray) -> np.ndarray:
    """One BGR uint8 window -> CHW float32 array in ImageNet normalization.

    Returns a plain array rather than a tensor so a caller with no torch --
    the Core ML parity probe, for one -- can use the identical function
    instead of a reimplementation that drifts.
    """
    rgb = cv2.cvtColor(window_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / PIXEL_DIVISOR
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def preprocess_spec() -> dict:
    """The chain above as data, for manifests and for host-side checking.

    Key names match what the iOS host already decodes into its
    ``PreprocessFile``, so a host that reads this needs no new parser.
    """
    from .sw import WINDOW_SIZE

    return {
        "input_size": [WINDOW_SIZE, WINDOW_SIZE],
        "resize_mode": "none",
        "color_space": COLOR_SPACE,
        "pixel_divisor": PIXEL_DIVISOR,
        # These read as 0.48500001430511475 rather than 0.485 on purpose:
        # they are the float32 values the server actually computed with,
        # widened for JSON. Rounding them to the literals a reader expects
        # would hand a double-precision host a number the bank was never
        # built with. Do not "tidy" them.
        "normalize": {
            "mean": [float(v) for v in IMAGENET_MEAN],
            "std": [float(v) for v in IMAGENET_STD],
        },
        "normalize_note": "float32 values widened to double; use as-is",
        "layout": "CHW",
        "dtype": "float32",
    }
