# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Tests for ``clscore.io`` (read/write + heatmap overlay)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from clscore.io import IMAGE_EXTS, imread, imwrite, is_clean, list_pairs, overlay

# ---- overlay ---------------------------------------------------------------


def test_overlay_returns_same_shape_as_input():
    img = np.full((40, 60, 3), 128, dtype=np.uint8)
    hm = np.zeros((40, 60), dtype=np.float32)
    out = overlay(img, hm, vmin=0.0, vmax=1.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_overlay_clamps_below_vmin_and_above_vmax():
    """A heatmap value below vmin should be treated as 0 (cool colour),
    above vmax as 1 (hot colour). Without the clamp, JET cycles back into
    cool colours past 1.0 — which would mis-paint extreme anomalies."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    hm_under = np.full((10, 10), -100.0, dtype=np.float32)
    hm_over = np.full((10, 10), 100.0, dtype=np.float32)
    out_under = overlay(img, hm_under, vmin=0.0, vmax=1.0)
    out_over = overlay(img, hm_over, vmin=0.0, vmax=1.0)
    # JET 0 is mostly blue, JET 1 is mostly red -> different colours.
    assert not np.array_equal(out_under, out_over)


def test_overlay_collapsed_range_does_not_divide_by_zero():
    """vmin == vmax used to silently produce inf/nan colours; the eps in
    the denominator keeps the output finite."""
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    hm = np.full((8, 8), 0.5, dtype=np.float32)
    out = overlay(img, hm, vmin=0.5, vmax=0.5)
    assert np.isfinite(out).all()


# ---- imread / imwrite (Windows non-ASCII safe round-trip) ------------------


def test_imread_imwrite_round_trip(tmp_path):
    img = np.random.randint(0, 255, size=(20, 30, 3), dtype=np.uint8)
    p = tmp_path / "round.png"
    assert imwrite(p, img) is True
    back = imread(p)
    assert back is not None
    assert back.shape == img.shape


def test_imread_returns_none_on_missing_file(tmp_path):
    """The fallback path uses ``np.fromfile`` which raises on a missing
    file; the wrapper swallows that and returns None so callers don't
    need a separate try/except per call."""
    assert imread(tmp_path / "nope.png") is None


def test_imread_handles_non_ascii_path(tmp_path):
    """The whole reason imread/imwrite exist: cv2.imread silently
    chokes on Windows when the path contains non-ASCII characters.
    ``np.fromfile`` + ``imdecode`` works around it."""
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    p = tmp_path / "テスト.png"
    assert imwrite(p, img) is True
    back = imread(p)
    assert back is not None and back.shape == img.shape


# ---- list_pairs / is_clean -------------------------------------------------


def test_list_pairs_pairs_images_with_masks(tmp_path):
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()
    for i, ext in enumerate([".png", ".jpg"]):
        cv2.imwrite(str(img_dir / f"a{i}{ext}"), np.zeros((4, 4, 3), np.uint8))
    cv2.imwrite(str(msk_dir / "a0.png"), np.zeros((4, 4), np.uint8))
    # Only a0 has a mask; a1 should pair with None.
    pairs = list_pairs(tmp_path)
    assert len(pairs) == 2
    img_names = sorted(p[0].name for p in pairs)
    assert img_names == ["a0.png", "a1.jpg"]
    pair_dict = {p[0].stem: p[1] for p in pairs}
    assert pair_dict["a0"] is not None
    assert pair_dict["a1"] is None


def test_list_pairs_skips_non_image_extensions(tmp_path):
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()
    cv2.imwrite(str(img_dir / "good.png"), np.zeros((4, 4, 3), np.uint8))
    (img_dir / "junk.txt").write_text("not an image", encoding="utf-8")
    pairs = list_pairs(tmp_path)
    assert len(pairs) == 1


def test_is_clean_recognises_all_zero_mask(tmp_path):
    p = tmp_path / "all_zero.png"
    cv2.imwrite(str(p), np.zeros((10, 10), np.uint8))
    assert is_clean(p) is True


def test_is_clean_rejects_mask_with_any_nonzero():
    """A single non-zero pixel means there's annotation, so this is
    not a 'confirmed clean' image — must return False."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mask.png"
        m = np.zeros((10, 10), np.uint8)
        m[3, 4] = 1
        cv2.imwrite(str(p), m)
        assert is_clean(p) is False


def test_is_clean_returns_false_for_none():
    assert is_clean(None) is False


def test_image_exts_covers_common_formats():
    """Sanity guard — extending the supported list silently can
    accidentally pull in raw / heif / etc. that cv2 can't decode."""
    expected = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    assert set(IMAGE_EXTS) == expected
