# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Image I/O and overlay utilities, robust against non-ASCII paths on Windows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite(path: Path, image: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def list_pairs(project_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return (image_path, mask_path or None) pairs from <project>/images and <project>/masks."""
    img_dir = project_dir / "images"
    msk_dir = project_dir / "masks"
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        mp = msk_dir / f"{img.stem}.png"
        pairs.append((img, mp if mp.exists() else None))
    return pairs


def is_clean(mask_path: Path | None) -> bool:
    """True iff mask exists and is all-zero (i.e. confirmed defect-free)."""
    if mask_path is None:
        return False
    m = imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return m is not None and bool((m == 0).all())


def overlay(image_bgr: np.ndarray, heatmap: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """JET colormap overlay; expects heatmap already at image resolution."""
    norm = np.clip((heatmap - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, color, 0.45, 0)


def overlay_diverging(image_bgr: np.ndarray, heatmap: np.ndarray, vmin: float, vthr: float) -> np.ndarray:
    """Full-image diverging score map centred on the verdict threshold.

    ``vmin`` (the OK images' typical level) maps to full blue, ``vthr``
    (the raw verdict threshold) to the neutral midpoint, and the red side
    saturates one ``vthr - vmin`` span above the threshold — so blue reads
    "OK-level", white reads "at the limit" and vermilion reads "NG-level"
    on the same absolute scale for every image. Colourblind-safe
    endpoints (Okabe–Ito blue #0072B2 / vermilion #D55E00) interpolated
    through a light neutral, never through purple.
    """
    span = max(vthr - vmin, 1e-6)
    # -1 (full blue) … 0 (neutral, at threshold) … +1 (full red)
    t = np.clip((heatmap.astype(np.float32) - vthr) / span, -1.0, 1.0)
    blue = np.array((178, 114, 0), dtype=np.float32)   # BGR #0072B2
    mid = np.array((235, 235, 235), dtype=np.float32)  # light neutral
    red = np.array((0, 94, 213), dtype=np.float32)     # BGR #D55E00
    neg = np.clip(-t, 0.0, 1.0)[..., None]
    pos = np.clip(t, 0.0, 1.0)[..., None]
    color = mid * (1.0 - neg - pos) + blue * neg + red * pos
    out = image_bgr.astype(np.float32) * 0.55 + color * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)
