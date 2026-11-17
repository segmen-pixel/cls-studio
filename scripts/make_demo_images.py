#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Generate a synthetic demo image set — try cls-studio with no data at hand.

Writes brushed-metal-looking OK plates plus a few NG variants (scratches and
a stain) and two un-taught probe images. Enough to walk the whole First Run
guide: teach the ``ok_*`` files, teach the ``ng_*`` files as NG, run the
separation check, then drop the ``probe_*`` files on the Inspect tab.

The same set backs the documentation screenshots
(``docs/contributing/screenshots.md``), so docs never need customer images.

Usage:
  python scripts/make_demo_images.py [--out DIR] [--ok N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 1024, 768


def plate(seed: int) -> np.ndarray:
    """One OK plate: brushed texture, mild vignette, per-lot tint, fixture holes."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W), 148.0)
    streaks = cv2.GaussianBlur(rng.normal(0, 14, (H, W)), (1, 31), 0)
    base += streaks + rng.normal(0, 3.5, (H, W)) + rng.uniform(-9, 9)
    yy, xx = np.mgrid[0:H, 0:W]
    vign = ((xx - W / 2) ** 2 / (W / 2) ** 2 + (yy - H / 2) ** 2 / (H / 2) ** 2)
    base -= vign * 22
    img = np.clip(base, 0, 255).astype(np.uint8)
    bgr = cv2.merge([img, img, np.clip(img * 1.03, 0, 255).astype(np.uint8)])
    for cx, cy in ((130, 120), (W - 130, H - 120)):
        cv2.circle(bgr, (cx, cy), 26, (70, 70, 74), -1)
        cv2.circle(bgr, (cx, cy), 26, (105, 105, 110), 2)
    return bgr


def scratch(img: np.ndarray, seed: int) -> np.ndarray:
    """A jagged scratch with a glint edge."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    pts = [(int(rng.uniform(200, 500)), int(rng.uniform(200, 500)))]
    ang = rng.uniform(0, np.pi)
    for _ in range(int(rng.uniform(6, 10))):
        ang += rng.normal(0, 0.35)
        r = rng.uniform(28, 60)
        pts.append((int(pts[-1][0] + r * np.cos(ang)), int(pts[-1][1] + r * np.sin(ang))))
    for a, b in zip(pts, pts[1:]):
        cv2.line(out, a, b, (52, 52, 58), int(rng.uniform(2, 4)), cv2.LINE_AA)
        cv2.line(out, a, b, (200, 200, 205), 1, cv2.LINE_AA)
    return out


def stain(img: np.ndarray, seed: int) -> np.ndarray:
    """A soft brownish blotch."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    cx, cy = int(rng.uniform(300, W - 300)), int(rng.uniform(220, H - 220))
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((xx - cx) / 90.0) ** 2 + ((yy - cy) / 60.0) ** 2))
    out[..., 0] -= blob * 45
    out[..., 1] -= blob * 30
    out[..., 2] -= blob * 12
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("demo_images"),
                    help="output directory (default: ./demo_images)")
    ap.add_argument("--ok", type=int, default=12, help="number of OK plates (default 12)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.ok):
        cv2.imwrite(str(args.out / f"ok_{i:02d}.png"), plate(seed=i))
    for i in range(3):
        cv2.imwrite(str(args.out / f"ng_scratch_{i}.png"), scratch(plate(seed=100 + i), seed=200 + i))
    cv2.imwrite(str(args.out / "ng_stain_0.png"), stain(plate(seed=110), seed=210))
    cv2.imwrite(str(args.out / "probe_ok.png"), plate(seed=300))
    cv2.imwrite(str(args.out / "probe_ng.png"), scratch(plate(seed=301), seed=302))
    n = len(list(args.out.glob("*.png")))
    print(f"wrote {n} images to {args.out.resolve()}")
    print("teach the ok_* files as OK, the ng_* files as NG, then inspect the probe_* files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
