#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Assemble the frames from docs-demo-gif.mjs into docs/images/hero.gif.

A stepped slideshow, not a video: each frame holds for a few seconds, the
NG-heatmap frame longest — it is the product's whole point.

Usage:
  python e2e/_tools/assemble-gif.py [--frames DIR] [--out FILE] [--width 960]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
# frame index -> hold time (ms)
DURATIONS = [1800, 2600, 2200, 1800, 3600]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path, default=HERE.parent / "screenshots" / "docs" / "gif")
    ap.add_argument("--out", type=Path,
                    default=HERE.parents[3] / "docs" / "images" / "hero.gif")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--colors", type=int, default=128)
    args = ap.parse_args()

    paths = sorted(args.frames.glob("frame_*.png"))
    if not paths:
        print(f"no frames in {args.frames}")
        return 2
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        h = round(im.height * args.width / im.width)
        im = im.resize((args.width, h), Image.LANCZOS)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=args.colors))
    durations = (DURATIONS + [2000] * len(frames))[: len(frames)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    print(f"{args.out} <- {len(frames)} frames, {args.out.stat().st_size / 1e3:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
