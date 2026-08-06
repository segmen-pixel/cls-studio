# LGPL upstream notices

Cls-Studio is licensed under Apache License 2.0. The installer bundles one
dynamically-linked component governed by the GNU Lesser General Public
License (LGPL). The full text of that license is reproduced verbatim in
this directory:

| File | Used by |
|---|---|
| `LGPL-2.1.txt` | FFmpeg (LGPL build, shipped inside `opencv-python-headless`) |

## How obligations are satisfied

The LGPL grants users the right to **replace** the bundled library with
their own build. To make that practical we:

1. Ship the library as a *dynamically-linked* DLL (no static linking), so
   the user can drop in a compatible build without rebuilding Cls-Studio.
2. Reproduce the full LGPL text alongside the binary (this directory).
3. List the upstream source URL below so users can fetch the matching
   source.

| Component | Version source | LGPL track | Upstream source |
|---|---|---|---|
| FFmpeg (LGPL build, via OpenCV) | https://ffmpeg.org/download.html | LGPL-2.1+ | https://github.com/FFmpeg/FFmpeg |

If you need the exact source revision matching a specific Cls-Studio
release, open an issue at
https://github.com/segmen-pixel/cls-studio/issues citing the release tag
and we will pin and publish it.
