# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Source-image storage transcodes browser-undisplayable formats to PNG.

Teaching saves a copy of each image for the thumbnail grid. Browsers can't
render TIFF, so the raw file showed broken; the store now transcodes non-web
formats to (lossless) PNG while leaving web formats untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.core.cls_state import get_state


def _img(h=24, w=30, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_tiff_source_image_transcoded_to_png(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    state = get_state()
    ok, buf = cv2.imencode(".tiff", _img())
    assert ok, "opencv build lacks TIFF encode"
    name = state.save_source_image("normal", "wafer.tif", buf.tobytes())
    assert name.lower().endswith(".png")
    path = state.images_dir("normal") / name
    saved = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    assert saved is not None and saved.shape[:2] == (24, 30)


def test_web_formats_kept_as_is(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    state = get_state()
    ok, buf = cv2.imencode(".jpg", _img(seed=1))
    assert ok
    name = state.save_source_image("normal", "photo.jpg", buf.tobytes())
    assert name.lower().endswith(".jpg")  # displayable already → not inflated to PNG
    # bytes are stored verbatim (no re-encode)
    stored = (state.images_dir("normal") / name).read_bytes()
    assert stored == buf.tobytes()
