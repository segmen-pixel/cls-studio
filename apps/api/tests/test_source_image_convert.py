# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""What an imported image becomes on disk, under each import-format setting.

The stored format used to follow whatever the camera emitted: web formats
were kept, everything else became a barely-compressed PNG, and nobody could
say otherwise. These pin the setting that replaced that, and the property
that made it safe to add -- both import routes ask the same function, so the
teach route and the store ingest route cannot answer differently.

Every test restores the defaults: the settings file is process-global and
shared with the rest of the session.
"""

from __future__ import annotations

import contextlib

import cv2
import numpy as np

from app.core.cls_state import get_state
from app.core.import_format import (
    DEFAULTS,
    encode_source_image,
    read_import_settings,
    save_import_settings,
)


@contextlib.contextmanager
def _restored_defaults():
    try:
        yield
    finally:
        save_import_settings(**{"fmt": DEFAULTS["format"], "jpg_quality": DEFAULTS["jpg_quality"]})


def _img(h=24, w=30, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _encoded(ext: str, seed: int = 0) -> bytes:
    ok, buf = cv2.imencode(ext, _img(seed=seed))
    assert ok, f"opencv build lacks {ext} encode"
    return buf.tobytes()


def test_the_default_is_jpg():
    assert read_import_settings() == {"format": "jpg", "jpg_quality": 92}


def test_settings_round_trip_and_clamp():
    with _restored_defaults():
        assert save_import_settings("png", 100)["format"] == "png"
        assert save_import_settings("jpg", 5)["jpg_quality"] == 60  # clamped up
        assert save_import_settings("jpg", 999)["jpg_quality"] == 100  # clamped down
        assert save_import_settings("nonsense", 90)["format"] == "jpg"  # falls back


def test_tiff_follows_the_setting():
    """A TIFF is the case that forced a transcode; which one is now a choice."""
    tiff = _encoded(".tiff")
    with _restored_defaults():
        save_import_settings("jpg", 92)
        assert encode_source_image("wafer.tif", tiff)[0].lower().endswith(".jpg")
        save_import_settings("png", 92)
        assert encode_source_image("wafer.tif", tiff)[0].lower().endswith(".png")
        save_import_settings("raw", 92)
        name, data = encode_source_image("wafer.tif", tiff)
        assert name.lower().endswith(".tif")
        assert data == tiff, "raw must not re-encode"


def test_input_already_in_the_target_format_is_not_re_encoded():
    """Re-encoding a JPEG as a JPEG spends a generation of quality on nothing."""
    jpg, png = _encoded(".jpg", seed=1), _encoded(".png", seed=2)
    with _restored_defaults():
        save_import_settings("jpg", 92)
        assert encode_source_image("photo.jpg", jpg) == ("photo.jpg", jpg)
        save_import_settings("png", 92)
        assert encode_source_image("plate.png", png) == ("plate.png", png)


def test_jpg_setting_shrinks_a_png_source():
    """The point of the setting for a customer who wants the disk to last."""
    png = _encoded(".png", seed=3)
    with _restored_defaults():
        save_import_settings("jpg", 75)
        name, data = encode_source_image("plate.png", png)
        assert name.lower().endswith(".jpg")
        assert len(data) < len(png)


def test_quality_changes_the_bytes():
    png = _encoded(".png", seed=4)
    with _restored_defaults():
        save_import_settings("jpg", 60)
        small = encode_source_image("plate.png", png)[1]
        save_import_settings("jpg", 100)
        large = encode_source_image("plate.png", png)[1]
        assert len(small) < len(large)


def test_undecodable_input_is_kept_verbatim():
    """Feature extraction has already accepted it; refusing here loses an image
    the bank has rows for."""
    junk = b"not an image at all"
    with _restored_defaults():
        save_import_settings("jpg", 92)
        assert encode_source_image("weird.bin", junk) == ("weird.bin", junk)


def test_the_teach_route_goes_through_the_same_encoder(client, project_id):
    """End to end: save_source_image must not carry its own transcode rule."""
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    state = get_state()
    tiff = _encoded(".tiff", seed=5)
    with _restored_defaults():
        save_import_settings("png", 92)
        name = state.save_source_image("normal", "wafer.tif", tiff)
        assert name.lower().endswith(".png")
        path = state.images_dir("normal") / name
        saved = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        assert saved is not None and saved.shape[:2] == (24, 30)

        save_import_settings("raw", 92)
        name = state.save_source_image("normal", "wafer2.tif", tiff)
        assert name.lower().endswith(".tif")
        assert (state.images_dir("normal") / name).read_bytes() == tiff


def test_endpoint_round_trip(client):
    with _restored_defaults():
        r = client.put("/api/v1/system/import", json={"format": "raw", "jpg_quality": 80})
        assert r.status_code == 200
        body = r.json()
        assert body["format"] == "raw" and body["jpg_quality"] == 80
        assert body["formats"] == ["raw", "png", "jpg"]
        assert client.get("/api/v1/system/import").json()["format"] == "raw"


def test_endpoint_rejects_an_unknown_format(client):
    with _restored_defaults():
        assert client.put(
            "/api/v1/system/import", json={"format": "webp", "jpg_quality": 90}
        ).status_code == 422
        assert client.put(
            "/api/v1/system/import", json={"format": "jpg", "jpg_quality": 10}
        ).status_code == 422
