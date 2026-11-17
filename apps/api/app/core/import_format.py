# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""How an imported source image is stored on disk, and the one place that decides it.

Two things live here on purpose.

The SETTING, under the ``import_format`` key of ``runtime_settings.json``:
projects differ in what is worth keeping. A line inspecting fine surface
defects wants the pixels untouched; a line running at volume wants the disk
to last the year. Before this, neither could say so -- the stored format
followed whatever the camera happened to emit, and a TIFF was expanded into
a barely-compressed PNG with no way to decline.

The DECISION, :func:`encode_source_image`: the teach route
(``cls_state.save_source_image``) and the store ingest route
(``cls_store._write_store_image``) each carried their own copy of "is this a
web format? then keep the bytes, else transcode to PNG". Two routes to the
same asset is this codebase's most productive source of bugs, so there is now
one function and both call it.

``jpg`` is the default because it is the right answer for most lines, and
because the alternative -- silently inheriting the camera's format -- is what
made disk usage unpredictable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import bank_images
from .runtime_settings import merge_runtime_settings, read_runtime_settings

SETTINGS_KEY = "import_format"
FORMATS = ("raw", "png", "jpg")
DEFAULTS: dict[str, Any] = {"format": "jpg", "jpg_quality": 92}
MIN_QUALITY, MAX_QUALITY = 60, 100


def read_import_settings() -> dict[str, Any]:
    """Current settings with defaults filled in; malformed values are ignored."""
    raw = read_runtime_settings().get(SETTINGS_KEY)
    out = dict(DEFAULTS)
    if isinstance(raw, dict):
        if raw.get("format") in FORMATS:
            out["format"] = raw["format"]
        try:
            out["jpg_quality"] = min(
                MAX_QUALITY,
                max(MIN_QUALITY, int(raw.get("jpg_quality", out["jpg_quality"]))),
            )
        except (TypeError, ValueError):
            pass
    return out


def save_import_settings(fmt: str, jpg_quality: int) -> dict[str, Any]:
    """Persist and return the resolved settings."""
    merge_runtime_settings(
        {
            SETTINGS_KEY: {
                "format": fmt if fmt in FORMATS else DEFAULTS["format"],
                "jpg_quality": min(MAX_QUALITY, max(MIN_QUALITY, int(jpg_quality))),
            }
        }
    )
    return read_import_settings()


def encode_source_image(filename: str, data: bytes) -> tuple[str, bytes]:
    """Return ``(name, bytes)`` for the copy of ``filename`` that goes to disk.

    The single owner of that decision, for every import route.

    ``raw`` keeps the uploaded bytes and the uploaded extension -- including
    formats a browser cannot draw. That is the trade the setting exists to
    let an operator make: a TIFF stays a TIFF, and its card falls back to the
    generated thumbnail rather than to the file itself.

    ``png`` and ``jpg`` transcode, EXCEPT when the upload is already in the
    target format -- re-encoding a JPEG as a JPEG would spend a generation of
    quality on nothing.

    Undecodable input is always kept verbatim: feature extraction has already
    accepted it by the time this runs, so refusing to store it here would
    lose an image the bank already has rows for.
    """
    safe = bank_images.safe_image_name(filename)
    cfg = read_import_settings()
    fmt = cfg["format"]
    suffix = Path(safe).suffix.lower()

    if fmt == "raw":
        return safe, data
    if fmt == "jpg" and suffix in (".jpg", ".jpeg"):
        return safe, data
    if fmt == "png" and suffix == ".png":
        return safe, data

    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return safe, data
        if fmt == "jpg":
            ok, buf = cv2.imencode(
                ".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpg_quality"])]
            )
            ext = ".jpg"
        else:
            ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            ext = ".png"
        if not ok:
            return safe, data
        return f"{Path(safe).stem or 'image'}{ext}", buf.tobytes()
    except Exception:
        return safe, data
