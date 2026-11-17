# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Operator capture storage for the active project.

Captures are the raw frames the operator grabs at the line — saved under
``<project_dir>/captures/`` so the Images tab can show thumbnails without
re-encoding bank patches. Bank correctness never depends on a capture
existing; persistence is best-effort.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core.cls_state import get_state
from ..core.security import read_upload

router = APIRouter(tags=["captures"])

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class CaptureSaved(BaseModel):
    name: str


class CaptureList(BaseModel):
    captures: list[str] = Field(default_factory=list)


def _safe_capture_path(name: str):
    """Resolve a capture filename inside the active captures dir, or 400."""
    if not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="invalid capture name")
    state = get_state()
    captures_dir = state.captures_dir()
    target = (captures_dir / name).resolve()
    if not target.is_relative_to(captures_dir.resolve()):
        raise HTTPException(status_code=400, detail="invalid capture path")
    return target


@router.post("/captures", response_model=CaptureSaved)
async def save_capture(image: UploadFile = File(...)) -> CaptureSaved:
    """Persist an uploaded frame under the active project's captures dir."""
    state = get_state()
    captures_dir = state.captures_dir()
    data = await read_upload(image)
    captures_dir.mkdir(parents=True, exist_ok=True)

    raw = (image.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not _SAFE_NAME_RE.match(raw):
        ext = ".jpg" if raw.lower().endswith((".jpg", ".jpeg")) else ".png"
        raw = f"cap_{state.bank.meta.inspection_count}{ext}"

    target = captures_dir / raw
    stem, suffix = target.stem, target.suffix
    i = 1
    while target.exists():
        target = captures_dir / f"{stem}_{i}{suffix}"
        i += 1
    target.write_bytes(data)
    return CaptureSaved(name=target.name)


@router.get("/captures", response_model=CaptureList)
def list_captures() -> CaptureList:
    state = get_state()
    captures_dir = state.captures_dir()
    if not captures_dir.is_dir():
        return CaptureList(captures=[])
    names = sorted(
        p.name for p in captures_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    return CaptureList(captures=names)


@router.get("/captures/{name}")
def get_capture(name: str) -> FileResponse:
    target = _safe_capture_path(name)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="capture not found")
    return FileResponse(target)
