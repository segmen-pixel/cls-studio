# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Persisted operator-inspection results for the active project.

Every ``/score`` call (unless ``persist=false``) appends an entry here:
metadata rows in ``<project>/inspections/log.json`` plus the downscaled
original JPEG and heatmap PNG as sibling files. The Operator tab re-lists
them on mount, so results that finished server-side survive a browser
reload instead of evaporating with the page's state. The log is pruned to
the newest ``CLS_INSPECTION_LOG_CAP`` (default 200) entries; scoring
correctness never depends on this store — persistence is best-effort.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core.cls_state import ClsStudioState, get_state
from ..core.paths import write_bytes_atomic

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inspections"])

LOG_FILE = "log.json"
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
_DEFAULT_CAP = 200

# One writer at a time for the read-modify-write on log.json. Scoring is
# already serialised per-request on the GPU side, but nothing stops two
# concurrent /score calls from interleaving their log appends.
_LOG_LOCK = threading.Lock()


def _cap() -> int:
    raw = os.environ.get("CLS_INSPECTION_LOG_CAP")
    try:
        return max(1, int(raw)) if raw is not None else _DEFAULT_CAP
    except ValueError:
        return _DEFAULT_CAP


class InspectionEntry(BaseModel):
    id: str
    name: str
    ts: str
    topk_score: float
    max_score: float
    p99_score: float
    n_exemplar_rows: int = 0
    alpha: float = 0.0
    server_ms: float = 0.0
    orig: str = Field("", description="Filename of the stored original preview (JPEG)")
    heat: str = Field("", description="Filename of the stored heatmap (PNG)")


class InspectionList(BaseModel):
    entries: list[InspectionEntry] = Field(default_factory=list)


def _read_log(d: Path) -> list[dict]:
    path = d / LOG_FILE
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw.get("entries", []) if isinstance(raw, dict) else []
    except (OSError, ValueError):
        return []  # torn/corrupt log = start fresh; the files just linger


def _write_log(d: Path, entries: list[dict]) -> None:
    write_bytes_atomic(d / LOG_FILE, json.dumps({"entries": entries}).encode("utf-8"))


def append_inspection(
    state: ClsStudioState,
    *,
    name: str,
    topk_score: float,
    max_score: float,
    p99_score: float,
    n_exemplar_rows: int,
    alpha: float,
    server_ms: float,
    orig_jpeg: bytes,
    heat_png: bytes,
) -> str:
    """Persist one scored inspection; called by /score. Returns the entry id
    so the client can target it with the per-entry delete. Best-effort."""
    d = state.inspections_dir()
    d.mkdir(parents=True, exist_ok=True)
    eid = uuid.uuid4().hex[:12]
    orig_name = f"{eid}_orig.jpg"
    heat_name = f"{eid}_heat.png"
    write_bytes_atomic(d / orig_name, orig_jpeg)
    write_bytes_atomic(d / heat_name, heat_png)
    entry = InspectionEntry(
        id=eid,
        name=(name or "image")[:160],
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        topk_score=float(topk_score),
        max_score=float(max_score),
        p99_score=float(p99_score),
        n_exemplar_rows=int(n_exemplar_rows),
        alpha=float(alpha),
        server_ms=float(server_ms),
        orig=orig_name,
        heat=heat_name,
    ).model_dump()
    with _LOG_LOCK:
        entries = _read_log(d)
        entries.append(entry)
        # Prune beyond the cap, oldest first, together with their files.
        drop, entries = entries[:-_cap()], entries[-_cap():]
        for e in drop:
            for f in (e.get("orig"), e.get("heat")):
                if f and _SAFE_NAME_RE.match(str(f)):
                    (d / str(f)).unlink(missing_ok=True)
        _write_log(d, entries)
    return eid


@router.get("/inspections", response_model=InspectionList)
def list_inspections() -> InspectionList:
    """Persisted inspection results for the active project, oldest first."""
    state = get_state()
    state.require_active()
    d = state.inspections_dir()
    return InspectionList(entries=[InspectionEntry(**e) for e in _read_log(d)])


@router.get("/inspections/file/{name}")
def get_inspection_file(name: str) -> FileResponse:
    """Serve a stored preview / heatmap file."""
    state = get_state()
    state.require_active()
    if not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="invalid inspection file name")
    d = state.inspections_dir()
    target = (d / name).resolve()
    if not target.is_relative_to(d.resolve()) or not target.is_file():
        raise HTTPException(status_code=404, detail="inspection file not found")
    return FileResponse(target)


@router.delete("/inspections/{entry_id}", response_model=InspectionList)
def delete_inspection(entry_id: str) -> InspectionList:
    """Drop one entry (the Operator list's Delete key). Idempotent."""
    state = get_state()
    state.require_active()
    if not re.fullmatch(r"[0-9a-f]{12}", entry_id):
        raise HTTPException(status_code=400, detail="invalid inspection id")
    d = state.inspections_dir()
    with _LOG_LOCK:
        entries = _read_log(d)
        keep = [e for e in entries if e.get("id") != entry_id]
        if len(keep) != len(entries):
            for e in entries:
                if e.get("id") != entry_id:
                    continue
                for f in (e.get("orig"), e.get("heat")):
                    if f and _SAFE_NAME_RE.match(str(f)):
                        (d / str(f)).unlink(missing_ok=True)
            _write_log(d, keep)
    return InspectionList(entries=[InspectionEntry(**e) for e in keep])


@router.delete("/inspections", response_model=InspectionList)
def clear_inspections() -> InspectionList:
    """Drop the whole inspection log (the Operator tab's clear button)."""
    state = get_state()
    state.require_active()
    d = state.inspections_dir()
    with _LOG_LOCK:
        for e in _read_log(d):
            for f in (e.get("orig"), e.get("heat")):
                if f and _SAFE_NAME_RE.match(str(f)):
                    (d / str(f)).unlink(missing_ok=True)
        if (d / LOG_FILE).exists():
            _write_log(d, [])
    return InspectionList(entries=[])
