# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Server-side staging for dropped-but-not-yet-taught images.

Dropped images used to live only in the browser (File objects): any reload,
tab close or crash silently discarded them — reported twice on 2026-07-18 as
"my images disappeared". With upload-on-drop, files now
land in ``<bank>/_staging/`` the moment they are dropped, their OK/NG labels
are stored next to them (``staging.json``), and the list is restored on every
reload from any machine. 評価を実行 teaches staged files by name through the
same core as ``/bank/append_batch``; only files that actually reached the
bank are consumed — failures stay staged for a retry.
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..core.cls_schemas import BankState, Tier
from ..core.cls_state import ClsStudioState, bank_state_of, get_state
from ..core.db_utils import touch_project
from ..core.paths import write_bytes_atomic
from ..core.security import check_upload_batch, read_upload
from .bank import _teach_decoded, check_binding

router = APIRouter(tags=["staging"])

STAGING_SUBDIR = "_staging"
META_FILE = "staging.json"
_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,136}")
_TIERS = ("normal", "critical", "negative")


class StagedItem(BaseModel):
    name: str
    tier: Tier | None = None


class StagingList(BaseModel):
    items: list[StagedItem]


def _staging_dir(state: ClsStudioState):
    state.require_active()
    assert state.bank_dir is not None
    return state.bank_dir / STAGING_SUBDIR


def _load_meta(d) -> dict[str, str | None]:
    p = d / META_FILE
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {
            str(k): (v if v in _TIERS else None) for k, v in raw.items()
        } if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_meta(d, meta: dict[str, str | None]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(d / META_FILE, json.dumps(meta, ensure_ascii=False).encode("utf-8"))


def _reconcile(d, meta: dict[str, str | None]) -> dict[str, str | None]:
    """Meta ↔ files: drop entries whose file vanished, adopt unknown files.

    The files are the source of truth — a torn/lost ``staging.json`` must
    never hide staged images (they just come back unlabeled)."""
    if not d.is_dir():
        return {}
    on_disk = {
        f.name for f in d.iterdir()
        if f.is_file() and f.name != META_FILE and not f.name.endswith(".tmp")
    }
    out = {n: t for n, t in meta.items() if n in on_disk}
    for n in sorted(on_disk):
        out.setdefault(n, None)
    return out


def _items(meta: dict[str, str | None]) -> StagingList:
    return StagingList(items=[StagedItem(name=n, tier=t) for n, t in sorted(meta.items())])


@router.get("/bank/staging", response_model=StagingList)
def list_staging() -> StagingList:
    """Staged files of the active bank — restores the UI list after a reload."""
    state = get_state()
    d = _staging_dir(state)
    return _items(_reconcile(d, _load_meta(d)))


@router.post("/bank/staging/upload", response_model=StagingList)
async def upload_staging(
    files: list[UploadFile] = File(...),
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> StagingList:
    """Persist dropped files immediately (upload-on-drop)."""
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    d = _staging_dir(state)
    d.mkdir(parents=True, exist_ok=True)
    meta = _reconcile(d, _load_meta(d))
    # Per-file read_upload only bounds ONE part; without this an N-part drop
    # writes N x 200 MB into <bank>/_staging/.
    check_upload_batch(len(files), 0)
    total_bytes = 0
    for up in files:
        try:
            # read_upload raises the same 413 when ONE part busts
            # MAX_UPLOAD_BYTES, so it has to share the flush path below —
            # otherwise that shape escapes with staging.json stale.
            data = await read_upload(up)
            total_bytes += len(data)
            check_upload_batch(len(files), total_bytes)
        except HTTPException:
            # Flush what already reached disk before rejecting the rest —
            # _reconcile() would adopt the files anyway, but staging.json and
            # the project-card summary must not lag behind them.
            _save_meta(d, meta)
            if state.active_project_id:
                touch_project(state.active_project_id)
            raise
        if not data:
            continue
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", (up.filename or "image.png").rsplit("/", 1)[-1])[:128] or "image.png"
        target = d / safe
        stem, suffix = target.stem, target.suffix or ".png"
        i = 1
        while target.exists():
            target = d / f"{stem}_{i}{suffix}"
            i += 1
        write_bytes_atomic(target, data)
        meta[target.name] = None
    _save_meta(d, meta)
    # The project card counts staged images and may thumbnail them — its
    # summary cache must roll the moment a drop lands (same contract as
    # teach), not up to 30s later.
    if state.active_project_id:
        touch_project(state.active_project_id)
    return _items(meta)


@router.get("/bank/staging/file/{name}")
def staging_file(name: str) -> FileResponse:
    state = get_state()
    d = _staging_dir(state)
    if not _NAME_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail="invalid staged name")
    p = (d / name).resolve()
    if not p.is_relative_to(d.resolve()) or not p.is_file():
        raise HTTPException(status_code=404, detail="staged file not found")
    return FileResponse(p)


class LabelRequest(BaseModel):
    names: list[str]
    tier: Tier | None = None


@router.post("/bank/staging/label", response_model=StagingList)
def label_staging(
    body: LabelRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> StagingList:
    """Set (or clear, tier=null) the tier label of staged files."""
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    d = _staging_dir(state)
    meta = _reconcile(d, _load_meta(d))
    for n in body.names:
        if n in meta:
            meta[n] = body.tier
    _save_meta(d, meta)
    if state.active_project_id:
        touch_project(state.active_project_id)  # card's labeled count
    return _items(meta)


class NamesRequest(BaseModel):
    names: list[str]


@router.post("/bank/staging/delete", response_model=StagingList)
def delete_staging(
    body: NamesRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> StagingList:
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    d = _staging_dir(state)
    meta = _reconcile(d, _load_meta(d))
    for n in body.names:
        if n not in meta:
            continue
        try:
            (d / n).unlink(missing_ok=True)
        except OSError:
            pass  # held handle (thumbnail stream) — reconcile drops it later
        meta.pop(n, None)
    _save_meta(d, meta)
    if state.active_project_id:
        touch_project(state.active_project_id)  # card count + thumbnail
    return _items(meta)


class StagedTeachRequest(BaseModel):
    names: list[str]
    tier: Tier
    label: str = ""


class StagedTeachResult(BaseModel):
    tier: Tier
    label: str
    appended_patches: int
    taught: list[str]
    failed: list[str]
    bank: BankState


@router.post("/bank/staging/teach", response_model=StagedTeachResult)
async def teach_staged(
    body: StagedTeachRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> StagedTeachResult:
    """Teach staged files by name; consume ONLY the ones that reached the bank.

    Failures (undecodable file, capacity stop, aborted group, server error)
    stay staged so the operator just runs again — a failed teach must never
    make images disappear.
    """
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    d = _staging_dir(state)
    meta = _reconcile(d, _load_meta(d))

    decoded: list[tuple[str, bytes, np.ndarray]] = []
    failed: list[str] = []
    for n in body.names:
        if n not in meta or not _NAME_RE.fullmatch(n):
            failed.append(n)
            continue
        try:
            data = (d / n).read_bytes()
        except OSError:
            failed.append(n)
            continue
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            failed.append(n)
            continue
        decoded.append((n, data, arr))

    resolved, added, taught = ("", 0, [])
    if decoded:
        def _run() -> tuple[str, int, list[str]]:
            return _teach_decoded(state, body.tier, decoded, body.label)

        resolved, added, taught = await run_in_threadpool(_run)
        taught_set = set(taught)
        for n in taught:
            try:
                (d / n).unlink(missing_ok=True)
            except OSError:
                pass  # reconcile picks the orphan up; the bank rows are saved
            meta.pop(n, None)
        _save_meta(d, meta)
        failed.extend(n for (n, _dt, _a) in decoded if n not in taught_set)
    if state.active_project_id:
        touch_project(state.active_project_id)
    return StagedTeachResult(
        tier=body.tier, label=resolved, appended_patches=added,
        taught=taught, failed=failed, bank=bank_state_of(state.bank),
    )
