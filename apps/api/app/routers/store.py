# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Feature store, label sets, and the assembly that turns them into a bank.

These routes are the new shape of teaching, split in three:

    POST /store/ingest        run the backbone once per image        (expensive)
    POST /labelsets/assign    say what each image is                 (free)
    POST /bank/assemble       rebuild the scoreable bank             (numpy)

The old ``/bank/append*`` routes still work and still do all three at once;
nothing about an existing bank changes until ``/store/migrate`` is called on
it. That is deliberate — the migration is additive and verified, so a project
can move over when its operator is ready rather than on upgrade.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..core.cls_schemas import (
    AnnotationRect,
    AssembleResult,
    AssemblyStatus,
    AssignResult,
    GroupPreview,
    IngestResult,
    LabelSetInfo,
    LabelSetList,
    MarkResult,
    MigrationResult,
    StoreImageInfo,
    StoreListResponse,
    Tier,
)
from ..core.cls_state import bank_state_of, get_state
from ..core.cls_store import (
    RENDER_SUBDIR,
    active_labelset,
    all_labelsets,
    assemble_active_bank,
    assembly_status,
    create_labelset,
    delete_labelset,
    drop_renditions,
    ingest_decoded,
    labelsets_dir,
    load_labelset,
    load_store,
    migrate_active_bank,
    rendered_image,
)
from ..core.db_utils import touch_project
from ..core.exceptions import ValidationError
from ..core.security import check_upload_batch, read_upload
from .bank import _capacity_ceiling, _max_patches_per_image, check_binding

router = APIRouter(tags=["store"])


# ---- listing ---------------------------------------------------------------


def _listing(state) -> StoreListResponse:
    store = load_store(state)
    ls = active_labelset(state)
    images = []
    for e in store.entries:
        a = ls.assignments.get(e.id)
        images.append(
            StoreImageInfo(
                id=e.id,
                name=e.name,
                rows=int(e.rows),
                grid_rows=int(e.grid_rows),
                width=int(e.width),
                height=int(e.height),
                has_image=bool(e.image_ref),
                tier=a.tier if a else "",
                label=a.resolved_label() if a else "",
                severity=a.clamped_severity() if a else 0,
                marks=len(a.marks) if a else 0,
                rects=[AnnotationRect(**r) for r in (a.rects if a else [])],
                group=e.group,
            )
        )
    return StoreListResponse(
        images=images,
        total_rows=store.total_rows(),
        dim=int(store.meta.dim),
        model=store.meta.model,
        labelset_id=ls.id,
    )


@router.get("/store", response_model=StoreListResponse)
def get_store() -> StoreListResponse:
    """Every ingested image and, from the active label set, what it is."""
    state = get_state()
    state.require_active()
    return _listing(state)


@router.get("/store/image/{entry_id}")
def get_store_image(entry_id: str, size: str = "full") -> FileResponse:
    """One store entry's image, downscaled for display unless ``size=full``.

    ``thumb`` for list rows, ``preview`` for the centre pane. The store keeps
    the original because a re-ingest needs it, but serving those to a picker
    means hundreds of megabytes to fill a list of 40px rows.
    """
    state = get_state()
    state.require_active()
    assert state.bank_dir is not None
    store = load_store(state)
    entry = store.by_id(entry_id)
    if entry is None or not entry.image_ref:
        raise HTTPException(status_code=404, detail="no source image for this entry")
    root = state.bank_dir.resolve()
    path = (state.bank_dir / entry.image_ref).resolve()
    # The ref comes off disk, so it is data rather than a constant: a store
    # index edited by hand must not be able to serve files outside the bank.
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="source image not found")
    return FileResponse(
        rendered_image(path, store.directory / RENDER_SUBDIR, entry.id, size)
    )


# ---- ingest ----------------------------------------------------------------


@router.post("/store/ingest", response_model=IngestResult)
async def ingest_images(
    images: list[UploadFile] = File(...),
    tier: str = Form(""),
    label: str = Form(""),
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> IngestResult:
    """Extract features for one or more images into the store.

    No tier is required: an ingested image starts unassigned and shows up in
    the labelling grid waiting for a decision. ``tier`` / ``label`` are an
    optional convenience for the common drop-a-folder-of-OK-images case, and
    only pre-fill the active label set — the features are identical either
    way, which is the entire point.
    """
    state = get_state()
    state.require_active()
    check_binding(state, binding)

    check_upload_batch(len(images), 0)
    decoded: list[tuple[str, bytes, np.ndarray]] = []
    failed: list[str] = []
    total_bytes = 0
    for up in images:
        data = await read_upload(up)
        total_bytes += len(data)
        check_upload_batch(len(images), total_bytes)
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            failed.append(up.filename or "?")
            continue
        decoded.append((up.filename or "image.png", data, arr))
    if not decoded:
        raise HTTPException(status_code=422, detail="no decodable images in batch")

    def _run() -> tuple[int, int]:
        store = load_store(state)
        entries = ingest_decoded(
            state, store, decoded, max_patches=_max_patches_per_image()
        )
        if tier in ("normal", "critical", "negative"):
            ls = active_labelset(state)
            for e in entries:
                ls.assign(e.id, tier, label=label)
            ls.save(labelsets_dir(state))
        return len(entries), sum(int(e.rows) for e in entries)

    ingested, rows = await run_in_threadpool(_run)
    if state.active_project_id:
        touch_project(state.active_project_id)
    return IngestResult(
        ingested=ingested, rows=rows, failed=failed, status=AssemblyStatus(**assembly_status(state))
    )


class DeleteRequest(BaseModel):
    ids: list[str]


@router.post("/store/delete", response_model=StoreListResponse)
def delete_from_store(
    body: DeleteRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> StoreListResponse:
    """Remove images from the store and from every label set that names them.

    Dropping the assignments too is what keeps a label set from accumulating
    references to features that no longer exist — assembly would skip them
    silently and the counts in the UI would drift from the bank.
    """
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    store = load_store(state)
    with state.lock:
        removed = store.remove(body.ids)
        if removed:
            store.save_index()
            for ls in all_labelsets(state):
                if any(ls.unassign(i) for i in list(body.ids)):
                    ls.save(labelsets_dir(state))
            for entry_id in body.ids:
                for p in store.images_dir().glob(f"{entry_id}.*"):
                    p.unlink(missing_ok=True)
            drop_renditions(store, list(body.ids))
    return _listing(state)


@router.post("/store/migrate", response_model=MigrationResult)
async def migrate_store(
    verify: bool = True,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> MigrationResult:
    """Carve this bank's existing rows into a store + "standard" label set.

    Nothing is re-extracted and nothing the bank already holds is modified.
    Unless ``verify=false``, the new layout has to re-assemble into the same
    arrays before it is kept; if it does not, the store is discarded and the
    differences are returned.
    """
    state = get_state()
    state.require_active()
    check_binding(state, binding)

    def _run() -> dict:
        return migrate_active_bank(state, verify=verify)

    try:
        result = await run_in_threadpool(_run)
    except ValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result["ok"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "migration did not reproduce the bank — nothing was kept",
                "problems": result["problems"][:20],
            },
        )
    return MigrationResult(
        images=result["images"],
        rows=result["rows"],
        labelset_id=result.get("labelset_id", ""),
        status=AssemblyStatus(**assembly_status(state)),
    )


# ---- label sets ------------------------------------------------------------


def _labelset_list(state) -> LabelSetList:
    active = active_labelset(state)
    return LabelSetList(
        labelsets=[
            LabelSetInfo(
                id=ls.id, name=ls.name, description=ls.description,
                counts=ls.counts(), updated_at=ls.updated_at,
            )
            for ls in all_labelsets(state)
        ],
        active_id=active.id,
    )


@router.get("/labelsets", response_model=LabelSetList)
def get_labelsets() -> LabelSetList:
    state = get_state()
    state.require_active()
    return _labelset_list(state)


class CreateLabelSetRequest(BaseModel):
    name: str
    copy_active: bool = True


@router.post("/labelsets/create", response_model=LabelSetList)
def post_create_labelset(body: CreateLabelSetRequest) -> LabelSetList:
    """New label set over the same store, optionally forked from the active one."""
    state = get_state()
    state.require_active()
    from clscore.labelset import write_active_id

    src = active_labelset(state) if body.copy_active else None
    ls = create_labelset(state, body.name, copy_from=src)
    write_active_id(labelsets_dir(state), ls.id)
    return _labelset_list(state)


class LabelSetIdRequest(BaseModel):
    id: str


@router.post("/labelsets/select", response_model=LabelSetList)
def post_select_labelset(body: LabelSetIdRequest) -> LabelSetList:
    state = get_state()
    state.require_active()
    from clscore.labelset import write_active_id

    try:
        ls = load_labelset(state, body.id)
    except ValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    write_active_id(labelsets_dir(state), ls.id)
    return _labelset_list(state)


@router.post("/labelsets/delete", response_model=LabelSetList)
def post_delete_labelset(body: LabelSetIdRequest) -> LabelSetList:
    state = get_state()
    state.require_active()
    try:
        delete_labelset(state, body.id)
    except ValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _labelset_list(state)


class AssignRequest(BaseModel):
    ids: list[str]
    tier: Tier
    label: str = ""
    severity: int = 2


@router.post("/labelsets/assign", response_model=AssignResult)
def post_assign(
    body: AssignRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> AssignResult:
    """Put images in a tier. Costs nothing — no feature is touched."""
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    store = load_store(state)
    known = {e.id for e in store.entries}
    unknown = [i for i in body.ids if i not in known]
    if unknown:
        raise HTTPException(
            status_code=404, detail=f"not in this store: {', '.join(unknown[:5])}"
        )
    ls = active_labelset(state)
    for i in body.ids:
        ls.assign(i, body.tier, label=body.label, severity=body.severity)
    ls.save(labelsets_dir(state))
    return AssignResult(
        changed=len(body.ids), status=AssemblyStatus(**assembly_status(state))
    )


class UnassignRequest(BaseModel):
    ids: list[str]


@router.post("/labelsets/unassign", response_model=AssignResult)
def post_unassign(
    body: UnassignRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> AssignResult:
    """Return images to the unlabelled pool."""
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    ls = active_labelset(state)
    changed = sum(1 for i in body.ids if ls.unassign(i))
    ls.save(labelsets_dir(state))
    return AssignResult(changed=changed, status=AssemblyStatus(**assembly_status(state)))


class MarkRequest(BaseModel):
    id: str
    rects: list[dict] = []


@router.post("/labelsets/mark", response_model=MarkResult)
def post_mark(
    body: MarkRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> MarkResult:
    """Mark one image's defect regions, as patches of its grid.

    The rectangles are resolved against the store's geometry here and stored
    as grid indices, never as bank rows: rows move when the image is capped or
    the bank is re-assembled at a different capacity, and the grid does not.
    """
    from clscore.sw import rows_for_rects

    state = get_state()
    state.require_active()
    check_binding(state, binding)
    store = load_store(state)
    entry = store.by_id(body.id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"not in this store: {body.id}")
    h, w = int(entry.height), int(entry.width)
    if not (h and w):
        # Migrated entries whose source image could not be read carry no size;
        # resolving rectangles without it would land the marks on the wrong
        # patches, which is worse than refusing.
        raise HTTPException(
            status_code=409,
            detail="this image's pixel size is unknown — re-ingest it to mark regions",
        )
    geo = store.meta.geometry()
    rows = rows_for_rects(
        h, w, [(r["x"], r["y"], r["w"], r["h"]) for r in body.rects], **geo
    )
    ls = active_labelset(state)
    try:
        ls.mark(body.id, rows, rects=body.rects)
    except KeyError as exc:
        raise HTTPException(
            status_code=409, detail="assign this image to a tier before marking it"
        ) from exc
    ls.save(labelsets_dir(state))
    return MarkResult(
        id=body.id, marks=len(rows), status=AssemblyStatus(**assembly_status(state))
    )


# ---- assembly --------------------------------------------------------------


class SetGroupRequest(BaseModel):
    ids: list[str]
    group: str = ""


@router.post("/store/group", response_model=StoreListResponse)
def post_set_group(
    body: SetGroupRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> StoreListResponse:
    """Assign images to a manual validation group (an empty string clears it)."""
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    store = load_store(state)
    known = {e.id for e in store.entries}
    unknown = [i for i in body.ids if i not in known]
    if unknown:
        raise HTTPException(
            status_code=404, detail=f"not in this store: {', '.join(unknown[:5])}"
        )
    target = set(body.ids)
    for e in store.entries:
        if e.id in target:
            e.group = body.group.strip()
    store.save_index()
    return _listing(state)


@router.get("/store/groups", response_model=GroupPreview)
def get_group_preview(
    mode: str = "datetime", sep: str = "_", fields: int = 1
) -> GroupPreview:
    """What ``mode`` would split this store into, before anything is validated.

    Filenames vary per site, so the rule has to be checked against the real
    names before an AUROC is computed from it -- a convention guessed wrong
    silently produces one group per image, which is just leave-one-out again
    wearing a different name.
    """
    from clscore.grouping import GROUP_MODES, derive_groups, group_summary

    state = get_state()
    state.require_active()
    if mode not in GROUP_MODES:
        raise HTTPException(status_code=422, detail=f"unknown group mode: {mode}")
    store = load_store(state)
    manual = {e.name: e.group for e in store.entries if e.group}
    groups = derive_groups(
        [e.name for e in store.entries], mode, sep=sep, fields=fields, manual=manual
    )
    summary = group_summary(groups)
    ungrouped = len(summary.get("", []))
    return GroupPreview(
        mode=mode,
        groups=summary,
        grouped=len(groups) - ungrouped,
        ungrouped=ungrouped,
    )


@router.get("/bank/assembly", response_model=AssemblyStatus)
def get_assembly_status() -> AssemblyStatus:
    """Whether the loaded bank still matches the store + active label set."""
    state = get_state()
    state.require_active()
    return AssemblyStatus(**assembly_status(state))


@router.post("/bank/assemble", response_model=AssembleResult)
async def post_assemble(
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> AssembleResult:
    """Rebuild the bank from the store and the active label set."""
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    ceiling = _capacity_ceiling(state)

    def _run() -> None:
        assemble_active_bank(state, normal_ceiling=ceiling)

    try:
        await run_in_threadpool(_run)
    except ValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if state.active_project_id:
        touch_project(state.active_project_id)
    return AssembleResult(
        bank=bank_state_of(state.bank), status=AssemblyStatus(**assembly_status(state))
    )
