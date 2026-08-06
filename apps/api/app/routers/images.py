# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""List, thumbnail and prune the images that populate the active bank.

Reads the per-tier row-range indices in ``BankMeta`` so the Develop tab can
show every image an operator taught the bank, and deletes exactly the
matching rows on request.

The source image comes from one of two places. ``/bank/append*`` wrote a copy
under ``<bank>/_images/<tier>/``; assembling from the feature store does not,
because the store already keeps the original it extracted the rows from. Both
are served here, ``_images/`` first, so banks from either era show a picture.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core.cls_schemas import AnnotationRect, BankState, Tier
from ..core.cls_state import bank_state_of, get_state
from ..core.cls_store import active_labelset, load_store
from ..core.db_utils import touch_project
from ..core.exceptions import AppError

router = APIRouter(tags=["images"])


class BankImage(BaseModel):
    name: str
    tier: Tier
    label: str = ""
    patches: int = 0
    url: str = ""
    annotations: list[AnnotationRect] = Field(default_factory=list)


class BankImagesResponse(BaseModel):
    images: list[BankImage] = Field(default_factory=list)


class DeleteImagesRequest(BaseModel):
    tier: Tier
    label: str | None = None
    names: list[str]


def _rects_of(e: dict) -> list[AnnotationRect]:
    """Stored annotation rects, dropping anything malformed rather than 500ing."""
    out: list[AnnotationRect] = []
    for r in e.get("annotations", []) or []:
        try:
            out.append(AnnotationRect(**r))
        except (TypeError, ValueError):
            continue
    return out


def _index_images(tier: Tier, label: str, index: list[dict]) -> list[BankImage]:
    out: list[BankImage] = []
    for e in index:
        name = str(e.get("name", ""))
        out.append(
            BankImage(
                name=name,
                tier=tier,
                label=label,
                patches=int(e.get("count", 0)),
                url=f"/bank/images/{tier}/{name}",
                annotations=_rects_of(e),
            )
        )
    return out


@router.get("/bank/images", response_model=BankImagesResponse)
def list_bank_images() -> BankImagesResponse:
    """Every image recorded in the active bank, across all three tiers."""
    state = get_state()
    state.require_active()
    meta = state.bank.meta
    out: list[BankImage] = list(_index_images("normal", "", meta.normal_image_index))
    for label, idx in meta.critical_image_index.items():
        out.extend(_index_images("critical", label, idx))
    for label, idx in meta.negative_image_index.items():
        out.extend(_index_images("negative", label, idx))
    return BankImagesResponse(images=out)


def _store_source(state, tier: str, name: str) -> Path | None:
    """The store's own copy of a taught image, if it has one.

    Matched on name AND tier rather than name alone: the same filename can be
    ingested and then labelled into two different tiers, and serving the wrong
    one would put a defect where the viewer expects a good part.
    """
    if state.bank_dir is None:
        return None
    try:
        store = load_store(state)
        labelset = active_labelset(state, create=False)
    except Exception:  # noqa: BLE001 - no store yet is a miss, not a failure
        return None
    root = state.bank_dir.resolve()
    for entry in store.entries:
        if entry.name != name or not entry.image_ref:
            continue
        assignment = labelset.assignments.get(entry.id)
        if assignment is None or assignment.tier != tier:
            continue
        path = (state.bank_dir / entry.image_ref).resolve()
        # image_ref comes off disk, so it is data: a hand-edited store index
        # must not be able to serve files from outside the bank.
        if path.is_relative_to(root) and path.is_file():
            return path
    return None


@router.get("/bank/images/{tier}/{name}")
def get_bank_image(tier: Tier, name: str) -> FileResponse:
    """Serve one taught source image (for the thumbnail grid)."""
    state = get_state()
    state.require_active()
    path = state.image_path(tier, name)
    if path.is_file():
        return FileResponse(path)
    # A bank assembled from the store has no _images/ copy — only the retired
    # /bank/append path ever wrote one. The rows were extracted from an image
    # the store still holds, so serve that rather than answer 404 and leave
    # the viewer on a black rectangle.
    source = _store_source(state, tier, name)
    if source is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(source)


@router.post("/bank/images/delete", response_model=BankState)
def delete_bank_images(
    body: DeleteImagesRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> BankState:
    """Prune every patch that came from the named images out of the bank."""
    from ..core.cls_eval_cache import _eval_cache_purge
    from .bank import check_binding

    state = get_state()
    state.require_active()
    check_binding(state, binding)
    with state.lock:
        state.bank.remove_images(body.tier, body.label, body.names)
        # Best-effort: drop the saved thumbnails too. OSError included: on
        # Windows the unlink raises PermissionError while a browser is still
        # streaming the thumbnail, and that must not abort the request
        # between the in-memory removal above and save_bank() below (memory
        # and disk would diverge until the next save).
        for name in body.names:
            try:
                state.image_path(body.tier, name).unlink(missing_ok=True)
            except (AppError, HTTPException, OSError):
                pass
        state.mark_dirty()
        state.save_bank()
        # Labelled-tier deletes keep the rest of the eval sweep valid; only
        # the removed images' cached evals must go (normal deletes roll the
        # cache fingerprint instead).
        _eval_cache_purge(state, body.tier, body.label, body.names)
        # Deleting the first-taught image can change the project card's
        # thumbnail — invalidate the summary cache like bank.py's mutations.
        if state.active_project_id:
            touch_project(state.active_project_id)
        return bank_state_of(state.bank)
