# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""List, thumbnail and prune the images that populate the active bank.

Reads the per-tier row-range indices in ``BankMeta`` so the Develop tab can
show every image an operator taught the bank, and deletes exactly the
matching rows on request.

Where the source image actually sits -- ``_images/<tier>/`` from the retired
append path, or the feature store's own copy -- is ``core.bank_images``'
answer, shared with the project card and the export.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core import bank_images
from ..core.cls_schemas import AnnotationRect, BankState, Tier
from ..core.cls_state import bank_state_of, get_state
from ..core.db_utils import touch_project
from ..core.exceptions import AppError

router = APIRouter(tags=["images"])


class BankImage(BaseModel):
    name: str
    tier: Tier
    label: str = Field("", description="On-disk stem — the key for evaluate/annotate/delete")
    label_display: str = Field(
        "",
        description=(
            "The text the operator typed for this defect class. A stem is "
            "unique but not readable (a Japanese class is stored as "
            "'A-3f2a1b9c'); show this and key on `label`. Falls back to the "
            "stem for banks assembled before it was recorded."
        ),
    )
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


def _index_images(
    tier: Tier, label: str, index: list[dict], display: str = "",
) -> list[BankImage]:
    out: list[BankImage] = []
    for e in index:
        name = str(e.get("name", ""))
        entry_id = str(e.get(bank_images.INDEX_ENTRY_ID_KEY, ""))
        out.append(
            BankImage(
                name=name,
                tier=tier,
                label=label,
                label_display=display or label,
                patches=int(e.get("count", 0)),
                # Percent-encoded, and carrying the store entry when the bank
                # has one. The name here is the operator's ORIGINAL filename
                # (the store keeps it verbatim), so a `#` used to truncate the
                # request at the fragment and a `%` used to start an escape.
                url=bank_images.bank_image_url(tier, name, entry_id=entry_id),
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
    shown = getattr(meta, "label_display", {}) or {}
    out: list[BankImage] = list(_index_images("normal", "", meta.normal_image_index))
    for label, idx in meta.critical_image_index.items():
        out.extend(_index_images("critical", label, idx, shown.get(label, "")))
    for label, idx in meta.negative_image_index.items():
        out.extend(_index_images("negative", label, idx, shown.get(label, "")))
    return BankImagesResponse(images=out)


@router.get("/bank/images/{tier}/{name}")
def get_bank_image(
    tier: Tier,
    name: str,
    entry_id: str = Query("", alias="id"),
) -> FileResponse:
    """Serve one taught source image (for the thumbnail grid).

    ``id`` is the store entry the bank's row index was stamped with. It is
    what makes this exact rather than a guess: without it the route had to
    reconstruct "which tier is this image in" from the ACTIVE label set, while
    the tier in the path came from the index, FROZEN at assemble time. Every
    re-label or unassign made the two disagree and answered 404 — for the
    thumbnail, the viewer and the heatmap alike — until the operator
    assembled again. Banks assembled before the stamp existed fall back to the
    label set, then to a unique name.
    """
    state = get_state()
    state.require_active()
    path = bank_images.resolve_bank_image(state.bank_dir, tier, name, entry_id=entry_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path)


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
