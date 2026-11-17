# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Score one image against the active project's bank."""

from __future__ import annotations

import asyncio
import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from ..core.cls_schemas import ScoreResult
from ..core.cls_scoring import decode_image, png_overlay_b64, score_image
from ..core.cls_state import get_state
from ..core.security import read_upload

router = APIRouter(tags=["score"])

# Long edge of the ``original_jpeg_base64`` preview. Browsers cannot render
# TIFF (the common line-camera format), so the heatmap-off view needs a
# server-transcoded copy; 1280 keeps it sharp enough to eyeball defects while
# staying a fraction of the heatmap PNG's size.
_PREVIEW_LONG_EDGE = 1280


async def _client_gone(request: Request) -> bool:
    """Resolve True once the client hangs up; block forever otherwise.

    Meant to be raced against the scoring task and cancelled afterwards. The
    request body is fully consumed by the time this runs, so the only message
    left to arrive on the channel is ``http.disconnect``.
    """
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return True


def _preview_jpeg(image_bgr: np.ndarray) -> bytes:
    h, w = image_bgr.shape[:2]
    scale = _PREVIEW_LONG_EDGE / max(h, w)
    img = image_bgr
    if scale < 1.0:
        img = cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


@router.post("/score", response_model=ScoreResult, summary="Score one image against the active bank")
async def score(
    request: Request,
    image: UploadFile = File(...),
    alpha: float = Query(0.0, description="Critical-bank contribution weight"),
    beta: float = Query(0.0, description="Negative-bank (FP-suppression) weight"),
    k: int = Query(5, ge=1, le=50, description="Normal-bank neighbours per patch"),
    topk: int = Query(10, ge=1, le=256, description="k of the image-level top-k mean score"),
    exemplar_alpha: bool = Query(
        True,
        description=(
            "Restrict the alpha term to the defect-exemplar rows (marked "
            "rows, else each NG's cached top-10) instead of the whole "
            "critical tier; with no exemplars available alpha stays inert"
        ),
    ),
    weighted: bool = Query(False, description="Time-aware severity x freshness weighting"),
    record_hits: bool = Query(True, description="Update freshness on argmin rows"),
    with_attribution: bool = Query(True, description="Compute per-label attribution"),
    persist: bool = Query(
        True,
        description=(
            "Append this result (scores + preview + heatmap) to the project's "
            "inspection log so the Operator tab can restore it after a reload"
        ),
    ),
    hm_lo: float | None = Query(
        None,
        description=(
            "Absolute heatmap anchor: raw score rendered fully transparent "
            "(typically the OK images' median statistic). Both anchors given "
            "and hm_hi > hm_lo switches the overlay to the anomaly-focus "
            "style; otherwise the legacy per-image percentile JET is used."
        ),
    ),
    hm_hi: float | None = Query(
        None,
        description="Absolute heatmap anchor rendered at full colour (typically the raw verdict threshold)",
    ),
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> ScoreResult:
    """Score one image and return the heatmap overlay + per-label attribution."""
    from ..core.cls_exemplar import exemplar_critical_tensors
    from .bank import check_binding

    state = get_state()
    state.require_active()
    check_binding(state, binding)
    if state.bank.normal.size == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "bank is empty — append at least one normal image via "
                "/api/v1/bank/append/normal before scoring."
            ),
        )

    t_total = time.perf_counter()
    content = await read_upload(image)
    t_decode = time.perf_counter()
    image_bgr = decode_image(content)
    decode_ms = (time.perf_counter() - t_decode) * 1000.0

    # Offload the model forward + overlay render to a worker thread so the
    # event loop stays responsive (see the same note in bank.append).
    # (project, bank) identity captured while the scoring lock is held —
    # the persist step below re-checks it so a /bank/select landing between
    # scoring and persisting can't file this inspection under the wrong
    # project's log.
    bound_ids: dict = {}

    def _run() -> tuple:
        with state.gpu_lease(), state.lock:
            bound_ids["v"] = (state.active_project_id, state.active_bank_id)
            critical_override = None
            n_exemplars = 0
            if exemplar_alpha:
                # Built even at alpha=0: attribution then measures proximity
                # to the exemplars too, and the full critical tier is never
                # materialised on the GPU just for the attribution bars.
                critical_override, n_exemplars = exemplar_critical_tensors(state)
            full, topk_score, crit_attr, neg_attr, timing = score_image(
                state, image_bgr, alpha, beta, k,
                weighted=weighted, record_hits=record_hits, with_attribution=with_attribution,
                critical_override=critical_override, topk=topk,
            )
            state.bank.tick()
        t_overlay = time.perf_counter()
        heatmap_b64 = png_overlay_b64(full, image_bgr, hm_lo, hm_hi)
        timing["overlay_ms"] = (time.perf_counter() - t_overlay) * 1000.0
        original_jpeg = _preview_jpeg(image_bgr)
        return full, topk_score, n_exemplars, crit_attr, neg_attr, timing, heatmap_b64, original_jpeg

    # Watch for the client hanging up WHILE the scoring runs. The operator's
    # cancel aborts the fetch; run_in_threadpool cannot be interrupted, so the
    # work finishes either way, but what we can still honour is not writing the
    # result into the inspection log — otherwise a cancelled image reappears on
    # the next mount, and a 「クリア」 issued during a score is undone by that
    # score's own append a moment later.
    #
    # Request.is_disconnected() is NOT usable for this: it awaits receive()
    # inside an already-cancelled scope, so it only ever reports a disconnect
    # that happens to be buffered, and in practice stays False for the whole
    # request. Verified against the live server — with is_disconnected() the
    # abandoned score was still persisted every time (2026-07-31). Awaiting
    # receive() properly is what actually observes the hang-up.
    work = asyncio.ensure_future(run_in_threadpool(_run))
    watch = asyncio.ensure_future(_client_gone(request))
    try:
        await asyncio.wait({work, watch}, return_when=asyncio.FIRST_COMPLETED)
        client_gone = watch.done() and not watch.cancelled() and watch.result() is True
        full, topk_score, n_exemplars, crit_attr, neg_attr, timing, heatmap_b64, original_jpeg = await work
    finally:
        watch.cancel()
    timing["decode_ms"] = decode_ms
    timing["total_server_ms"] = (time.perf_counter() - t_total) * 1000.0

    inspection_id = ""
    if persist and client_gone:
        import logging

        logging.getLogger(__name__).info("inspection persist skipped: client disconnected")
        persist = False
    if persist and bound_ids.get("v") != (state.active_project_id, state.active_bank_id):
        import logging

        logging.getLogger(__name__).warning(
            "inspection persist skipped: active bank changed after scoring (%s -> %s/%s)",
            bound_ids.get("v"), state.active_project_id, state.active_bank_id,
        )
        persist = False
    if persist:
        # Best-effort: the Operator tab restores these after a page reload
        # (results otherwise die with the browser's state). Never fail the
        # scoring response over a persistence hiccup.
        try:
            from .inspections import append_inspection

            inspection_id = await run_in_threadpool(
                append_inspection,
                state,
                name=(image.filename or "image").replace("\\", "/").rsplit("/", 1)[-1],
                topk_score=float(topk_score),
                max_score=float(full.max()),
                p99_score=float(np.percentile(full, 99)),
                n_exemplar_rows=int(n_exemplars),
                alpha=float(alpha),
                server_ms=float(timing["total_server_ms"]),
                orig_jpeg=original_jpeg,
                heat_png=base64.b64decode(heatmap_b64),
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("inspection persist failed: %s", exc)

    return ScoreResult(
        max_score=float(full.max()),
        mean_score=float(full.mean()),
        p99_score=float(np.percentile(full, 99)),
        topk_score=float(topk_score),
        n_exemplar_rows=int(n_exemplars),
        heatmap_png_base64=heatmap_b64,
        original_jpeg_base64=base64.b64encode(original_jpeg).decode("ascii") if original_jpeg else "",
        inspection_id=inspection_id,
        critical_attribution=crit_attr,
        negative_attribution=neg_attr,
        timings={k_: float(v) for k_, v in timing.items()},
    )
