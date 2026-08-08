# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Scoring path for the active cls-studio bank.

Ported from cls-studio v0.1 (``cls-studio/api/app.py`` ``_score`` /
``_png_overlay_b64`` helpers), adapted to the :class:`ClsStudioState`
lazy-model / cached-tensor interface. Behaviour is unchanged.
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from clscore.io import overlay, overlay_diverging
from clscore.scoring import (
    attribution_per_label,
    compose_score_grid,
    extract_distance_components,
    image_topk_mean,
    per_label_winners,
)

from .cls_state import ClsStudioState
from .exceptions import ImageDecodeError, PredictError


def decode_image(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("could not decode image bytes")
    return img


def png_overlay_b64(
    full_hm: np.ndarray,
    image_bgr: np.ndarray,
    hm_lo: float | None = None,
    hm_hi: float | None = None,
) -> str:
    """Heatmap → PNG overlay.

    With absolute anchors (``hm_lo`` < ``hm_hi``, typically the bank's OK
    median and raw verdict threshold from the separation check) the
    diverging score map is used: blue = OK level, neutral = at the
    threshold, vermilion = NG level, on the same absolute scale for every
    image. Without anchors it falls back to the legacy per-image
    percentile JET — note that one always paints something red, even on a
    perfect OK image, because the scale is relative to the image itself.
    """
    if hm_lo is not None and hm_hi is not None and hm_hi > hm_lo:
        ov = overlay_diverging(image_bgr, full_hm, hm_lo, hm_hi)
    else:
        vmin = float(np.percentile(full_hm, 5))
        vmax = float(np.percentile(full_hm, 99)) or 1.0
        ov = overlay(image_bgr, full_hm, vmin, vmax)
    ok, buf = cv2.imencode(".png", ov)
    if not ok:
        raise PredictError("overlay PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _record_hits(state: ClsStudioState, windows: list[dict], *, alpha: float, beta: float) -> None:
    """Tick freshness on the argmin row of the strongest window per label."""
    if alpha != 0.0:
        for label, w in per_label_winners(windows, "critical_min", "critical_argmin").items():
            if w.bank_row is not None:
                state.bank.hit("critical", label, np.array([w.bank_row], dtype=np.int64))
    if beta != 0.0:
        for label, w in per_label_winners(windows, "negative_min", "negative_argmin").items():
            if w.bank_row is not None:
                state.bank.hit("negative", label, np.array([w.bank_row], dtype=np.int64))


def score_image(
    state: ClsStudioState,
    image_bgr: np.ndarray,
    alpha: float,
    beta: float,
    k: int,
    *,
    weighted: bool = False,
    record_hits: bool = True,
    with_attribution: bool = True,
    critical_override: dict | None = None,
    topk: int = 10,
) -> tuple[np.ndarray, float, dict[str, float], dict[str, float], dict]:
    """Score one image; return ``(heatmap, topk_score, crit_attr, neg_attr, timing)``.

    ``critical_override`` replaces the full critical tier for the alpha term
    — the Operator path passes the defect-exemplar rows here so alpha only
    fires near known defects (a full-tier alpha fires on every patch, since
    most of an NG image is normal surface). Its row indices don't map back
    to the bank arrays, so critical hit-recording is skipped while an
    override is active. ``topk_score`` is the separation-check statistic
    (``image_topk_mean``) computed on the same composite scores as the
    heatmap, so Develop-tab thresholds transfer to inference directly.
    """
    model, device, _dtype = state.ensure_model()
    timing: dict = {}
    t0 = time.perf_counter()
    # Tiers are moved to the GPU lazily: with an exemplar override the full
    # critical tier (as large as the normal bank when NG images are taught
    # whole) never gets materialised on the device at all.
    # IVF first: with resident storage the index IS the normal bank and the
    # full fp16 tensor is never materialised (VRAM saving).
    ivf, ivf_nprobe = state.get_normal_ivf()
    if ivf is not None and ivf.has_storage:
        n_t, n_rows = None, ivf.n_rows
    else:
        n_t = state.get_normal_tensor()
        n_rows = int(n_t.shape[0])
    c_d = critical_override if critical_override is not None else state.get_tier_tensors("critical")
    g_d = state.get_tier_tensors("negative")
    timing["bank_tensors_ms"] = (time.perf_counter() - t0) * 1000.0
    # Surfaced so operators (and remote verification) can see which
    # compression transforms actually applied to this inference.
    from .runtime_compression import read_compression_settings

    timing["int8_active"] = 1.0 if read_compression_settings()["int8"] else 0.0
    timing["ivf_active"] = 1.0 if ivf is not None else 0.0

    need_critical = alpha != 0.0 or with_attribution or record_hits
    need_negative = beta != 0.0 or with_attribution or record_hits

    t1 = time.perf_counter()
    windows, full_shape, grid_shape = extract_distance_components(
        model, image_bgr, n_t, num_neighbors=k, device=device,
        critical=c_d if need_critical else None,
        negative=g_d if need_negative else None,
        timing=timing,
        critical_meta=state.bank.critical_meta if weighted and need_critical and critical_override is None else None,
        negative_meta=state.bank.negative_meta if weighted and need_negative else None,
        inspection_count=state.bank.meta.inspection_count,
        weighted=weighted,
        track_argmin=record_hits,
        # Device-sized so teaching/heatmaps run fast on big GPUs and don't OOM
        # on small ones as the normal bank grows.
        max_batch=state.max_batch(),
        cdist_chunk=state.cdist_chunk_for(n_rows, ivf_active=ivf is not None),
        ivf=ivf,
        ivf_nprobe=ivf_nprobe or 8,
    )
    timing["extract_total_ms"] = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    full, _ = compose_score_grid(windows, grid_shape, full_shape, alpha=alpha, beta=beta)
    topk_score = image_topk_mean(windows, alpha=alpha, beta=beta, k=topk)
    timing["compose_ms"] = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    if with_attribution:
        crit_attr, neg_attr = attribution_per_label(
            windows,
            alpha=alpha if alpha != 0.0 else 1.0,
            beta=beta if beta != 0.0 else 1.0,
        )
    else:
        crit_attr, neg_attr = {}, {}
    timing["attribution_ms"] = (time.perf_counter() - t3) * 1000.0

    if record_hits:
        _record_hits(
            state, windows,
            # Override rows are subset-relative — hitting them would tick
            # freshness on the wrong bank rows, so critical hits are off.
            alpha=0.0 if critical_override is not None else alpha,
            beta=beta,
        )

    return full, topk_score, crit_attr, neg_attr, timing
