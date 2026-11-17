# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Per-image score composition: distance components + alpha/beta combination.

The scoring pipeline is split into two phases so that alpha/beta can be tuned
without recomputing the (expensive) DINOv2 features and bank distances:

    1. extract_distance_components(...)  -> per-window distance cache
    2. compose_score_grid(...)            -> alpha/beta sweep is cheap from here

Critical and negative are passed in as ``dict[label, tensor]`` — each label
is a separate sub-bank ("scratch", "stain", ...) so we can attribute a hot
window to a specific defect class. The scalar score formula is unchanged
(min over labels keeps it backward compatible):

    s = bank_topk_mean
        + alpha / (1 + min_label critical_min[label])      (if non-empty)
        - beta  / (1 + min_label negative_min[label])      (if non-empty)

Multiple windows are merged via per-pixel MAX over the patch grid, then
bilinearly resampled back to the original image resolution.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from .feature_extractor import extract_windows_tokens_batched
from .incident import IncidentMetaArray, multiplier_for
from .sw import DINO_PATCH, WINDOW_SIZE, WINDOW_STRIDE, pad_to_min, sw_offsets

if TYPE_CHECKING:  # avoid a hard import: compress needs nothing from here
    from .compress import IvfIndex

logger = logging.getLogger(__name__)

__all__ = [
    "BOOST_FLOOR",
    "EPS",
    "LabelWinner",
    "attribution_per_label",
    "compose_score_grid",
    "extract_distance_components",
    "image_auroc",
    "image_topk_mean",
    "per_label_winners",
    "safe_cdist_chunk",
    "score_image",
    "score_stored_features",
]

EPS: float = 1e-6
# Denominator floor for the exemplar boost/suppression terms. The old floor
# was EPS, which made the term singular on (near-)exact matches: inspecting a
# taught NG image gave alpha/EPS ~ 8.4e8 at alpha=840 (observed on a
# production project, 2026-08-03), and the beta term could equally explode
# NEGATIVE and mask real defects. A floor of 1.0 bounds the contribution at
# alpha (resp. beta) while staying within ~20% of the old curve for d >= 5.
BOOST_FLOOR: float = 1.0


def _bank_topk_mean(
    chunk: torch.Tensor,
    bank: torch.Tensor | None,
    kk: int,
    *,
    excl: list[tuple[int, int]] | None = None,
    ivf: IvfIndex | None = None,
    ivf_nprobe: int = 8,
) -> torch.Tensor:
    """Top-``kk`` mean distance of ``chunk`` vs ``bank``, one value per query.

    ``excl`` masks bank row ranges ``[start, start+count)`` out of the
    search: its own rows for leave-own-image-out, or every range of a lot
    for leave-own-group-out. With ``ivf`` set, columns outside each
    query's ``ivf_nprobe`` nearest clusters are masked too — the exact
    candidate-set semantics the compression sweep validated. When a query's
    probed clusters end up entirely masked (tiny cluster fully excluded by
    leave-own-image-out), that query silently falls back to a full scan so
    a finite score always comes back; the sweep never hit this corner, so
    the fallback only fires where the sweep would have produced NaN.

    With ``ivf=None`` the ops are identical to the pre-compression code
    path (cdist -> topk -> mean), keeping existing results bit-exact.
    An index with resident storage (``ivf.has_storage``) answers from its
    cluster-sorted slices directly — ``bank`` may then be ``None`` and no
    full bank tensor is touched (the gather fast path).
    """
    if ivf is not None and ivf.has_storage:
        return ivf.search_topk_mean(chunk, kk, excl_ranges=excl, nprobe=ivf_nprobe)
    if bank is None:
        raise ValueError("bank tensor required when no IVF storage is attached")
    d = torch.cdist(chunk, bank, compute_mode="use_mm_for_euclid_dist")
    for _s0, _cnt in excl or ():
        d[:, _s0 : _s0 + _cnt] = float("inf")
    if ivf is None:
        topk, _ = torch.topk(d, kk, dim=1, largest=False)
        return topk.mean(dim=1)
    d.masked_fill_(~ivf.allowed_mask(chunk, ivf_nprobe), float("inf"))
    topk = torch.topk(d, kk, dim=1, largest=False).values
    finite = torch.where(torch.isinf(topk), torch.nan, topk)
    mean = torch.nanmean(finite, dim=1)
    bad = ~torch.isfinite(mean)
    if bool(bad.any()):
        d2 = torch.cdist(chunk[bad], bank, compute_mode="use_mm_for_euclid_dist")
        for _s0, _cnt in excl or ():
            d2[:, _s0 : _s0 + _cnt] = float("inf")
        topk2 = torch.topk(d2, kk, dim=1, largest=False).values
        mean[bad] = topk2.mean(dim=1).to(mean.dtype)
    return mean


def safe_cdist_chunk(
    n_bank: int,
    free_bytes: int,
    *,
    elem_bytes: int = 2,
    overhead: float = 3.0,
    safety: float = 0.6,
    floor: int = 32,
    ceil: int = 4096,
) -> int:
    """Largest query-row chunk for ``cdist(chunk, bank[n_bank])`` within budget.

    The ``[chunk x n_bank]`` distance matrix dominates VRAM; ``overhead``
    covers the matmul intermediate + the topk copy, ``safety`` leaves
    headroom for fragmentation. Because the footprint is exactly predictable
    (unlike a model forward), this is computed analytically rather than
    probed — no OOM round trip. Grows the chunk on big GPUs / small banks and
    shrinks it as the bank grows, which is what keeps a large OK bank from
    OOMing on modest hardware. Clamped to ``[floor, ceil]``.
    """
    if n_bank <= 0:
        return ceil
    per_row = float(n_bank) * elem_bytes * overhead
    if per_row <= 0:
        return ceil
    chunk = int((free_bytes * safety) // per_row)
    return max(floor, min(ceil, chunk))


def _cuda_sync(device: str) -> None:
    """Block until pending CUDA work finishes — needed for honest wall-clock
    timing because PyTorch dispatches kernels asynchronously."""
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _per_label_reduce(
    flat_chunk: torch.Tensor,
    banks: Mapping[str, torch.Tensor] | None,
    metas: Mapping[str, IncidentMetaArray] | None = None,
    inspection_count: int = 0,
    weighted: bool = False,
    return_argmin: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per-label min reduction over a chunk of queries against each sub-bank.

    With ``weighted=True``, each bank row's distance contribution is
    divided by ``severity_weight * freshness + EPS`` (see
    ``incident.multiplier_for``): a fresh, high-severity row pulls the
    distance down (= more likely to dominate the min, = stronger
    detection signal), while a decayed row's contribution grows large
    and effectively drops out of the min. With ``weighted=False`` (the
    default) ``metas`` and ``inspection_count`` are ignored entirely so
    the function returns *bit-exact* the same values as Phase 1a — this
    is what protects existing benchmarks from drift when the weighting
    machinery is wired in but not turned on.

    When ``return_argmin=True``, the argmin row indices are returned as
    the second tuple element. They reflect the *weighted* argmin under
    ``weighted=True``, so the row marked as "hit" is the one that
    actually dominated the score, not just the geometric nearest
    neighbour. Phase 1c uses this to drive ``Bank.hit`` after every
    inference; computing min and argmin in one ``torch.min(..., dim=1)``
    call avoids a second cdist. With ``return_argmin=False`` the second
    element is an empty dict.

    Returns:
        Tuple of ``({label: min_distance_per_query}, {label: argmin_row})``.
        The argmin dict is empty when ``return_argmin=False``.
    """
    if not banks:
        return {}, {}
    out_min: dict[str, torch.Tensor] = {}
    out_argmin: dict[str, torch.Tensor] = {}
    for label, b in banks.items():
        if b is None or b.shape[0] == 0:
            continue
        # Forced MM mode: the default heuristic drops to the brute kernel
        # when BOTH operands have <= 25 rows, and that kernel has no fp16
        # support ("cdist_cuda not implemented for 'Half'") — exactly the
        # small-exemplar case (one marked NG image = ~10 rows). The MM path
        # is what large banks already take, so numerics are unchanged.
        d = torch.cdist(flat_chunk, b, compute_mode="use_mm_for_euclid_dist")
        if weighted and metas is not None and label in metas:
            mult = multiplier_for(metas[label], inspection_count)
            mult_t = torch.from_numpy(mult).to(d.device, dtype=d.dtype)
            # Broadcast per-bank-row weights across the [Q, B] distance
            # tensor; ``+ EPS`` keeps multiplier=0 (a fully decayed row)
            # from blowing up to inf and dominating the min the wrong way.
            d = d / (mult_t.unsqueeze(0) + EPS)
        if return_argmin:
            mn = torch.min(d, dim=1)
            out_min[label] = mn.values
            out_argmin[label] = mn.indices
        else:
            out_min[label] = d.min(dim=1).values
    return out_min, out_argmin


def _per_label_min(
    flat_chunk: torch.Tensor,
    banks: Mapping[str, torch.Tensor] | None,
    metas: Mapping[str, IncidentMetaArray] | None = None,
    inspection_count: int = 0,
    weighted: bool = False,
) -> dict[str, torch.Tensor]:
    """Return ``{label: min_distance_per_query}`` over each sub-bank.

    Thin wrapper over :func:`_per_label_reduce` that discards the
    argmin dict. See ``_per_label_reduce`` for the weighting semantics.
    """
    out_min, _ = _per_label_reduce(
        flat_chunk, banks, metas=metas, inspection_count=inspection_count,
        weighted=weighted, return_argmin=False,
    )
    return out_min


def _per_label_min_argmin(
    flat_chunk: torch.Tensor,
    banks: Mapping[str, torch.Tensor] | None,
    metas: Mapping[str, IncidentMetaArray] | None = None,
    inspection_count: int = 0,
    weighted: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return ``({label: min}, {label: argmin})`` over each sub-bank.

    Thin wrapper over :func:`_per_label_reduce` with ``return_argmin=True``.
    """
    return _per_label_reduce(
        flat_chunk, banks, metas=metas, inspection_count=inspection_count,
        weighted=weighted, return_argmin=True,
    )


@torch.no_grad()
def extract_distance_components(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    bank: torch.Tensor | None,
    num_neighbors: int,
    device: str,
    layers: list[int] | None = None,
    stride: int = WINDOW_STRIDE,
    critical: Mapping[str, torch.Tensor] | None = None,
    negative: Mapping[str, torch.Tensor] | None = None,
    critical_meta: Mapping[str, IncidentMetaArray] | None = None,
    negative_meta: Mapping[str, IncidentMetaArray] | None = None,
    inspection_count: int = 0,
    weighted: bool = False,
    track_argmin: bool = False,
    max_batch: int = 32,
    cdist_chunk: int = 64,
    timing: dict | None = None,
    ivf: IvfIndex | None = None,
    ivf_nprobe: int = 8,
) -> tuple[list[dict], tuple[int, int], tuple[int, int]]:
    """Run DINOv2 + bank distances for all windows; return alpha/beta-independent cache.

    Returns:
        windows: list of dicts with keys 'y', 'x', 'win_p', 'bank_topk_mean',
                 'critical_min' (dict[label, ndarray] | None),
                 'negative_min' (dict[label, ndarray] | None),
                 and (when ``track_argmin=True``) 'critical_argmin',
                 'negative_argmin' (dict[label, ndarray] | None) holding the
                 row index inside each sub-bank that won the per-query min;
                 the API uses these to call ``Bank.hit`` after scoring.
        full_shape: (H, W) of the original image
        grid_shape: (Hp_full, Wp_full) of the patch grid (after pad)
    """
    padded, (orig_h, orig_w) = pad_to_min(image_bgr)
    grid_h = padded.shape[0] // DINO_PATCH
    grid_w = padded.shape[1] // DINO_PATCH
    win_p = WINDOW_SIZE // DINO_PATCH
    # With IVF resident storage the index is the bank; ``bank`` may be None.
    n_bank = int(bank.shape[0]) if bank is not None else ivf.n_rows
    bank_dtype = bank.dtype if bank is not None else ivf.dtype
    k = min(num_neighbors, n_bank)

    offsets = sw_offsets(*padded.shape[:2], stride=stride)
    crops = [padded[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE] for (y, x) in offsets]

    # ---- DINOv2 forward (GPU-bound) -------------------------------------
    _cuda_sync(device)
    t0 = time.perf_counter()
    tokens = extract_windows_tokens_batched(
        model, crops, device, layers=layers, max_batch=max_batch
    )
    # tokens: [N, win_p, win_p, D_out]
    _cuda_sync(device)
    if timing is not None:
        timing["forward_ms"] = (time.perf_counter() - t0) * 1000.0
        timing["n_windows"] = len(offsets)

    # ---- cdist over normal/critical/negative banks (GPU-bound) ---------
    t1 = time.perf_counter()
    windows: list[dict] = []
    for idx, (y, x) in enumerate(offsets):
        flat = tokens[idx].reshape(-1, tokens.shape[-1]).to(dtype=bank_dtype)
        bank_chunks: list[torch.Tensor] = []
        crit_acc: dict[str, list[torch.Tensor]] = {}
        neg_acc: dict[str, list[torch.Tensor]] = {}
        crit_arg_acc: dict[str, list[torch.Tensor]] = {}
        neg_arg_acc: dict[str, list[torch.Tensor]] = {}
        for s in range(0, flat.shape[0], cdist_chunk):
            chunk = flat[s : s + cdist_chunk]
            # MM mode inside: see _per_label_reduce — fp16 + small tail
            # chunks would otherwise hit the Half-less brute kernel. The
            # IVF routing (when enabled) applies to the normal bank only.
            bank_chunks.append(
                _bank_topk_mean(chunk, bank, k, ivf=ivf, ivf_nprobe=ivf_nprobe)
            )
            if track_argmin:
                # One pass that yields min and argmin together; same cost as
                # the unweighted path because ``torch.min(..., dim=1)`` already
                # produces both internally.
                crit_min, crit_arg = _per_label_min_argmin(
                    chunk, critical, critical_meta, inspection_count, weighted
                )
                neg_min, neg_arg = _per_label_min_argmin(
                    chunk, negative, negative_meta, inspection_count, weighted
                )
                for lab, mins in crit_min.items():
                    crit_acc.setdefault(lab, []).append(mins)
                for lab, args in crit_arg.items():
                    crit_arg_acc.setdefault(lab, []).append(args)
                for lab, mins in neg_min.items():
                    neg_acc.setdefault(lab, []).append(mins)
                for lab, args in neg_arg.items():
                    neg_arg_acc.setdefault(lab, []).append(args)
            else:
                # Pre-Phase-1c path: untouched, so the bit-exact regression
                # tests for ``track_argmin=False`` keep passing.
                for lab, mins in _per_label_min(
                    chunk, critical, critical_meta, inspection_count, weighted
                ).items():
                    crit_acc.setdefault(lab, []).append(mins)
                for lab, mins in _per_label_min(
                    chunk, negative, negative_meta, inspection_count, weighted
                ).items():
                    neg_acc.setdefault(lab, []).append(mins)
        windows.append(
            {
                "y": y,
                "x": x,
                "win_p": win_p,
                "bank_topk_mean": torch.cat(bank_chunks).cpu().numpy().astype(np.float32),
                "critical_min": (
                    {lab: torch.cat(parts).cpu().numpy().astype(np.float32)
                     for lab, parts in crit_acc.items()}
                    if crit_acc else None
                ),
                "negative_min": (
                    {lab: torch.cat(parts).cpu().numpy().astype(np.float32)
                     for lab, parts in neg_acc.items()}
                    if neg_acc else None
                ),
                "critical_argmin": (
                    {lab: torch.cat(parts).cpu().numpy().astype(np.int64)
                     for lab, parts in crit_arg_acc.items()}
                    if crit_arg_acc else None
                ),
                "negative_argmin": (
                    {lab: torch.cat(parts).cpu().numpy().astype(np.int64)
                     for lab, parts in neg_arg_acc.items()}
                    if neg_arg_acc else None
                ),
            }
        )
    _cuda_sync(device)
    if timing is not None:
        timing["cdist_ms"] = (time.perf_counter() - t1) * 1000.0
        timing["bank_size"] = n_bank
        timing["n_critical_labels"] = len(critical or {})
        timing["n_negative_labels"] = len(negative or {})
    return windows, (orig_h, orig_w), (grid_h, grid_w)


def _stack_min(per_label: dict[str, np.ndarray] | None) -> np.ndarray | None:
    """Reduce ``{label: min_distance}`` to a single per-query min vector."""
    if not per_label:
        return None
    return np.minimum.reduce(list(per_label.values()))


@dataclass(frozen=True)
class LabelWinner:
    """The single strongest (bank_row, min_distance) for one label across all windows.

    ``bank_row`` is the row index inside ``bank.<tier>[label]`` that won
    the per-label min — usable directly with ``Bank.hit``. It's ``None``
    when the caller didn't request argmin tracking (e.g. attribution-only
    paths where the row identity doesn't matter).
    """

    bank_row: int | None
    distance: float


def per_label_winners(
    windows: list[dict],
    min_key: str,
    argmin_key: str,
) -> dict[str, LabelWinner]:
    """Walk ``windows`` and return the strongest contribution per label.

    "Strongest" = smallest distance. If ``argmin_key`` is absent on a
    window (i.e. ``track_argmin=False``) we still report the winning
    distance so attribution can use it; ``bank_row`` just stays ``None``.
    Used by both ``attribution_per_label`` (UI bar chart) and
    ``api.app._record_hits`` (Phase 1c freshness updates) so the two
    callers can never disagree on what counts as "the patch that made
    this look like 'scratch'".
    """
    out: dict[str, LabelWinner] = {}
    for w in windows:
        mins = w.get(min_key)
        if not mins:
            continue
        args = w.get(argmin_key)
        for label, m in mins.items():
            local_idx = int(np.argmin(m))
            d = float(m[local_idx])
            cur = out.get(label)
            if cur is None or d < cur.distance:
                bank_row = (
                    int(args[label][local_idx])
                    if args and label in args
                    else None
                )
                out[label] = LabelWinner(bank_row=bank_row, distance=d)
    return out


def compose_score_grid(
    windows: list[dict],
    grid_shape: tuple[int, int],
    full_shape: tuple[int, int],
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine cached distance components into a heatmap.

    Returns (full, grid):
        full: heatmap at original (H, W) resolution
        grid: heatmap at patch-grid resolution
    """
    orig_h, orig_w = full_shape
    grid_h, grid_w = grid_shape
    score_grid = np.full((grid_h, grid_w), -np.inf, dtype=np.float32)
    for w in windows:
        win_p = w["win_p"]
        score = w["bank_topk_mean"].copy()
        crit = _stack_min(w["critical_min"])
        neg = _stack_min(w["negative_min"])
        if crit is not None and alpha != 0.0:
            score = score + alpha * (1.0 / (BOOST_FLOOR + crit))
        if neg is not None and beta != 0.0:
            score = score - beta * (1.0 / (BOOST_FLOOR + neg))
        score = score.reshape(win_p, win_p)
        py = w["y"] // DINO_PATCH
        px = w["x"] // DINO_PATCH
        cur = score_grid[py : py + win_p, px : px + win_p]
        score_grid[py : py + win_p, px : px + win_p] = np.maximum(cur, score)
    finite = score_grid[score_grid != -np.inf]
    if finite.size:
        score_grid[score_grid == -np.inf] = finite.mean()

    # The grid is indexed in PADDED pixel space: window offsets come from
    # ``sw_offsets`` over the reflect-padded image, and grid_shape is
    # ``padded // DINO_PATCH``. Resizing it straight to the original size
    # squeezes the padding into the picture, so every feature slides towards
    # the top-left — measured at 47 px horizontally and 62 px vertically on a
    # 400x300 image, which pads out to 518x518.
    #
    # Resize to the extent the grid actually spans, then take the original
    # image's region out of it. A grid whose span falls a few pixels short of
    # the original (the size was not a multiple of DINO_PATCH, so the last
    # partial patch row/column was truncated) gets its edge replicated rather
    # than rescaled, which would reintroduce the same shift in miniature.
    cover_h = grid_h * DINO_PATCH
    cover_w = grid_w * DINO_PATCH
    full = cv2.resize(score_grid, (cover_w, cover_h), interpolation=cv2.INTER_LINEAR)
    if cover_h < orig_h or cover_w < orig_w:
        full = cv2.copyMakeBorder(
            full, 0, max(0, orig_h - cover_h), 0, max(0, orig_w - cover_w),
            cv2.BORDER_REPLICATE,
        )
    full = full[:orig_h, :orig_w]
    return full, score_grid


def image_topk_mean(
    windows: list[dict],
    alpha: float = 0.0,
    beta: float = 0.0,
    k: int = 10,
) -> float:
    """Image-level top-k mean over the per-window patch scores.

    Concatenates every window's per-patch composite score (the same
    ``raw + alpha/(eps+crit) - beta/(eps+neg)`` composition as
    ``compose_score_grid``, but *without* the per-pixel max-merge) and
    returns the mean of the ``k`` largest. This is the exact statistic the
    separation check computes from an image's stored rows — overlapping
    windows contribute one value per window there too — so a threshold
    picked on the check transfers to inference unchanged. ``k`` is clamped
    to the available patch count; empty ``windows`` yield ``nan``.
    """
    parts: list[np.ndarray] = []
    for w in windows:
        score = w["bank_topk_mean"].astype(np.float32, copy=False)
        crit = _stack_min(w["critical_min"])
        neg = _stack_min(w["negative_min"])
        if crit is not None and alpha != 0.0:
            score = score + alpha * (1.0 / (BOOST_FLOOR + crit))
        if neg is not None and beta != 0.0:
            score = score - beta * (1.0 / (BOOST_FLOOR + neg))
        parts.append(score)
    if not parts:
        return float("nan")
    allv = np.concatenate(parts)
    kk = max(1, min(int(k), int(allv.size)))
    top = np.partition(allv, allv.size - kk)[allv.size - kk :]
    return float(top.mean())


def attribution_per_label(
    windows: list[dict],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate per-label proximity contributions across all windows.

    Single scalar per label — ``alpha * (1 / (1 + min_distance))`` for the
    strongest (smallest-distance) (window, query) across the image, which
    mirrors what ``compose_score_grid`` would add/subtract if only that
    label were present. Drives the "how strongly this looks like 'scratch'"
    bar chart in the UI.

    Implementation note: we share ``per_label_winners`` with
    ``api.app._record_hits`` so attribution and Phase 1c hit-tracking can
    never disagree on "the patch that won this label" — that disagreement
    silently drove freshness updates onto a different row than the UI was
    showing before this was unified.
    """
    crit = per_label_winners(windows, "critical_min", "critical_argmin")
    neg = per_label_winners(windows, "negative_min", "negative_argmin")
    return (
        {lab: float(alpha * (1.0 / (BOOST_FLOOR + w.distance))) for lab, w in crit.items()},
        {lab: float(beta * (1.0 / (BOOST_FLOOR + w.distance))) for lab, w in neg.items()},
    )


@torch.no_grad()
def score_image(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    bank: torch.Tensor,
    num_neighbors: int,
    device: str,
    layers: list[int] | None = None,
    stride: int = WINDOW_STRIDE,
    critical: Mapping[str, torch.Tensor] | None = None,
    alpha: float = 0.0,
    negative: Mapping[str, torch.Tensor] | None = None,
    beta: float = 0.0,
    critical_meta: Mapping[str, IncidentMetaArray] | None = None,
    negative_meta: Mapping[str, IncidentMetaArray] | None = None,
    inspection_count: int = 0,
    weighted: bool = False,
    track_argmin: bool = False,
    max_batch: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot helper: extract components and compose with the given alpha/beta.

    See ``extract_distance_components`` for ``critical_meta``,
    ``negative_meta``, ``inspection_count``, and ``weighted``: passing
    them through here keeps the convenience wrapper aware of the
    Phase 1b weighting without forcing every caller to hand-assemble
    the two-step pipeline.
    """
    windows, full_shape, grid_shape = extract_distance_components(
        model,
        image_bgr,
        bank,
        num_neighbors,
        device,
        layers=layers,
        stride=stride,
        critical=critical,
        negative=negative,
        critical_meta=critical_meta,
        negative_meta=negative_meta,
        inspection_count=inspection_count,
        weighted=weighted,
        track_argmin=track_argmin,
        max_batch=max_batch,
    )
    return compose_score_grid(windows, grid_shape, full_shape, alpha, beta)


@torch.no_grad()
def _merge_ranges(ranges: list[tuple[int, int]], n_bank: int) -> list[tuple[int, int]]:
    """Clip to the bank, drop empties, and merge overlaps into disjoint spans.

    Overlap has to be resolved before the excluded rows are counted: sizing
    ``k`` off a double-counted total asks for more neighbours than the mask
    leaves behind, and the surplus comes back as ``inf`` averaged into the
    score.
    """
    spans = []
    for s, c in ranges:
        # Clip the END against the ORIGINAL start. Clamping the start to 0
        # first and then adding the count slides a partly-negative range
        # into the bank instead of trimming it, masking rows that were
        # never asked for.
        end = int(s) + max(0, int(c))
        s0 = max(0, int(s))
        e0 = min(int(n_bank), end)
        if e0 > s0:
            spans.append((s0, e0))
    if not spans:
        return []
    spans.sort()
    out = [spans[0]]
    for s0, e0 in spans[1:]:
        ls, le = out[-1]
        if s0 <= le:
            out[-1] = (ls, max(le, e0))
        else:
            out.append((s0, e0))
    return [(s0, e0 - s0) for s0, e0 in out]


def score_stored_features(
    features: np.ndarray,
    bank: np.ndarray | torch.Tensor | None,
    *,
    k: int = 5,
    device: str = "cpu",
    exclude_start: int = -1,
    exclude_count: int = 0,
    exclude_ranges: list[tuple[int, int]] | None = None,
    cdist_chunk: int = 512,
    ivf: IvfIndex | None = None,
    ivf_nprobe: int = 8,
) -> np.ndarray:
    """Top-k mean distance of already-extracted patch features vs the bank.

    Scores rows that are ALREADY in patch-token space (e.g. rows stored in
    the bank itself) without a model forward — the raw ``bank_topk_mean``
    component of the score formula, no alpha/beta terms.

    ``bank`` may be a numpy array (moved to ``device``) or a torch tensor
    already resident on the target device (e.g. the app's cached bank
    tensors) — the latter avoids re-uploading the bank per call.

    ``exclude_start``/``exclude_count`` mask a row range of ``bank`` out of
    the neighbour search. Used for leave-own-image-out scoring of
    normal-tier rows: without it every stored patch finds itself at
    distance 0 and the score collapses.

    ``exclude_ranges`` masks additional ranges, which is how leave-own-
    GROUP-out is expressed: a lot photographed in sequence leaves near
    duplicates of the query behind when only its own rows are excluded, so
    the caller passes every image of the lot instead. Ranges are clipped to
    the bank and merged, so overlapping input cannot inflate the neighbour
    count past the rows the mask actually leaves.
    """
    if features.size == 0:
        return np.empty((0,), dtype=np.float32)
    if bank is None:
        # IVF resident storage is the bank (see _bank_topk_mean).
        if ivf is None or not ivf.has_storage:
            raise ValueError("bank required when no IVF storage is attached")
        bank_t = None
        feats_t = torch.from_numpy(np.ascontiguousarray(features)).to(
            ivf.device, dtype=ivf.dtype
        )
        n_bank = ivf.n_rows
    else:
        if isinstance(bank, torch.Tensor):
            bank_t = bank
        else:
            bank_t = torch.from_numpy(np.ascontiguousarray(bank)).to(device)
        feats_t = torch.from_numpy(np.ascontiguousarray(features)).to(bank_t.device, dtype=bank_t.dtype)
        n_bank = int(bank_t.shape[0])
    ranges: list[tuple[int, int]] = []
    if exclude_start >= 0 and exclude_count > 0:
        ranges.append((int(exclude_start), int(exclude_count)))
    ranges.extend((int(s), int(c)) for s, c in (exclude_ranges or ()))
    ranges = _merge_ranges(ranges, n_bank)
    kk = min(int(k), max(1, n_bank - sum(c for _s, c in ranges)))
    out: list[torch.Tensor] = []
    for s in range(0, int(feats_t.shape[0]), cdist_chunk):
        chunk = feats_t[s : s + cdist_chunk]
        # MM mode inside: see _per_label_reduce (fp16 brute-kernel guard).
        out.append(
            _bank_topk_mean(
                chunk, bank_t, kk,
                excl=ranges or None,
                ivf=ivf, ivf_nprobe=ivf_nprobe,
            )
        )
    return torch.cat(out).cpu().numpy().astype(np.float32)


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / AUROC with average-ranking for tied scores."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    n_pos = pos.size
    n_neg = neg.size
    combined = np.concatenate([pos, neg])
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    order = combined.argsort()
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    rank_sum = np.zeros(counts.size, dtype=np.float64)
    np.add.at(rank_sum, inv, ranks)
    avg_ranks = rank_sum / counts
    resolved = avg_ranks[inv]
    rank_pos = resolved[:n_pos].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
