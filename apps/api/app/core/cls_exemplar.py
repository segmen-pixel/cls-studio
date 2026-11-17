# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Defect-exemplar row selection and GPU tensors for the active bank.

An annotated critical image contributes its heavy-severity rows; an
unannotated one falls back to its cached top-scoring rows from the eval
cache. Shared by the separation check's cmin endpoint and the
inference-time alpha term so the two can never disagree on what counts
as an exemplar. Extracted verbatim from ``routers/bank.py`` during the
post-audit split; behaviour is unchanged.
"""

from __future__ import annotations

import numpy as np

from .cls_eval_cache import _eval_cache_for, eval_cache_key
from .cls_state import ClsStudioState


def _exemplar_rows(bank, cache: dict[str, dict], auto_top: int = 10) -> dict[str, list[tuple[int, str]]]:
    """Defect-exemplar rows per critical label: ``[(row, source_image_key), ...]``.

    An annotated image contributes its heavy-severity rows; an unannotated
    one falls back to its cached top-``auto_top`` rows (nothing if it was
    never evaluated). Shared by the separation check's cmin endpoint and
    the inference-time alpha term so the two can never disagree on what
    counts as an exemplar.
    """
    from clscore.incident import SEVERITY_HEAVY

    out: dict[str, list[tuple[int, str]]] = {}
    for label, index in bank.meta.critical_image_index.items():
        feats = bank.critical.get(label)
        meta_arr = bank.critical_meta.get(label)
        if feats is None or meta_arr is None:
            continue
        rows: list[tuple[int, str]] = []
        for e in index:
            start, count = int(e.get("start", -1)), int(e.get("count", 0))
            name = str(e.get("name", ""))
            if start < 0 or count <= 0 or not name or start + count > int(feats.shape[0]):
                continue
            key = eval_cache_key("critical", label, e)
            if e.get("annotations"):
                sev = meta_arr.severity[start : start + count]
                lis = np.nonzero(sev == SEVERITY_HEAVY)[0].tolist()
            else:
                ent = cache.get(key) or {}
                lis = [int(li) for li in (ent.get("top_indices") or [])[: max(0, int(auto_top))] if 0 <= int(li) < count]
            rows.extend((start + int(li), key) for li in lis)
        if rows:
            out[label] = rows
    return out


def exemplar_critical_tensors(state: ClsStudioState) -> tuple[dict, int]:
    """``({label: exemplar_rows_tensor}, total_rows)`` on the bank device.

    The inference-time alpha term must measure distance to the defect
    exemplars only — >99% of an NG image is normal paper, so an alpha
    against the whole critical tier fires uniformly everywhere and turns
    into noise. An empty dict (no marks, no cached evals) leaves the alpha
    term inert, which degrades to the plain raw score.

    Rows are gathered straight from the numpy arrays: only the few hundred
    exemplar rows ever ride to the GPU, never the multi-GB critical tier.
    """
    import torch

    cache = _eval_cache_for(state)
    # Device/dtype anchor WITHOUT touching the normal tensor: with IVF
    # resident storage the full fp16 bank is deliberately never
    # materialised, and pulling it here just for .device would silently
    # undo that VRAM saving on every exemplar-alpha score.
    _, device, dtype = state.ensure_model()
    out: dict = {}
    total = 0
    for label, rows in _exemplar_rows(state.bank, cache).items():
        arr = state.bank.critical.get(label)
        if arr is None:
            continue
        idx = np.asarray([r for r, _ in rows], dtype=np.int64)
        out[label] = torch.from_numpy(np.ascontiguousarray(arr[idx])).to(
            device, dtype=dtype
        )
        total += int(idx.size)
    return out, total
