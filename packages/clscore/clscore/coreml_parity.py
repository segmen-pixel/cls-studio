# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Does an exported encoder still land queries in the bank's space?

A converted encoder that is subtly wrong does not raise. It returns features
of the right shape, the distances still compute, and the verdicts move. So an
export is only a deliverable once it has been compared against the server that
built the bank, and this module is that comparison.

Two metrics, because one is not enough.

``rel_l2`` catches gross damage -- a dropped channel, a transposed layout, no
normalisation at all. It cannot catch a systematic scale error, and cosine
similarity is worse still: it is scale-invariant, so the exact mistake Core
ML's ImageType would force (three ImageNet standard deviations collapsed to
one averaged scalar) reads as cosine 0.9999998, indistinguishable from perfect.

``score_drift`` is the one that decides. It re-scores sampled queries against
the real bank and asks how far the top-k mean distance moved, which is the
number the operator actually sees. Random error partially cancels in a
distance; systematic error does not, so the same magnitude of drift weighs
about five times heavier when it is a bug than when it is arithmetic noise.

Measured on a real 40,960-row bank (2026-08-18), sampling 512 queries against
20,000 rows:

    perturbation             rel_l2 max   score drift
    fp16 store round-trip     0.000e+00       0.0000%
    random 1e-4               1.178e-04       0.0021%
    random 1e-3               1.190e-03       0.0105%
    random 3e-3               3.563e-03       0.0335%
    random 1e-2               1.171e-02       0.1200%
    systematic +0.01%         1.000e-04       0.0066%
    systematic +0.1%          1.000e-03       0.0586%
    std 0.229 -> 0.226        1.310e-02       0.5159%

The thresholds sit in the gap that table opens up: random noise an order of
magnitude worse than a plausible fp16 forward still passes, while the mildest
systematic error worth investigating does not.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MAX_REL_L2", "MAX_SCORE_DRIFT_PCT", "compare_features", "topk_mean_distance"]

# Gross-error gate. A pessimistic fp16 forward lands near 3e-3; anything past
# this is structural rather than arithmetic.
MAX_REL_L2 = 5e-3
# Systematic-error gate, in percent of the server's own score. Sits between
# random 3e-3 (0.0335%) and a systematic 0.1% scale error (0.0586%).
MAX_SCORE_DRIFT_PCT = 0.05


def topk_mean_distance(queries: np.ndarray, bank: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the k nearest bank rows, per query.

    The same statistic the server scores with, so a drift measured here is a
    drift the operator would have seen.
    """
    q = np.asarray(queries, dtype=np.float32)
    b = np.asarray(bank, dtype=np.float32)
    k = max(1, min(int(k), b.shape[0]))
    sq = (q * q).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (q @ b.T)
    return np.sort(np.sqrt(np.maximum(sq, 0.0)), axis=1)[:, :k].mean(1)


def compare_features(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    bank: np.ndarray | None = None,
    k: int = 10,
) -> dict:
    """Compare an exported encoder's output against the server's.

    Args:
        reference: ``[N, D]`` features from the server path.
        candidate: ``[N, D]`` features from the exported encoder.
        bank: Rows to score against. Without it only the shape and magnitude
            checks run, and ``passed`` reflects those alone -- say so to the
            caller rather than implying the real gate was met.
        k: Neighbours in the scored statistic.
    """
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        return {"passed": False, "reason": f"shape mismatch: {ref.shape} vs {cand.shape}"}

    denom = np.maximum(np.linalg.norm(ref, axis=1), 1e-12)
    rel = np.linalg.norm(ref - cand, axis=1) / denom
    out: dict = {
        "rows": int(ref.shape[0]),
        "dim": int(ref.shape[1]),
        "rel_l2_max": float(rel.max()),
        "rel_l2_mean": float(rel.mean()),
        "max_abs": float(np.abs(ref - cand).max()),
        "rel_l2_limit": MAX_REL_L2,
        "scored": False,
    }
    out["passed"] = out["rel_l2_max"] <= MAX_REL_L2

    if bank is None:
        out["reason"] = "no bank supplied: magnitude checked, score drift not"
        return out

    base = topk_mean_distance(ref, bank, k)
    moved = topk_mean_distance(cand, bank, k)
    drift = np.abs(moved - base) / np.maximum(base, 1e-9) * 100.0
    out.update(
        scored=True,
        k=int(k),
        score_drift_pct_mean=float(drift.mean()),
        score_drift_pct_max=float(drift.max()),
        score_drift_limit_pct=MAX_SCORE_DRIFT_PCT,
    )
    out["passed"] = bool(out["passed"] and out["score_drift_pct_mean"] <= MAX_SCORE_DRIFT_PCT)
    if not out["passed"]:
        out["reason"] = "exported encoder does not reproduce the bank\'s feature space"
    return out
