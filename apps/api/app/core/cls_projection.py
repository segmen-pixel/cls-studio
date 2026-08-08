# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""2D projection sampling of the active bank for the visualisation panel.

Downsamples each tier (with the eval cache's top-scoring rows guaranteed
in), fits a normal or contrastive PCA and returns the projected points.
Extracted verbatim from ``routers/bank.py`` (``get_projection``) during
the post-audit split; behaviour is unchanged.
"""

from __future__ import annotations

import numpy as np

from .cls_eval_cache import _eval_cache_for, eval_cache_key
from .cls_schemas import (
    ProjectionAxisInfo,
    ProjectionPoint,
    ProjectionResponse,
)
from .cls_state import ClsStudioState


def compute_projection(
    state: ClsStudioState,
    mode: str = "auto",
    max_points_per_tier: int = 800,
    alpha: float = 1.0,
    seed: int = 42,
    with_scores: bool = False,
    guarantee_top: int = 10,
    granularity: str = "patch",
) -> ProjectionResponse:
    """2D projection of the active bank's patches (see the router docstring)."""
    from clscore.projection import fit_contrastive_pca, fit_normal_pca

    bank = state.bank

    guarantee_top = max(0, int(guarantee_top))
    eval_cache = (
        _eval_cache_for(state) if (guarantee_top > 0 or granularity == "image") else {}
    )

    def _names_for(n: int, index: list[dict]) -> list[str]:
        # Row → source-image name via the per-image (name, start, count)
        # ranges kept in BankMeta. Legacy rows without an entry stay "".
        names = [""] * n
        for e in index:
            start, count = int(e.get("start", -1)), int(e.get("count", 0))
            name = str(e.get("name", ""))
            if start < 0 or count <= 0 or not name:
                continue
            for i in range(start, min(start + count, n)):
                names[i] = name
        return names

    def _ranges_for(n: int, index: list[dict]) -> list[str]:
        # Row → the row range it belongs to, as "start:count". This is the
        # leave-own-image-out key, and it deliberately is NOT the filename:
        # the store allows two photographs to share one (two folders in a zip
        # both holding img001.png), and grouping by name collapsed them onto
        # whichever range the dict happened to keep. Each duplicate was then
        # scored with the OTHER one's rows excluded and its own left in the
        # bank, so its patches found themselves at distance ~0 and a defect
        # plotted as unusually normal. Rows with no index entry stay "" and
        # are scored without any exclusion, as before.
        keys = [""] * n
        for e in index:
            start, count = int(e.get("start", -1)), int(e.get("count", 0))
            if start < 0 or count <= 0:
                continue
            key = f"{start}:{count}"
            for i in range(start, min(start + count, n)):
                keys[i] = key
        return keys

    def _forced_for(tier: str, label: str, index: list[dict]) -> list[int]:
        # Row indices of each image's cached top patches, relative to the
        # array the index entries describe. Images without a cached eval
        # (or with a pre-top_indices cache entry) contribute nothing.
        if guarantee_top <= 0:
            return []
        out: list[int] = []
        for e in index:
            start, count = int(e.get("start", -1)), int(e.get("count", 0))
            name = str(e.get("name", ""))
            if start < 0 or count <= 0 or not name:
                continue
            ent = eval_cache.get(eval_cache_key(tier, label, e))
            if not ent:
                continue
            out.extend(
                start + int(li)
                for li in ent.get("top_indices", [])[:guarantee_top]
                if 0 <= int(li) < count
            )
        return out

    def _flatten_tier(
        tier: str, tier_dict: dict[str, np.ndarray], index_dict: dict[str, list[dict]],
    ) -> tuple[np.ndarray, list[str], list[str], list[int]]:
        feats_list: list[np.ndarray] = []
        labels: list[str] = []
        names: list[str] = []
        forced: list[int] = []
        offset = 0
        for label, feats in tier_dict.items():
            if feats.size == 0:
                continue
            feats_list.append(feats)
            labels.extend([label] * feats.shape[0])
            names.extend(_names_for(feats.shape[0], index_dict.get(label, [])))
            forced.extend(offset + i for i in _forced_for(tier, label, index_dict.get(label, [])))
            offset += int(feats.shape[0])
        if not feats_list:
            return np.empty((0, bank.meta.dim), dtype=np.float32), [], [], []
        return np.concatenate(feats_list, axis=0), labels, names, forced

    normal = bank.normal.astype(np.float32, copy=False)
    normal_names = _names_for(int(normal.shape[0]), bank.meta.normal_image_index)
    normal_forced = _forced_for("normal", "", bank.meta.normal_image_index)
    critical, critical_labels, critical_names, critical_forced = _flatten_tier(
        "critical", bank.critical, bank.meta.critical_image_index)
    negative, negative_labels, negative_names, negative_forced = _flatten_tier(
        "negative", bank.negative, bank.meta.negative_image_index)

    total = {
        "normal": int(normal.shape[0]),
        "critical": int(critical.shape[0]),
        "negative": int(negative.shape[0]),
    }

    if granularity == "image":
        # One point per taught image: mean of its cached top-k patch rows
        # (mean of all rows when the separation check never scored it), so
        # the map corresponds 1:1 with the separation-check histogram.
        k_img = guarantee_top if guarantee_top > 0 else 10

        def _image_vecs(
            tier: str, label: str, index: list[dict], feats_src: np.ndarray,
        ) -> tuple[list[np.ndarray], list[tuple[str, str, float | None]]]:
            vecs: list[np.ndarray] = []
            meta: list[tuple[str, str, float | None]] = []
            for e in index:
                start, count = int(e.get("start", -1)), int(e.get("count", 0))
                name = str(e.get("name", ""))
                if start < 0 or count <= 0 or not name or start + count > int(feats_src.shape[0]):
                    continue
                rows = feats_src[start : start + count]
                ent = eval_cache.get(eval_cache_key(tier, label, e)) or {}
                lis = [int(li) for li in ent.get("top_indices", [])[:k_img] if 0 <= int(li) < count]
                # fp32 accumulator: the bank arrays are fp16 and a plain
                # fp16 mean over thousands of rows loses precision.
                vec = rows[lis].mean(axis=0, dtype=np.float32) if lis else rows.mean(axis=0, dtype=np.float32)
                tops = ent.get("top_scores", [])[:k_img]
                vecs.append(vec.astype(np.float32, copy=False))
                meta.append((label, name, float(np.mean(tops)) if tops else None))
            return vecs, meta

        n_vecs, n_meta = _image_vecs("normal", "", bank.meta.normal_image_index, normal)
        ng_vecs: list[np.ndarray] = []
        ng_meta: list[tuple[str, str, str, float | None]] = []
        for tier_name, tier_dict, index_dict in (
            ("critical", bank.critical, bank.meta.critical_image_index),
            ("negative", bank.negative, bank.meta.negative_image_index),
        ):
            for label, feats in tier_dict.items():
                if feats.size == 0:
                    continue
                vecs, meta = _image_vecs(tier_name, label, index_dict.get(label, []), feats)
                ng_vecs.extend(vecs)
                ng_meta.extend((tier_name, lb, nm, sc) for lb, nm, sc in meta)

        total_img = {
            "normal": len(n_vecs),
            "critical": sum(1 for t, *_ in ng_meta if t == "critical"),
            "negative": sum(1 for t, *_ in ng_meta if t == "negative"),
        }
        if len(n_vecs) < 2:
            return ProjectionResponse(
                mode="empty", granularity="image", axis_info=None, points=[],
                total=total_img, sampled=total_img,
            )

        normal_v = np.stack(n_vecs)
        ng_v = (
            np.stack(ng_vecs) if ng_vecs
            else np.empty((0, bank.meta.dim), dtype=np.float32)
        )
        resolved_mode = mode
        if mode == "auto":
            resolved_mode = "anomaly" if ng_v.shape[0] >= 2 else "normal"
        if resolved_mode == "anomaly" and ng_v.shape[0] >= 2:
            projector = fit_contrastive_pca(normal_v, ng_v, alpha=alpha)
        else:
            projector = fit_normal_pca(normal_v)
            resolved_mode = "normal"

        points: list[ProjectionPoint] = []
        for i, (x, y) in enumerate(projector.transform(normal_v)):
            points.append(ProjectionPoint(
                tier="normal", label="", x=float(x), y=float(y),
                image=n_meta[i][1], score=n_meta[i][2],
            ))
        if ng_v.shape[0]:
            for i, (x, y) in enumerate(projector.transform(ng_v)):
                points.append(ProjectionPoint(
                    tier=ng_meta[i][0], label=ng_meta[i][1], x=float(x), y=float(y),
                    image=ng_meta[i][2], score=ng_meta[i][3],
                ))
        return ProjectionResponse(
            mode=resolved_mode, granularity="image",
            axis_info=ProjectionAxisInfo(**projector.axis_info),
            points=points, total=total_img, sampled=total_img,
        )

    rng = np.random.default_rng(seed)
    guaranteed_included = 0

    def _sample(
        feats: np.ndarray, labels: list[str], names: list[str], k: int,
        forced: list[int],
    ) -> tuple[np.ndarray, list[str], list[str]]:
        nonlocal guaranteed_included
        n = int(feats.shape[0])
        if n == 0 or n <= k:
            guaranteed_included += len({i for i in forced if 0 <= i < n})
            return feats, labels, names
        forced_arr = np.unique(np.asarray(forced, dtype=np.int64))
        forced_arr = forced_arr[(forced_arr >= 0) & (forced_arr < n)]
        guaranteed_included += int(forced_arr.size)
        if forced_arr.size >= k:
            idx = forced_arr  # keep every guaranteed row even past the cap
        else:
            pool = np.setdiff1d(np.arange(n, dtype=np.int64), forced_arr, assume_unique=True)
            fill = rng.choice(pool, size=k - int(forced_arr.size), replace=False)
            idx = np.sort(np.concatenate([forced_arr, fill]))
        return (
            feats[idx],
            ([labels[i] for i in idx] if labels else []),
            ([names[i] for i in idx] if names else []),
        )

    k = max(2, int(max_points_per_tier))
    # The "labels" slot carries the per-row leave-own-image-out key here: the
    # normal tier has no labels, and the key has to survive the same sampling
    # permutation as the names beside it.
    normal_s, normal_range_s, normal_name_s = _sample(
        normal, _ranges_for(int(normal.shape[0]), bank.meta.normal_image_index),
        normal_names, k, normal_forced,
    )
    critical_s, critical_lbl_s, critical_name_s = _sample(
        critical, critical_labels, critical_names, k, critical_forced)
    negative_s, negative_lbl_s, negative_name_s = _sample(
        negative, negative_labels, negative_names, k, negative_forced)

    sampled = {
        "normal": int(normal_s.shape[0]),
        "critical": int(critical_s.shape[0]),
        "negative": int(negative_s.shape[0]),
    }

    # Optional per-point anomaly scores (top-k mean distance to the normal
    # bank) so the UI can color the map by how strongly the bank reacts.
    normal_score_s = critical_score_s = negative_score_s = None
    if with_scores and normal.shape[0] > 1:
        from clscore.scoring import score_stored_features

        ivf, ivf_nprobe = state.get_normal_ivf()
        bank_t = (
            None if (ivf is not None and ivf.has_storage)
            else state.get_normal_tensor()
        )
        # Normal rows live in the bank itself → leave-own-image-out, else
        # every patch finds itself at distance 0. Grouped by row RANGE, not by
        # filename: see _ranges_for.
        normal_score_s = np.zeros(int(normal_s.shape[0]), dtype=np.float32)
        groups: dict[str, list[int]] = {}
        for i, rk in enumerate(normal_range_s):
            groups.setdefault(rk, []).append(i)
        for rk, idxs in groups.items():
            st, ct = (
                (int(rk.split(":")[0]), int(rk.split(":")[1])) if rk else (-1, 0)
            )
            normal_score_s[idxs] = score_stored_features(
                np.ascontiguousarray(normal_s[idxs]), bank_t,
                exclude_start=st, exclude_count=ct,
                ivf=ivf, ivf_nprobe=ivf_nprobe or 8,
            )
        if critical_s.size:
            critical_score_s = score_stored_features(
                np.ascontiguousarray(critical_s), bank_t,
                ivf=ivf, ivf_nprobe=ivf_nprobe or 8,
            )
        if negative_s.size:
            negative_score_s = score_stored_features(
                np.ascontiguousarray(negative_s), bank_t,
                ivf=ivf, ivf_nprobe=ivf_nprobe or 8,
            )

    ng_parts = [a for a in (critical_s, negative_s) if a.size > 0]
    ng_all = np.concatenate(ng_parts, axis=0) if ng_parts else np.empty((0, bank.meta.dim), dtype=np.float32)

    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "anomaly" if (ng_all.shape[0] >= 2 and normal_s.shape[0] >= 2) else "normal"

    # Nothing to project.
    if normal_s.shape[0] < 2:
        return ProjectionResponse(mode="empty", axis_info=None, points=[], total=total, sampled=sampled)

    if resolved_mode == "anomaly" and ng_all.shape[0] >= 2:
        projector = fit_contrastive_pca(normal_s, ng_all, alpha=alpha)
    else:
        projector = fit_normal_pca(normal_s)
        resolved_mode = "normal"

    points: list[ProjectionPoint] = []
    if normal_s.size:
        xy = projector.transform(normal_s)
        points.extend(
            ProjectionPoint(
                tier="normal", label="", x=float(x), y=float(y),
                image=(normal_name_s[i] if normal_name_s else ""),
                score=(float(normal_score_s[i]) if normal_score_s is not None else None),
            )
            for i, (x, y) in enumerate(xy)
        )
    if critical_s.size:
        xy = projector.transform(critical_s)
        points.extend(
            ProjectionPoint(
                tier="critical", label=critical_lbl_s[i], x=float(x), y=float(y),
                image=(critical_name_s[i] if critical_name_s else ""),
                score=(float(critical_score_s[i]) if critical_score_s is not None else None),
            )
            for i, (x, y) in enumerate(xy)
        )
    if negative_s.size:
        xy = projector.transform(negative_s)
        points.extend(
            ProjectionPoint(
                tier="negative", label=negative_lbl_s[i], x=float(x), y=float(y),
                image=(negative_name_s[i] if negative_name_s else ""),
                score=(float(negative_score_s[i]) if negative_score_s is not None else None),
            )
            for i, (x, y) in enumerate(xy)
        )

    return ProjectionResponse(
        mode=resolved_mode,
        guaranteed=guaranteed_included,
        axis_info=ProjectionAxisInfo(**projector.axis_info),
        points=points,
        total=total,
        sampled=sampled,
    )
