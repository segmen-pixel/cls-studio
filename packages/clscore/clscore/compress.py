# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Bank compression: int8 quantisation + IVF cluster routing (normal tier).

Both transforms were validated against production scoring on every internal
project (2026-07 compression sweep: int8 changed no verdict on 6/6 projects,
IVF nprobe=8 was AUROC-neutral or better on 6/6), so the math here mirrors
that sweep exactly:

- ``quantize_int8_roundtrip``: per-dim symmetric int8 quantise -> dequantise.
  The bank *on disk* stays fp16 — quantisation is applied when the scoring
  tensor is built, so turning the setting off restores full precision with
  no data migration.
- ``IvfIndex``: k-means cluster routing over the normal bank. Scoring masks
  every bank column outside the ``nprobe`` clusters nearest to each query,
  which yields exactly the candidate set a classic IVF-Flat gather would
  scan — the sweep validated this candidate-set semantics, so we keep it
  for the server runtime (fidelity first; the iPad export runtime is where
  a gather-based kernel buys real speed).

The index is derived data: it can always be rebuilt from the bank, so a
missing / stale ``ivf_index.npz`` silently degrades to a rebuild, never an
error. Teaching appends rows; those are assigned to their nearest existing
centroid incrementally (k-means quality decays slowly), and a full rebuild
happens only once the bank outgrows the built size by ``REBUILD_GROWTH``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "IVF_INDEX_FILE",
    "MIN_IVF_ROWS",
    "REBUILD_GROWTH",
    "IvfIndex",
    "default_n_clusters",
    "dequantize_int8",
    "normal_index_basis",
    "quantize_int8",
    "quantize_int8_roundtrip",
]

# Matches clscore.scoring: the default cdist heuristic drops to a brute
# kernel with no fp16 support when both operands are tiny.
_CDIST_MODE = "use_mm_for_euclid_dist"

IVF_INDEX_FILE = "ivf_index.npz"
# Below this row count a full scan is already sub-millisecond and the
# clusters would be too small to route meaningfully.
MIN_IVF_ROWS = 20_000
# Rebuild the k-means once the bank has grown past built_rows * REBUILD_GROWTH;
# until then appended rows ride on nearest-centroid assignment.
REBUILD_GROWTH = 1.5
_QUANT_CHUNK = 200_000  # rows per pass so a multi-GB bank never casts to fp32 whole


def quantize_int8(bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim symmetric int8 quantisation. Returns ``(codes int8, scale fp32)``.

    Chunked over rows: the abs-max scale needs one full pass per dim, the
    quantise a second, so peak extra memory stays at ``_QUANT_CHUNK x D``
    fp32 instead of the whole bank. Zero-only dims get scale 1.0 so they
    round-trip to exact zero instead of dividing by zero.
    """
    dim = int(bank.shape[1])
    absmax = np.zeros(dim, dtype=np.float32)
    for s in range(0, bank.shape[0], _QUANT_CHUNK):
        chunk = np.abs(bank[s : s + _QUANT_CHUNK].astype(np.float32))
        if chunk.size:
            np.maximum(absmax, chunk.max(axis=0), out=absmax)
    scale = absmax / 127.0
    scale[scale == 0] = 1.0
    codes = np.empty(bank.shape, dtype=np.int8)
    for s in range(0, bank.shape[0], _QUANT_CHUNK):
        b32 = bank[s : s + _QUANT_CHUNK].astype(np.float32)
        codes[s : s + _QUANT_CHUNK] = np.clip(np.round(b32 / scale), -127, 127)
    return codes, scale


def dequantize_int8(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_int8`, as fp16 (chunked)."""
    out = np.empty(codes.shape, dtype=np.float16)
    for s in range(0, codes.shape[0], _QUANT_CHUNK):
        out[s : s + _QUANT_CHUNK] = (
            codes[s : s + _QUANT_CHUNK].astype(np.float32) * scale
        ).astype(np.float16)
    return out


def quantize_int8_roundtrip(bank: np.ndarray) -> np.ndarray:
    """Per-dim symmetric int8 quantise -> dequantise, returned as fp16."""
    if bank.size == 0:
        return bank.astype(np.float16, copy=False)
    codes, scale = quantize_int8(bank)
    return dequantize_int8(codes, scale)


def default_n_clusters(n_rows: int) -> int:
    """~64+ rows per cluster, capped at 1024 (the sweep's recipe)."""
    return min(1024, max(16, int(n_rows) // 64))


def normal_index_basis(normal_image_index: list[dict]) -> list[dict]:
    """Membership-only view of the normal tier's per-image row index.

    The index's validity contract is "the rows I assigned are still the
    rows at those positions". Appends extend this list; deletes compact
    row ranges and therefore shift positions. Comparing the stored basis
    against the current one (prefix match = append-only growth) decides
    between incremental assignment and a rebuild.
    """
    return [
        {"name": str(e.get("name", "")), "start": int(e.get("start", -1)), "count": int(e.get("count", 0))}
        for e in normal_image_index
    ]


def _kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked Lloyd's k-means in fp32. Returns (centroids [k, D], assign [N])."""
    n = int(x.shape[0])
    g = torch.Generator(device=x.device).manual_seed(seed)
    cent = x[torch.randperm(n, generator=g, device=x.device)[:k]].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        for s in range(0, n, 65536):
            d = torch.cdist(x[s : s + 65536], cent, compute_mode=_CDIST_MODE)
            assign[s : s + 65536] = d.argmin(dim=1)
        cent_new = torch.zeros_like(cent)
        cnt = torch.zeros(k, device=x.device)
        cent_new.index_add_(0, assign, x)
        cnt.index_add_(0, assign, torch.ones(n, device=x.device))
        mask = cnt > 0
        cent_new[mask] /= cnt[mask].unsqueeze(1)
        cent_new[~mask] = cent[~mask]  # empty cluster: keep the old centroid
        cent = cent_new
    return cent, assign


class IvfIndex:
    """Cluster-routing index over the normal bank.

    ``centroids``: [C, D] on the scoring device, scoring dtype.
    ``row_cluster``: [N] int64 on the scoring device — cluster id per bank
    row, in bank row order (so exclusion masks by row range still line up).
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        row_cluster: torch.Tensor,
        *,
        built_rows: int,
        seed: int = 42,
        int8: bool = False,
        index_basis: list[dict] | None = None,
    ) -> None:
        self.centroids = centroids
        self.row_cluster = row_cluster
        self.built_rows = int(built_rows)
        self.seed = int(seed)
        self.int8 = bool(int8)
        self.index_basis: list[dict] = index_basis or []
        # Resident bank storage (see set_storage); None until attached.
        self._rows_t: torch.Tensor | None = None
        self._scale_t: torch.Tensor | None = None
        self._perm_t: torch.Tensor | None = None
        self._sorted_clusters: torch.Tensor | None = None
        self._offsets: list[int] | None = None

    # ---- build / extend ---------------------------------------------------

    @classmethod
    def build(
        cls,
        bank_t: torch.Tensor,
        *,
        n_clusters: int | None = None,
        iters: int = 25,
        seed: int = 42,
        int8: bool = False,
        index_basis: list[dict] | None = None,
    ) -> IvfIndex:
        """K-means over the scoring tensor (post-quantisation when int8 is on,
        so routing and scoring always see the same geometry)."""
        k = n_clusters or default_n_clusters(int(bank_t.shape[0]))
        cent, assign = _kmeans(bank_t.float(), k, iters=iters, seed=seed)
        return cls(
            cent.to(bank_t.dtype),
            assign,
            built_rows=int(bank_t.shape[0]),
            seed=seed,
            int8=int8,
            index_basis=index_basis,
        )

    def assign_rows(self, rows_t: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid cluster ids for new rows (incremental teach path)."""
        out = torch.empty(int(rows_t.shape[0]), dtype=torch.long, device=rows_t.device)
        for s in range(0, int(rows_t.shape[0]), 65536):
            d = torch.cdist(
                rows_t[s : s + 65536].to(self.centroids.dtype),
                self.centroids,
                compute_mode=_CDIST_MODE,
            )
            out[s : s + 65536] = d.argmin(dim=1)
        return out

    def extend(self, new_rows_t: torch.Tensor, index_basis: list[dict] | None = None) -> None:
        """Append nearest-centroid assignments for rows taught since build.

        Drops any attached storage: the cluster-sorted layout is stale the
        moment new rows exist, and re-attaching is a cheap re-sort compared
        to silently searching a layout that no longer covers the bank.
        """
        if int(new_rows_t.shape[0]) == 0:
            return
        self.row_cluster = torch.cat([self.row_cluster, self.assign_rows(new_rows_t)])
        if index_basis is not None:
            self.index_basis = index_basis
        self.drop_storage()

    def needs_rebuild(self, n_rows: int) -> bool:
        return n_rows > self.built_rows * REBUILD_GROWTH

    # ---- scoring-side mask ------------------------------------------------

    def allowed_mask(self, query_chunk: torch.Tensor, nprobe: int) -> torch.Tensor:
        """bool [Q, N]: True where the bank row is inside one of the query's
        ``nprobe`` nearest clusters. Exactly the sweep's candidate set."""
        dc = torch.cdist(
            query_chunk.to(self.centroids.dtype), self.centroids, compute_mode=_CDIST_MODE
        )
        npb = max(1, min(int(nprobe), int(self.centroids.shape[0])))
        probe = torch.topk(dc, npb, dim=1, largest=False).indices  # [Q, npb]
        allowed = torch.zeros(
            int(query_chunk.shape[0]), int(self.row_cluster.shape[0]),
            dtype=torch.bool, device=query_chunk.device,
        )
        row_c = self.row_cluster.unsqueeze(0)  # [1, N]
        for j in range(npb):
            allowed |= row_c == probe[:, j : j + 1]
        return allowed

    # ---- resident storage + gather search ---------------------------------

    @property
    def has_storage(self) -> bool:
        return self._rows_t is not None

    @property
    def n_rows(self) -> int:
        return int(self.row_cluster.shape[0])

    @property
    def device(self) -> torch.device:
        return self.centroids.device

    @property
    def dtype(self) -> torch.dtype:
        return self.centroids.dtype

    def set_storage(self, bank: np.ndarray) -> None:
        """Make the index self-contained: bank rows resident in cluster order.

        ``bank`` is the raw fp16 normal bank in original row order. Rows are
        sorted by cluster (a probed cluster becomes one contiguous range)
        and held on the scoring device — int8 codes + per-dim scale when
        the index was built for int8 (half the fp16 footprint), fp16
        otherwise. ``perm`` (sorted position -> original row id) keeps
        leave-own-image-out exclusion working against original row ranges.
        With storage attached, searches gather only the probed clusters and
        the full bank tensor never needs to exist on the device — this is
        where IVF's speed and the int8 VRAM saving materialise server-side.
        """
        assign = self.row_cluster.cpu().numpy()
        perm = np.argsort(assign, kind="stable")
        counts = np.bincount(assign, minlength=int(self.centroids.shape[0]))
        self._offsets = [0] + [int(x) for x in np.cumsum(counts)]
        self._perm_t = torch.from_numpy(perm.astype(np.int64)).to(self.device)
        self._sorted_clusters = torch.from_numpy(
            np.ascontiguousarray(assign[perm].astype(np.int64))
        ).to(self.device)
        if self.int8:
            codes, scale = quantize_int8(bank)
            self._rows_t = torch.from_numpy(codes[perm]).to(self.device)
            self._scale_t = torch.from_numpy(scale).to(self.device)
        else:
            self._rows_t = torch.from_numpy(
                np.ascontiguousarray(bank[perm])
            ).to(self.device, dtype=self.dtype)
            self._scale_t = None

    def drop_storage(self) -> None:
        self._rows_t = None
        self._scale_t = None
        self._perm_t = None
        self._sorted_clusters = None
        self._offsets = None

    def _dequant(self, sub: torch.Tensor) -> torch.Tensor:
        """Rows to the scoring dtype. fp32 multiply -> fp16 cast reproduces
        :func:`dequantize_int8` exactly, so gathered distances see the same
        values as scoring against a round-tripped fp16 bank."""
        if self._scale_t is not None:
            sub = (sub.to(torch.float32) * self._scale_t).to(torch.float16)
        return sub.to(self.dtype)

    def search_topk_mean(
        self,
        chunk: torch.Tensor,
        k: int,
        *,
        excl_ranges: list[tuple[int, int]] | None = None,
        nprobe: int = 8,
    ) -> torch.Tensor:
        """Top-``k`` mean distance of each query against its probed clusters.

        Same candidate set as ``allowed_mask`` + full cdist (the semantics
        the compression sweep validated), computed on a gather of the
        CHUNK's probed-cluster union: queries inside one window / one image
        are spatially coherent, so the union stays a small fraction of the
        bank and one dense cdist against it beats both the full scan and a
        per-cluster loop (hundreds of tiny kernel launches). Queries whose
        probed candidates are entirely masked out fall back to a scan of
        every cluster so a finite score always comes back.
        """
        q = chunk.to(self.dtype)
        dc = torch.cdist(q, self.centroids, compute_mode=_CDIST_MODE)
        npb = max(1, min(int(nprobe), int(self.centroids.shape[0])))
        probe = torch.topk(dc, npb, dim=1, largest=False).indices  # [Q, npb]
        mean = self._union_topk_mean(q, k, excl_ranges, probe)
        bad = ~torch.isfinite(mean)
        if bool(bad.any()):
            mean[bad] = self._full_topk_mean(q[bad], k, excl_ranges)
        return mean

    def _union_topk_mean(
        self,
        q: torch.Tensor,
        k: int,
        excl_ranges: list[tuple[int, int]] | None,
        probe: torch.Tensor,
    ) -> torch.Tensor:
        """One dense cdist against the union of the chunk's probed clusters,
        then per-query masking down to each query's own probe set."""
        assert self._offsets is not None
        uniq = torch.unique(probe).cpu().tolist()
        parts = [
            np.arange(self._offsets[c], self._offsets[c + 1], dtype=np.int64)
            for c in uniq
            if self._offsets[c + 1] > self._offsets[c]
        ]
        if not parts:
            return torch.full(
                (int(q.shape[0]),), float("nan"), device=q.device, dtype=q.dtype
            )
        row_ids = torch.from_numpy(np.concatenate(parts)).to(q.device)
        sub = self._dequant(self._rows_t.index_select(0, row_ids))
        d = torch.cdist(q, sub, compute_mode=_CDIST_MODE)
        sub_clusters = self._sorted_clusters.index_select(0, row_ids)
        allowed = torch.zeros(
            int(q.shape[0]), int(row_ids.shape[0]), dtype=torch.bool, device=q.device
        )
        for j in range(int(probe.shape[1])):
            allowed |= sub_clusters.unsqueeze(0) == probe[:, j : j + 1]
        d.masked_fill_(~allowed, float("inf"))
        if excl_ranges:
            orig = self._perm_t.index_select(0, row_ids)
            em = torch.zeros_like(orig, dtype=torch.bool)
            for s0, cnt in excl_ranges:
                em |= (orig >= s0) & (orig < s0 + cnt)
            d.masked_fill_(em.unsqueeze(0), float("inf"))
        vals = torch.topk(d, min(k, int(d.shape[1])), dim=1, largest=False).values
        finite = torch.where(torch.isinf(vals), torch.nan, vals)
        return torch.nanmean(finite, dim=1)

    def _full_topk_mean(
        self,
        q: torch.Tensor,
        k: int,
        excl_ranges: list[tuple[int, int]] | None,
        slab: int = 65536,
    ) -> torch.Tensor:
        """Full scan over the resident storage in slabs (fallback path)."""
        n = self.n_rows
        buf = torch.full(
            (int(q.shape[0]), k), float("inf"), device=q.device, dtype=q.dtype
        )
        for a in range(0, n, slab):
            b = min(a + slab, n)
            d = torch.cdist(q, self._dequant(self._rows_t[a:b]), compute_mode=_CDIST_MODE)
            if excl_ranges:
                perm = self._perm_t[a:b]
                em = torch.zeros_like(perm, dtype=torch.bool)
                for s0, cnt in excl_ranges:
                    em |= (perm >= s0) & (perm < s0 + cnt)
                d.masked_fill_(em.unsqueeze(0), float("inf"))
            vals = torch.topk(d, min(k, b - a), dim=1, largest=False).values
            buf = torch.topk(
                torch.cat([buf, vals], dim=1), k, dim=1, largest=False
            ).values
        finite = torch.where(torch.isinf(buf), torch.nan, buf)
        return torch.nanmean(finite, dim=1)

    # ---- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist to ``path`` (atomic via tmp-then-replace, like fsio)."""
        path = Path(path)
        meta = {
            "seed": self.seed,
            "n_clusters": int(self.centroids.shape[0]),
            "built_rows": self.built_rows,
            "int8": self.int8,
            "index_basis": self.index_basis,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                centroids=self.centroids.float().cpu().numpy().astype(np.float16),
                row_cluster=self.row_cluster.cpu().numpy().astype(np.int32),
                meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            )
            f.flush()
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, device: str, dtype: torch.dtype) -> IvfIndex | None:
        """Load a persisted index, or None when unreadable (rebuild instead)."""
        try:
            with np.load(Path(path)) as z:
                meta = json.loads(bytes(z["meta_json"].tobytes()).decode("utf-8"))
                centroids = torch.from_numpy(np.ascontiguousarray(z["centroids"])).to(device, dtype=dtype)
                row_cluster = torch.from_numpy(z["row_cluster"].astype(np.int64)).to(device)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        return cls(
            centroids,
            row_cluster,
            built_rows=int(meta.get("built_rows", 0)),
            seed=int(meta.get("seed", 42)),
            int8=bool(meta.get("int8", False)),
            index_basis=list(meta.get("index_basis", [])),
        )
