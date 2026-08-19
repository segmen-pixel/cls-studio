#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Memory-bank search benchmark on synthetic data (no dataset required).

Measures the part of cls-studio inference that scales with bank size: the
top-k mean distance search of one image's patch features against the normal
bank — full scan (fp16) vs the default compression (int8-resident storage +
IVF cluster routing, the exact production code paths). The DINOv2 forward
pass is NOT included; it is bank-size independent.

The synthetic bank is a mixture of Gaussian modes (real surfaces are
multi-modal: a bank holds a limited set of appearances, and one image's
patches concentrate in a few of them). Queries are drawn from a handful of
modes plus noise, mimicking one inspection image. This matters: IVF's
probed-cluster union stays small only when queries are coherent, which is
what production sees — on structureless uniform-random data IVF has
nothing to route and degrades to a full scan.

Usage:
    python scripts/benchmark_knn.py --device cuda:0
    python scripts/benchmark_knn.py --device cpu --bank-sizes 50000,250000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from clscore.compress import IvfIndex, quantize_int8_roundtrip  # noqa: E402
from clscore.scoring import score_stored_features  # noqa: E402

DIM = 768  # DINOv2 ViT-B patch feature dimension


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _median_ms(fn, device: str, runs: int) -> float:
    fn()  # warmup / allocator prime
    _sync(device)
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(out))


N_MODES = 256        # appearance modes in the synthetic bank
QUERY_MODES = 4      # modes one "image" draws from


def bench_one(n_rows: int, device: str, n_queries: int, k: int, nprobe: int, runs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((N_MODES, DIM)).astype(np.float32) * 4.0
    row_mode = rng.integers(0, N_MODES, size=n_rows)
    bank = (centers[row_mode]
            + rng.standard_normal((n_rows, DIM)).astype(np.float32)).astype(np.float16)
    q_modes = rng.choice(N_MODES, size=QUERY_MODES, replace=False)
    queries = (centers[rng.choice(q_modes, size=n_queries)]
               + rng.standard_normal((n_queries, DIM)).astype(np.float32))

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    bank_t = torch.from_numpy(bank).to(device, dtype=dtype)

    full_ms = _median_ms(
        lambda: score_stored_features(queries, bank_t, k=k), device, runs
    )

    t0 = time.perf_counter()
    geom = quantize_int8_roundtrip(bank)
    geom_t = torch.from_numpy(geom).to(device, dtype=dtype)
    idx = IvfIndex.build(geom_t, int8=True)
    del geom_t
    idx.set_storage(bank)
    build_s = time.perf_counter() - t0

    ivf_ms = _median_ms(
        lambda: score_stored_features(queries, None, k=k, ivf=idx, ivf_nprobe=nprobe),
        device, runs,
    )

    return {
        "rows": n_rows,
        "fp16_mb": round(n_rows * DIM * 2 / 2**20, 1),
        "int8_mb": round(n_rows * DIM * 1 / 2**20, 1),
        "clusters": int(idx.centroids.shape[0]),
        "build_s": round(build_s, 1),
        "full_ms": round(full_ms, 2),
        "ivf_ms": round(ivf_ms, 2),
        "speedup": round(full_ms / ivf_ms, 1) if ivf_ms > 0 else float("inf"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bank-sizes", default="50000,250000,1000000")
    ap.add_argument("--queries", type=int, default=2048,
                    help="query rows per measurement (~one taught image)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sizes = [int(s) for s in args.bank_sizes.split(",") if s.strip()]
    print(f"device={args.device}  queries={args.queries}  k={args.k}  "
          f"nprobe={args.nprobe}  runs={args.runs} (median)")
    print()
    print("| Bank rows | fp16 size | int8 size | Clusters | Index build | "
          "Full scan | int8+IVF | Speedup |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in sizes:
        r = bench_one(n, args.device, args.queries, args.k, args.nprobe,
                      args.runs, args.seed)
        print(f"| {r['rows']:,} | {r['fp16_mb']} MB | {r['int8_mb']} MB "
              f"| {r['clusters']} | {r['build_s']} s "
              f"| {r['full_ms']} ms | {r['ivf_ms']} ms | {r['speedup']}x |")


if __name__ == "__main__":
    main()
