# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Adaptive VRAM sizing: the cdist chunk shrinks as the bank grows / VRAM
tightens, so a large OK bank never OOMs, and probe_max_batch degrades safely
off-CUDA."""
from __future__ import annotations

from clscore.feature_extractor import probe_max_batch
from clscore.scoring import safe_cdist_chunk


def test_cdist_chunk_shrinks_as_bank_grows():
    free = 8 * 1024**3  # 8 GB free
    small = safe_cdist_chunk(10_000, free)
    big = safe_cdist_chunk(5_000_000, free)
    assert small > big  # a bigger bank forces a smaller query chunk
    assert big >= 32     # never below the floor


def test_cdist_chunk_grows_with_free_vram():
    n = 1_000_000
    lean = safe_cdist_chunk(n, 2 * 1024**3)
    fat = safe_cdist_chunk(n, 20 * 1024**3)
    assert fat > lean
    assert fat <= 4096   # never above the ceiling


def test_cdist_chunk_budget_is_respected():
    # chunk * n_bank * elem_bytes * overhead must fit under free * safety
    free = 4 * 1024**3
    n = 2_000_000
    chunk = safe_cdist_chunk(n, free, elem_bytes=2, overhead=3.0, safety=0.6)
    assert chunk * n * 2 * 3.0 <= free * 0.6 + 1  # +1 for the floor/int rounding
    assert chunk >= 32


def test_cdist_chunk_clamped_and_edge_cases():
    assert safe_cdist_chunk(0, 8 * 1024**3) == 4096          # empty bank -> ceiling
    assert safe_cdist_chunk(1, 1) == 32                       # tiny budget -> floor
    assert safe_cdist_chunk(10, 10**12) == 4096               # huge budget -> ceiling


def test_probe_max_batch_off_cuda_returns_top_candidate():
    # No CUDA in CI: the probe must not touch the GPU and returns the largest
    # candidate unprobed (model is unused on this path).
    assert probe_max_batch(model=None, device="cpu", candidates=(64, 32, 8)) == 64
    assert probe_max_batch(model=None, device="mps", candidates=(48, 16)) == 48
