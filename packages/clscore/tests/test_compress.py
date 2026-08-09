# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Bank compression: int8 round-trip bounds + IVF routing semantics.

The IVF tests pin the contract the compression sweep validated: probing
every cluster is exactly a full scan, shrinking the probe set can only
raise (never lower) a top-k mean distance, and a query whose probed
clusters are entirely masked falls back to a full scan instead of
returning NaN.
"""

from __future__ import annotations

import numpy as np
import torch

from clscore.compress import (
    IvfIndex,
    default_n_clusters,
    quantize_int8_roundtrip,
)
from clscore.scoring import score_stored_features

RNG = np.random.default_rng(7)


def _blobs(n_per: int = 200, dim: int = 32, centers: int = 3) -> np.ndarray:
    """Well-separated gaussian blobs so k-means recovers them reliably."""
    parts = [
        RNG.normal(loc=100.0 * c, scale=1.0, size=(n_per, dim)).astype(np.float32)
        for c in range(centers)
    ]
    return np.concatenate(parts, axis=0)


# ---- int8 quantisation ------------------------------------------------------


def test_quantize_error_bounded_per_dim():
    bank = RNG.normal(size=(500, 16)).astype(np.float16)
    dq = quantize_int8_roundtrip(bank)
    assert dq.dtype == np.float16
    b32 = bank.astype(np.float32)
    scale = np.abs(b32).max(axis=0) / 127.0
    err = np.abs(dq.astype(np.float32) - b32)
    # Half a quantisation step per dim, plus fp16 storage rounding.
    tol = scale * 0.5 + np.abs(b32).max(axis=0) * 2e-3 + 1e-6
    assert (err <= tol[None, :]).all()


def test_quantize_zero_dim_and_empty():
    bank = np.zeros((10, 4), dtype=np.float16)
    bank[:, 1] = 3.0
    dq = quantize_int8_roundtrip(bank)
    assert (dq[:, 0] == 0).all()  # all-zero dim survives (no div-by-zero)
    assert np.allclose(dq[:, 1], 3.0, atol=0.05)
    empty = quantize_int8_roundtrip(np.empty((0, 8), dtype=np.float16))
    assert empty.dtype == np.float16 and empty.shape == (0, 8)


def test_quantize_matches_unchunked_reference():
    """The chunked implementation must equal the sweep's one-shot math."""
    bank = RNG.normal(size=(300, 8)).astype(np.float16)
    b32 = bank.astype(np.float32)
    scale = np.abs(b32).max(axis=0) / 127.0
    scale[scale == 0] = 1.0
    ref = (np.clip(np.round(b32 / scale), -127, 127) * scale).astype(np.float16)
    assert np.array_equal(quantize_int8_roundtrip(bank), ref)


# ---- IVF routing ------------------------------------------------------------


def test_default_n_clusters():
    assert default_n_clusters(1_000) == 16
    assert default_n_clusters(64_000) == 1_000
    assert default_n_clusters(10_000_000) == 1024


def test_ivf_full_probe_equals_full_scan():
    bank = _blobs()
    bank_t = torch.from_numpy(bank)
    idx = IvfIndex.build(bank_t, n_clusters=3, seed=1)
    feats = _blobs(n_per=20)
    base = score_stored_features(feats, bank_t)
    routed = score_stored_features(feats, bank_t, ivf=idx, ivf_nprobe=3)
    assert np.allclose(base, routed, rtol=1e-5, atol=1e-6)


def test_ivf_narrow_probe_never_lowers_scores():
    bank = _blobs()
    bank_t = torch.from_numpy(bank)
    idx = IvfIndex.build(bank_t, n_clusters=3, seed=1)
    feats = _blobs(n_per=20)
    base = score_stored_features(feats, bank_t)
    routed = score_stored_features(feats, bank_t, ivf=idx, ivf_nprobe=1)
    assert (routed >= base - 1e-5).all()


def test_ivf_all_probed_excluded_falls_back_to_full_scan():
    """Leave-own-image-out masking an entire probed cluster must not NaN."""
    bank = _blobs(n_per=100, centers=2)  # rows [0,100) = blob A, [100,200) = blob B
    bank_t = torch.from_numpy(bank)
    idx = IvfIndex.build(bank_t, n_clusters=2, seed=1)
    feats = bank[:100]  # queries = blob A itself
    # Excluding all of blob A leaves nprobe=1 queries with zero candidates.
    routed = score_stored_features(
        feats, bank_t, exclude_start=0, exclude_count=100, ivf=idx, ivf_nprobe=1
    )
    base = score_stored_features(feats, bank_t, exclude_start=0, exclude_count=100)
    assert np.isfinite(routed).all()
    assert np.allclose(routed, base, rtol=1e-5, atol=1e-6)


def test_ivf_extend_assigns_nearest_centroid():
    bank = _blobs(n_per=100, centers=2)
    bank_t = torch.from_numpy(bank)
    idx = IvfIndex.build(bank_t, n_clusters=2, seed=1)
    blob_b_cluster = int(idx.row_cluster[150].item())
    new_rows = torch.from_numpy(
        RNG.normal(loc=100.0, scale=1.0, size=(30, bank.shape[1])).astype(np.float32)
    )
    idx.extend(new_rows, index_basis=[{"name": "x.png", "start": 200, "count": 30}])
    assert int(idx.row_cluster.shape[0]) == 230
    assert (idx.row_cluster[200:] == blob_b_cluster).all()
    assert idx.index_basis and idx.index_basis[0]["name"] == "x.png"


def test_ivf_needs_rebuild_threshold():
    bank_t = torch.from_numpy(_blobs(n_per=50, centers=2))
    idx = IvfIndex.build(bank_t, n_clusters=2, seed=1)  # built_rows=100
    assert not idx.needs_rebuild(150)
    assert idx.needs_rebuild(151)


def test_ivf_save_load_roundtrip(tmp_path):
    bank_t = torch.from_numpy(_blobs(n_per=60, centers=2))
    basis = [{"name": "a.png", "start": 0, "count": 120}]
    idx = IvfIndex.build(bank_t, n_clusters=2, seed=5, int8=True, index_basis=basis)
    p = tmp_path / "ivf_index.npz"
    idx.save(p)
    loaded = IvfIndex.load(p, "cpu", torch.float32)
    assert loaded is not None
    assert torch.equal(loaded.row_cluster, idx.row_cluster)
    # Centroids persist as fp16 — small storage rounding is expected.
    assert torch.allclose(loaded.centroids, idx.centroids, atol=0.1)
    assert loaded.built_rows == 120 and loaded.seed == 5 and loaded.int8 is True
    assert loaded.index_basis == basis


def test_ivf_load_missing_or_corrupt_returns_none(tmp_path):
    assert IvfIndex.load(tmp_path / "nope.npz", "cpu", torch.float32) is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not an npz")
    assert IvfIndex.load(bad, "cpu", torch.float32) is None


# ---- gather storage (server fast path) --------------------------------------
# Parity data uses small-magnitude blobs: mm-mode cdist computes
# ||a||^2+||b||^2-2ab, and large means amplify fp32 cancellation noise into
# reduction-order differences between the full-matrix and per-cluster GEMMs
# that have nothing to do with candidate-set correctness.


def _blobs_small(n_per: int = 200, dim: int = 32, centers: int = 3) -> np.ndarray:
    parts = [
        RNG.normal(loc=5.0 * c, scale=1.0, size=(n_per, dim)).astype(np.float32)
        for c in range(centers)
    ]
    return np.concatenate(parts, axis=0)


def test_gather_matches_mask_emulation():
    """Resident-storage gather must return the mask-emulation's numbers."""
    bank = _blobs_small()
    bank_t = torch.from_numpy(bank)
    ref_idx = IvfIndex.build(bank_t, n_clusters=3, seed=1)   # mask path
    got_idx = IvfIndex.build(bank_t, n_clusters=3, seed=1)
    got_idx.set_storage(bank)                                # gather path
    assert got_idx.has_storage and not ref_idx.has_storage
    feats = _blobs_small(n_per=20)
    for npb in (1, 2, 3):
        ref = score_stored_features(feats, bank_t, ivf=ref_idx, ivf_nprobe=npb)
        got = score_stored_features(feats, None, ivf=got_idx, ivf_nprobe=npb)
        assert np.allclose(ref, got, rtol=1e-4, atol=1e-3)


def test_gather_exclusion_and_fallback_match_mask():
    bank = _blobs_small(n_per=100, centers=2)
    bank_t = torch.from_numpy(bank)
    ref_idx = IvfIndex.build(bank_t, n_clusters=2, seed=1)
    got_idx = IvfIndex.build(bank_t, n_clusters=2, seed=1)
    got_idx.set_storage(bank)
    feats = bank[:100]  # own blob fully excluded below -> fallback fires
    ref = score_stored_features(
        feats, bank_t, exclude_start=0, exclude_count=100, ivf=ref_idx, ivf_nprobe=1
    )
    got = score_stored_features(
        feats, None, exclude_start=0, exclude_count=100, ivf=got_idx, ivf_nprobe=1
    )
    assert np.isfinite(got).all()
    assert np.allclose(ref, got, rtol=1e-4, atol=1e-3)


def test_gather_int8_storage_matches_roundtrip_bank():
    """int8 codes + scale dequantise to exactly the round-tripped rows."""
    bank16 = _blobs_small().astype(np.float16)
    dq_t = torch.from_numpy(quantize_int8_roundtrip(bank16).astype(np.float32))
    ref_idx = IvfIndex.build(dq_t, n_clusters=3, seed=1, int8=True)
    got_idx = IvfIndex.build(dq_t, n_clusters=3, seed=1, int8=True)
    got_idx.set_storage(bank16)
    feats = _blobs_small(n_per=15).astype(np.float32)
    ref = score_stored_features(feats, dq_t, ivf=ref_idx, ivf_nprobe=3)
    got = score_stored_features(feats, None, ivf=got_idx, ivf_nprobe=3)
    assert np.allclose(ref, got, rtol=1e-3, atol=1e-2)


def test_extend_drops_storage():
    bank = _blobs(n_per=100, centers=2)
    idx = IvfIndex.build(torch.from_numpy(bank), n_clusters=2, seed=1)
    idx.set_storage(bank)
    assert idx.has_storage
    idx.extend(torch.from_numpy(_blobs(n_per=10, centers=1)))
    assert not idx.has_storage
