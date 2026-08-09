# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""score_stored_features — top-k mean distance over already-stored rows."""

from __future__ import annotations

import numpy as np

from clscore.scoring import score_stored_features


def test_topk_mean_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(50, 8)).astype(np.float32)
    q = rng.normal(size=(7, 8)).astype(np.float32)
    got = score_stored_features(q, bank, k=5, cdist_chunk=3)
    d = np.linalg.norm(q[:, None, :] - bank[None, :, :], axis=-1)
    want = np.sort(d, axis=1)[:, :5].mean(axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_exclusion_masks_own_rows() -> None:
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(30, 8)).astype(np.float32)
    q = bank[10:15]  # rows that are stored in the bank itself
    zero = score_stored_features(q, bank, k=1)
    assert np.allclose(zero, 0.0, atol=1e-4)  # each patch finds itself
    loo = score_stored_features(q, bank, k=1, exclude_start=10, exclude_count=5)
    assert (loo > 1e-3).all()  # own rows masked → real neighbour distances


def test_k_clamped_to_remaining_rows() -> None:
    rng = np.random.default_rng(2)
    bank = rng.normal(size=(4, 8)).astype(np.float32)
    q = rng.normal(size=(2, 8)).astype(np.float32)
    out = score_stored_features(q, bank, k=10, exclude_start=0, exclude_count=2)
    assert out.shape == (2,)
    assert np.isfinite(out).all()


def test_torch_tensor_bank_matches_numpy_path() -> None:
    import torch

    rng = np.random.default_rng(3)
    bank = rng.normal(size=(20, 8)).astype(np.float32)
    q = rng.normal(size=(5, 8)).astype(np.float32)
    a = score_stored_features(q, bank, k=3)
    b = score_stored_features(q, torch.from_numpy(bank), k=3)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_empty_features() -> None:
    bank = np.zeros((5, 8), dtype=np.float32)
    out = score_stored_features(np.empty((0, 8), dtype=np.float32), bank, k=3)
    assert out.shape == (0,)
