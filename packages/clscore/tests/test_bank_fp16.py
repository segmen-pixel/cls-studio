# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""float16 bank storage.

GPU scoring always casts bank rows to fp16 before the distance computation,
so holding/persisting fp16 applies the exact same quantisation one step
earlier: CUDA verdicts are bit-identical while RAM, disk and export
packages halve. These tests pin the dtype invariant across every mutation
path so a stray fp32 concat can't silently double memory again.
"""

from __future__ import annotations

import numpy as np

from clscore.bank import BANK_DTYPE, Bank


def _f32(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_bank_holds_and_persists_float16(tmp_path):
    b = Bank(normal=_f32(6, seed=1))
    b.append("critical", _f32(3, seed=2), label="x", image_name="a.png")
    assert b.normal.dtype == BANK_DTYPE
    assert b.critical["x"].dtype == BANK_DTYPE
    b.save(tmp_path)
    assert np.load(tmp_path / "bank.npy").dtype == BANK_DTYPE
    assert np.load(tmp_path / "critical" / "x.npy").dtype == BANK_DTYPE


def test_legacy_fp32_bank_loads_as_quantised_fp16(tmp_path):
    normal32 = _f32(5, seed=3)
    np.save(tmp_path / "bank.npy", normal32)
    b = Bank.load(tmp_path)
    assert b.normal.dtype == BANK_DTYPE
    # Same rounding the GPU cast applied before this change.
    np.testing.assert_array_equal(b.normal, normal32.astype(np.float16))


def test_fp16_round_trip_is_bit_exact_after_first_save(tmp_path):
    b = Bank(normal=_f32(5, seed=4))
    b.save(tmp_path)
    b2 = Bank.load(tmp_path)
    np.testing.assert_array_equal(b2.normal, b.normal)


def test_append_and_remove_keep_bank_dtype():
    b = Bank(normal=np.zeros((0, 8), dtype=np.float32))
    b.append("normal", _f32(4, seed=5), image_name="a.png")
    b.append("normal", _f32(4, seed=6), image_name="b.png")
    assert b.normal.dtype == BANK_DTYPE
    b.remove_images("normal", None, ["a.png"])
    assert b.normal.dtype == BANK_DTYPE
    assert b.normal.shape == (4, 8)


def test_to_tensors_casts_from_fp16():
    import torch

    b = Bank(normal=_f32(4, seed=7))
    b.append("critical", _f32(2, seed=8), label="x")
    n_t, c_d, _ = b.to_tensors("cpu", dtype=torch.float32)
    assert n_t.dtype == torch.float32
    assert c_d["x"].dtype == torch.float32
    np.testing.assert_allclose(n_t.numpy(), b.normal.astype(np.float32))
