# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""The parity gate has to fail on things that do not look like failures.

An exported encoder that is subtly wrong produces features of the right shape
and distances that still compute, so these tests are written as injected bugs
rather than as assertions about a happy path: each case is a mistake a
conversion can actually make, and the gate has to separate it from the
arithmetic noise of an fp16 forward.
"""

from __future__ import annotations

import numpy as np
import pytest

from clscore.coreml_parity import (
    MAX_REL_L2,
    MAX_SCORE_DRIFT_PCT,
    compare_features,
    topk_mean_distance,
)

DIM = 768


def _features(n: int, seed: int = 0) -> np.ndarray:
    """Shaped like real patch features: nonzero mean, norm in the tens."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0.4, 1.6, size=(n, DIM))).astype(np.float32)


@pytest.fixture
def data():
    q = _features(256, seed=1)
    bank = _features(4000, seed=2)
    noise_unit = float(np.linalg.norm(q, axis=1).mean()) / np.sqrt(DIM)
    return q, bank, noise_unit


def test_an_identical_encoder_passes(data):
    q, bank, _ = data
    r = compare_features(q, q.copy(), bank=bank)
    assert r["passed"] and r["rel_l2_max"] == 0.0


def test_arithmetic_noise_of_an_fp16_forward_passes(data):
    """Three times a plausible fp16 error still has to get through, or the
    gate fails good conversions and nobody trusts it."""
    q, bank, unit = data
    rng = np.random.default_rng(3)
    cand = (q + rng.normal(0, 3e-3 * unit, q.shape)).astype(np.float32)
    r = compare_features(q, cand, bank=bank)
    assert r["passed"], r


def test_the_averaged_standard_deviation_is_caught(data):
    """Core ML's ImageType carries one scalar scale, so the three ImageNet
    standard deviations have to be collapsed to about 0.226. That is the
    mistake this whole gate exists for."""
    q, bank, _ = data
    r = compare_features(q, (q * np.float32(0.226 / 0.229)).astype(np.float32), bank=bank)
    assert not r["passed"]


def test_a_systematic_scale_error_is_caught_by_drift_not_by_magnitude(data):
    """The reason there are two metrics. A 0.1% scale error is well inside the
    magnitude budget an fp16 forward needs, and still moves every score,
    because a systematic shift does not cancel in a distance the way noise
    does."""
    q, bank, _ = data
    cand = (q * np.float32(1.001)).astype(np.float32)
    r = compare_features(q, cand, bank=bank)
    assert r["rel_l2_max"] < MAX_REL_L2          # magnitude alone would allow it
    assert r["score_drift_pct_mean"] > MAX_SCORE_DRIFT_PCT
    assert not r["passed"]


def test_cosine_similarity_would_have_missed_it(data):
    """Documents why the gate does not use cosine: it is scale-invariant, so
    the two errors above are invisible to it."""
    q, _, _ = data
    cand = q * np.float32(0.226 / 0.229)
    cos = (q * cand).sum(1) / (np.linalg.norm(q, axis=1) * np.linalg.norm(cand, axis=1))
    assert cos.min() > 0.9999


def test_structural_damage_is_caught(data):
    q, bank, _ = data
    dropped = np.concatenate([q[:, :1] * 0, q[:, 1:]], axis=1).astype(np.float32)
    assert not compare_features(q, dropped, bank=bank)["passed"]
    unnormalised = (q * np.float32(0.229) + np.float32(0.485)).astype(np.float32)
    assert not compare_features(q, unnormalised, bank=bank)["passed"]


def test_a_shape_mismatch_is_a_failure_not_an_exception(data):
    q, _, _ = data
    r = compare_features(q, q[:, :10])
    assert not r["passed"] and "shape mismatch" in r["reason"]


def test_without_a_bank_the_result_says_what_it_did_not_check(data):
    """A pass that only measured magnitude must not read like a full pass."""
    q, _, _ = data
    r = compare_features(q, q.copy())
    assert r["scored"] is False
    assert "not" in r["reason"]


def test_the_scored_statistic_is_the_one_the_server_uses(data):
    q, bank, _ = data
    d = topk_mean_distance(q[:8], bank, k=10)
    brute = np.sort(np.linalg.norm(q[:8, None, :] - bank[None, :, :], axis=2), axis=1)[:, :10].mean(1)
    assert np.allclose(d, brute, rtol=1e-4, atol=1e-3)
