# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Bank-compression settings: endpoint round-trip, cache/fingerprint coupling.

Every test restores the default settings on exit — the settings file is
process-global (shared with every other test in the session), and eval-cache
fingerprints embed the current settings.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from app.core.runtime_compression import (
    DEFAULTS,
    read_compression_settings,
    save_compression_settings,
)
from clscore.bank import Bank


@contextlib.contextmanager
def _restored_defaults():
    try:
        yield
    finally:
        save_compression_settings(**DEFAULTS)


def test_defaults_are_compressed():
    assert read_compression_settings() == {"int8": True, "ivf": True, "ivf_nprobe": 8}


def test_endpoint_roundtrip_and_cache_invalidation(client):
    from app.core.cls_state import get_state

    with _restored_defaults():
        state = get_state()
        state._tensor_cache = {"sentinel": object()}
        r = client.put(
            "/api/v1/system/compression",
            json={"int8": False, "ivf": True, "ivf_nprobe": 4},
        )
        assert r.status_code == 200
        assert r.json() == {"int8": False, "ivf": True, "ivf_nprobe": 4}
        # The PUT must drop cached bank tensors so the next score rebuilds
        # them under the new settings.
        assert state._tensor_cache is None
        assert client.get("/api/v1/system/compression").json()["int8"] is False


def test_endpoint_clamps_nprobe(client):
    with _restored_defaults():
        r = client.put(
            "/api/v1/system/compression",
            json={"int8": True, "ivf": True, "ivf_nprobe": 999},
        )
        assert r.status_code == 422  # ge/le validation, not silent clamping


def test_eval_fingerprint_tracks_compression_settings():
    from app.core.cls_eval_cache import _bank_content_fingerprint, _eval_fingerprint

    bank = Bank(normal=np.zeros((4, 8), dtype=np.float16))
    with _restored_defaults():
        fp_on = _eval_fingerprint(bank)
        cfp_on = _bank_content_fingerprint(bank)
        save_compression_settings(int8=False, ivf=False, ivf_nprobe=8)
        assert _eval_fingerprint(bank) != fp_on
        assert _bank_content_fingerprint(bank) != cfp_on


def _cpu_state_with_bank(monkeypatch, n_rows: int = 64):
    """An ClsStudioState with a random bank and a stubbed (CPU) model."""
    from app.core.cls_state import ClsStudioState

    rng = np.random.default_rng(3)
    st = ClsStudioState()
    st.bank = Bank(normal=rng.normal(size=(n_rows, 16)).astype(np.float16))

    def fake_ensure():
        st._device, st._dtype = "cpu", torch.float32
        return None, "cpu", torch.float32

    monkeypatch.setattr(st, "ensure_model", fake_ensure)
    return st


def test_get_normal_tensor_respects_int8(monkeypatch):
    with _restored_defaults():
        st = _cpu_state_with_bank(monkeypatch)
        save_compression_settings(int8=False, ivf=False, ivf_nprobe=8)
        raw = st.get_normal_tensor().clone()
        st.mark_dirty()
        save_compression_settings(int8=True, ivf=False, ivf_nprobe=8)
        quantised = st.get_normal_tensor()
        assert not torch.equal(raw, quantised)
        # Quantisation error is bounded — the tensors stay close.
        assert torch.allclose(raw, quantised, atol=0.05)


def test_get_normal_ivf_gates_and_builds(monkeypatch):
    import clscore.compress as compress

    with _restored_defaults():
        st = _cpu_state_with_bank(monkeypatch)
        # Bank far below MIN_IVF_ROWS: routing declines even though enabled.
        assert st.get_normal_ivf() == (None, 0)
        # Lower the gate: the index builds, caches, and reports nprobe.
        monkeypatch.setattr(compress, "MIN_IVF_ROWS", 10)
        idx, nprobe = st.get_normal_ivf()
        assert idx is not None and nprobe == 8
        assert int(idx.row_cluster.shape[0]) == 64
        assert idx.has_storage  # resident gather storage attached
        assert st.get_normal_ivf()[0] is idx  # cached until mark_dirty
        save_compression_settings(int8=True, ivf=False, ivf_nprobe=8)
        st.mark_dirty()
        assert st.get_normal_ivf() == (None, 0)  # setting off wins
