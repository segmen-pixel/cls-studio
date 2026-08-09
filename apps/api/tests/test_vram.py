# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""VRAM budget + idle release: settings, endpoints, lease, reaper.

Every test restores the default settings on exit — the settings file is
process-global and shared with every other test in the session.
"""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import torch

from app.core.runtime_vram import (
    DEFAULTS,
    MIN_IDLE_SECONDS,
    read_vram_settings,
    save_vram_settings,
)
from clscore.bank import Bank


@contextlib.contextmanager
def _restored_defaults():
    try:
        yield
    finally:
        save_vram_settings(**DEFAULTS)


def test_defaults():
    assert read_vram_settings() == {
        "budget_mb": 8192,
        "idle_release": True,
        "idle_seconds": 60,
        "drop_bank": False,
        "drop_model": False,
    }


def test_idle_seconds_is_clamped_for_hand_edited_json():
    """The Pydantic model 422s, but a hand-edited settings file must not
    produce a reaper that fires every poll."""
    from app.core.runtime_settings import merge_runtime_settings

    with _restored_defaults():
        merge_runtime_settings({"vram": {"idle_seconds": 1, "budget_mb": -5}})
        cfg = read_vram_settings()
        assert cfg["idle_seconds"] == MIN_IDLE_SECONDS
        assert cfg["budget_mb"] == 0


def test_endpoint_roundtrip(client):
    with _restored_defaults():
        r = client.put(
            "/api/v1/system/vram",
            json={
                "budget_mb": 4096,
                "idle_release": False,
                "idle_seconds": 300,
                "drop_bank": True,
                "drop_model": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["budget_mb"] == 4096
        assert body["idle_release"] is False
        assert body["idle_seconds"] == 300
        assert body["drop_bank"] is True
        assert "usage" in body
        assert client.get("/api/v1/system/vram").json()["budget_mb"] == 4096


def test_endpoint_rejects_out_of_range_idle(client):
    with _restored_defaults():
        r = client.put(
            "/api/v1/system/vram",
            json={"budget_mb": 8192, "idle_release": True, "idle_seconds": 1},
        )
        assert r.status_code == 422  # ge/le validation, not silent clamping


def test_release_endpoint_reports_and_does_not_error(client):
    r = client.post("/api/v1/system/vram/release", json={})
    assert r.status_code == 200
    assert "released" in r.json()


def test_release_drops_bank_tensors_only_when_asked(client):
    from app.core.cls_state import get_state

    state = get_state()
    state._tensor_cache = {"sentinel": object()}
    client.post("/api/v1/system/vram/release", json={"drop_bank": False})
    # The default release flushes the arena but keeps the bank resident: a
    # dropped bank costs a full int8 re-quantisation on the next score.
    assert state._tensor_cache is not None

    client.post("/api/v1/system/vram/release", json={"drop_bank": True})
    assert state._tensor_cache is None


def test_release_is_refused_while_leased(client):
    """The core safety property: the reaper must not cut into GPU work."""
    from app.core.cls_state import get_state

    state = get_state()
    state._tensor_cache = {"sentinel": object()}
    with state.gpu_lease():
        out = state.release_vram(drop_bank=True)
        assert out["released"] is False
        assert out["reason"] == "busy"
        assert state._tensor_cache is not None
    # Lease released -> the same call now goes through.
    assert state.release_vram(drop_bank=True).get("reason") != "busy"
    assert state._tensor_cache is None


def test_gpu_lease_is_reentrant_across_threads_and_balances():
    from app.core.cls_state import get_state

    state = get_state()
    assert state._gpu_inflight == 0
    started = threading.Event()
    release = threading.Event()

    def worker():
        with state.gpu_lease():
            started.set()
            release.wait(5.0)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started.wait(5.0)
    assert state._gpu_inflight == 1
    with state.gpu_lease():
        assert state._gpu_inflight == 2
    release.set()
    t.join(5.0)
    assert state._gpu_inflight == 0


def test_lease_survives_an_exception():
    from app.core.cls_state import get_state

    state = get_state()
    before = state._gpu_inflight
    with contextlib.suppress(RuntimeError):
        with state.gpu_lease():
            raise RuntimeError("boom")
    assert state._gpu_inflight == before


def test_leased_decorator_keeps_the_signature():
    """FastAPI reads the wrapped signature for dependency injection, so the
    decorator must not turn a route handler into (*args, **kwargs)."""
    import inspect

    from app.core.cls_state import gpu_leased

    def handler(tier: str, name: str = "x") -> str:
        return tier + name

    wrapped = gpu_leased(handler)
    assert list(inspect.signature(wrapped).parameters) == ["tier", "name"]
    assert wrapped("a", name="b") == "ab"


def test_reaper_start_is_idempotent():
    from app.core.vram_reaper import THREAD_NAME, start_vram_reaper, stop_vram_reaper

    try:
        for _ in range(3):
            start_vram_reaper()
        alive = [t for t in threading.enumerate() if t.name == THREAD_NAME and t.is_alive()]
        assert len(alive) == 1
    finally:
        stop_vram_reaper(timeout=2.0)
    time.sleep(0.05)
    assert not [t for t in threading.enumerate() if t.name == THREAD_NAME and t.is_alive()]


def test_reaper_skips_when_already_released():
    """Without this the reaper runs a synchronize + empty_cache every poll,
    forever, on a server nobody is using."""
    from app.core.cls_state import get_state
    from app.core.vram_reaper import _tick

    state = get_state()
    calls: list[dict] = []
    try:
        state.release_vram = lambda **kw: (calls.append(kw), {"released": True})[1]  # type: ignore[method-assign]
        with _restored_defaults():
            save_vram_settings(
                budget_mb=8192, idle_release=True, idle_seconds=MIN_IDLE_SECONDS
            )
            state._gpu_last_active = time.monotonic() - 3600
            # Relative to _gpu_last_active, never a literal: monotonic() is
            # seconds since boot, so on a freshly started machine (any CI
            # runner) `monotonic() - 3600` is NEGATIVE and a 0.0 literal
            # would read as "already released" and skip the tick.
            state._gpu_released_at = state._gpu_last_active - 1
            _tick()
            assert len(calls) == 1
            # _tick did not really run release_vram, so stamp it as it would.
            state._gpu_released_at = time.monotonic()
            _tick()
            assert len(calls) == 1  # skipped: nothing has happened since
    finally:
        del state.release_vram


def test_reaper_does_not_fire_while_leased():
    from app.core.cls_state import get_state
    from app.core.vram_reaper import _tick

    state = get_state()
    calls: list[dict] = []
    try:
        state.release_vram = lambda **kw: (calls.append(kw), {"released": True})[1]  # type: ignore[method-assign]
        with _restored_defaults():
            save_vram_settings(
                budget_mb=8192, idle_release=True, idle_seconds=MIN_IDLE_SECONDS
            )
            state._gpu_last_active = time.monotonic() - 3600
            state._gpu_released_at = state._gpu_last_active - 1
            with state.gpu_lease():
                _tick()
            assert calls == []
    finally:
        del state.release_vram


def _cpu_state_with_bank(monkeypatch, n_rows: int = 64):
    """A ClsStudioState with a random bank and a stubbed (CPU) model."""
    from app.core.cls_state import ClsStudioState

    rng = np.random.default_rng(3)
    st = ClsStudioState()
    st.bank = Bank(normal=rng.normal(size=(n_rows, 16)).astype(np.float16))

    def fake_ensure():
        st._device, st._dtype = "cpu", torch.float32
        return None, "cpu", torch.float32

    monkeypatch.setattr(st, "ensure_model", fake_ensure)
    return st


def test_get_normal_tensor_binds_device_from_the_return_value(monkeypatch):
    """The reaper can null ``_device``/``_dtype`` between the guard and the
    upload. ``.to(None, dtype=None)`` does not raise — it silently returns a
    CPU tensor at the source dtype, and the mismatch only surfaces much later
    inside torch.cdist."""
    from app.core.cls_state import ClsStudioState

    st = ClsStudioState()
    rng = np.random.default_rng(3)
    st.bank = Bank(normal=rng.normal(size=(8, 16)).astype(np.float16))

    def fake_ensure():
        # Returns a valid device/dtype but leaves the FIELDS cleared, exactly
        # as a release landing right after the load would.
        st._device, st._dtype = None, None
        return None, "cpu", torch.float32

    monkeypatch.setattr(st, "ensure_model", fake_ensure)
    t = st.get_normal_tensor()
    assert t.dtype is torch.float32
    assert t.device.type == "cpu"


def test_ensure_model_early_return_is_atomic():
    """The guard must test the same values it returns."""
    from app.core.cls_state import ClsStudioState

    reads: list[int] = []

    class Flaky(ClsStudioState):
        def __getattribute__(self, name):
            if name == "_device":
                reads.append(1)
                # First read looks loaded; any later read looks released.
                return "cpu" if len(reads) == 1 else None
            return super().__getattribute__(name)

    st = Flaky()
    object.__setattr__(st, "_model", object())
    st._dtype = torch.float32
    _model, dev, dt = st.ensure_model()
    assert dev == "cpu"
    assert dt is torch.float32


def test_cdist_chunk_is_bounded_by_the_budget(monkeypatch):
    """The budget, not the card's free VRAM, sets the ceiling — otherwise a
    release makes the NEXT score grab more than the one before it."""
    from app.core import cls_state as cls_state_mod

    st = cls_state_mod.ClsStudioState()
    st._device, st._dtype = "cuda:0", torch.float16

    fake_free = 23 * 1024 ** 3  # a nearly empty 24 GB card

    class _FakeCuda:
        @staticmethod
        def mem_get_info(dev):
            return fake_free, 24 * 1024 ** 3

        @staticmethod
        def memory_reserved(dev):
            return 0

        @staticmethod
        def memory_allocated(dev):
            return 0

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    with _restored_defaults():
        save_vram_settings(budget_mb=0, idle_release=False, idle_seconds=60)
        unlimited = st.cdist_chunk_for(1_000_000)
        save_vram_settings(budget_mb=2048, idle_release=False, idle_seconds=60)
        capped = st.cdist_chunk_for(1_000_000)

    assert capped < unlimited
    # 2 GiB * 0.6 / (1e6 rows * 2 bytes * 3.0 overhead) ~= 214 rows
    assert 150 < capped < 300


def test_cdist_chunk_counts_the_arena_we_already_hold(monkeypatch):
    """mem_get_info excludes our own reserved-but-free blocks. Ignoring them
    would starve every request once the server reached its budget."""
    from app.core import cls_state as cls_state_mod

    st = cls_state_mod.ClsStudioState()
    st._device, st._dtype = "cuda:0", torch.float16

    class _FakeCuda:
        @staticmethod
        def mem_get_info(dev):
            return 0, 24 * 1024 ** 3  # driver has nothing left to give

        @staticmethod
        def memory_reserved(dev):
            return 2 * 1024 ** 3  # ...but we hold 2 GiB

        @staticmethod
        def memory_allocated(dev):
            return 0  # ...and none of it is live

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    with _restored_defaults():
        save_vram_settings(budget_mb=2048, idle_release=False, idle_seconds=60)
        assert st.cdist_chunk_for(1_000_000) > 32  # not starved to the floor
