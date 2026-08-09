# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

from app.core import state as _state
from app.core import torch_device as td


def setup_function() -> None:
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        _state.ACTIVE_TORCH_JOBS.clear()
    td._torch_devices_cache.clear()
    # Clean up file-based GPU locks from previous tests
    import shutil
    if td._GPU_LOCK_ROOT.exists():
        shutil.rmtree(td._GPU_LOCK_ROOT, ignore_errors=True)


def test_auto_prefers_free_gpu(monkeypatch):
    monkeypatch.setattr(
        td,
        "list_torch_devices",
        lambda: [
            {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True},
            {"id": "cuda:0", "label": "GPU0", "kind": "cuda", "memory_mb": 8000, "available": True},
            {"id": "cuda:1", "label": "GPU1", "kind": "cuda", "memory_mb": 8000, "available": True},
        ],
    )
    monkeypatch.setattr(td, "_query_nvidia_smi", lambda: {})

    claimed = td.claim_torch_device("cuda:0", owner_kind="training", owner_id="run-a")
    assert claimed == "cuda:0"
    assert td.resolve_torch_device_or_cpu("auto") == "cuda:1"

    td.release_torch_device("cuda:0", owner_id="run-a")
    assert td.resolve_torch_device_or_cpu("auto") == "cuda:0"


def test_torch_device_state_marks_busy_device(monkeypatch):
    monkeypatch.setattr(
        td,
        "list_torch_devices",
        lambda: [
            {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True},
            {"id": "cuda:0", "label": "GPU0", "kind": "cuda", "memory_mb": 8000, "available": True},
            {"id": "cuda:1", "label": "GPU1", "kind": "cuda", "memory_mb": 8000, "available": True},
        ],
    )
    monkeypatch.setattr(td, "_query_nvidia_smi", lambda: {})
    monkeypatch.setattr(td, "current_configured_torch_device", lambda: "cuda:1")

    td.claim_torch_device("cuda:0", owner_kind="training", owner_id="run-a", project_id="p1")
    state = td.torch_device_state()

    busy = {item["id"]: item for item in state["devices"]}
    assert busy["cuda:0"]["busy"] is True
    assert busy["cuda:0"]["busy_owner_kind"] == "training"
    assert busy["cuda:1"]["selected"] is True
