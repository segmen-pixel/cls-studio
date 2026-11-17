# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from . import state as _state
from .cache_utils import ThreadSafeLRUCache
from .config import PROJECTS_DIR
from .paths import write_json
from .runtime_settings import merge_runtime_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-based GPU lock infrastructure
# ---------------------------------------------------------------------------
_GPU_LOCK_ROOT = PROJECTS_DIR / ".gpu_locks"
_GPU_LOCK_STALE_SEC = 90  # heartbeat older than this → stale


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_lock_dir(device_id: str) -> Path:
    return _GPU_LOCK_ROOT / device_id.replace(":", "_")


def _device_lock_meta_path(device_id: str) -> Path:
    return _device_lock_dir(device_id) / "owner.json"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows os.kill(pid, 0) is unreliable (SystemError on access-denied).
        # Use ctypes OpenProcess + GetExitCodeProcess instead.
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_device_lock(device_id: str) -> dict[str, Any] | None:
    path = _device_lock_meta_path(device_id)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _lock_is_stale(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return True
    # Check worker PID first (subprocess doing actual training)
    worker_pid = int(meta.get("worker_pid") or 0)
    api_pid = int(meta.get("pid") or 0)
    if worker_pid > 0 and _pid_is_alive(worker_pid):
        return False
    if api_pid > 0 and _pid_is_alive(api_pid):
        # API process alive but no worker yet → check heartbeat freshness
        try:
            heartbeat = datetime.fromisoformat(str(meta.get("heartbeat_at")))
            age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        except Exception:
            age = _GPU_LOCK_STALE_SEC + 1
        return age > _GPU_LOCK_STALE_SEC
    # Both PIDs dead → stale
    return True


def _try_claim_device_lock(
    device_id: str, *, owner_kind: str, owner_id: str, project_id: str | None,
) -> dict[str, str] | None:
    """Try to atomically create a lock dir. Returns meta dict on success, None if busy."""
    _GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock_dir = _device_lock_dir(device_id)
    try:
        lock_dir.mkdir()  # atomic on both Windows and POSIX
    except FileExistsError:
        meta = _read_device_lock(device_id)
        # Allow re-claim by same owner
        if meta and meta.get("owner_id") == owner_id:
            return meta
        if not _lock_is_stale(meta):
            return None
        logger.warning(
            "Reclaiming stale GPU lock on %s (prev owner=%s, pid=%s)",
            device_id, meta.get("owner_id") if meta else "?",
            meta.get("worker_pid") or meta.get("pid") if meta else "?",
        )
        shutil.rmtree(lock_dir, ignore_errors=True)
        try:
            lock_dir.mkdir()
        except FileExistsError:
            return None  # race with another claimer

    meta = {
        "device_id": device_id,
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "project_id": project_id or "",
        "pid": str(os.getpid()),
        "worker_pid": "",
        "hostname": socket.gethostname(),
        "claimed_at": _utcnow_iso(),
        "heartbeat_at": _utcnow_iso(),
    }
    write_json(lock_dir / "owner.json", meta)
    return meta


def _release_device_lock(device_id: str, *, owner_id: str | None = None) -> None:
    """Remove the lock dir for a device. If owner_id is given, only remove if it matches."""
    lock_dir = _device_lock_dir(device_id)
    if not lock_dir.exists():
        return
    if owner_id is not None:
        meta = _read_device_lock(device_id)
        if meta and meta.get("owner_id") != owner_id:
            return
    shutil.rmtree(lock_dir, ignore_errors=True)


def touch_torch_device_claim(
    device_id: str, *, owner_id: str, worker_pid: int | None = None,
) -> None:
    """Update heartbeat (and optionally worker PID) on an existing lock."""
    meta = _read_device_lock(device_id)
    if not meta or meta.get("owner_id") != owner_id:
        return
    meta["heartbeat_at"] = _utcnow_iso()
    if worker_pid is not None:
        meta["worker_pid"] = str(worker_pid)
    try:
        write_json(_device_lock_meta_path(device_id), meta)
    except OSError:
        pass


def _recover_locks_from_disk() -> dict[str, dict[str, str]]:
    """Read all non-stale GPU locks from disk. Used to restore state after API restart."""
    result: dict[str, dict[str, str]] = {}
    if not _GPU_LOCK_ROOT.exists():
        return result
    for lock_dir in _GPU_LOCK_ROOT.iterdir():
        if not lock_dir.is_dir():
            continue
        device_id = lock_dir.name.replace("_", ":", 1)
        meta = _read_device_lock(device_id)
        if meta and not _lock_is_stale(meta):
            result[device_id] = meta
        elif meta:
            # Clean up stale lock
            logger.info("Cleaning stale GPU lock: %s (owner=%s)", device_id, meta.get("owner_id"))
            shutil.rmtree(lock_dir, ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# nvidia-smi metrics for GPU scoring (C: multi-GPU scheduler)
# ---------------------------------------------------------------------------
_GPU_METRICS_CACHE = ThreadSafeLRUCache(maxsize=1, ttl=3.0)


def _query_nvidia_smi() -> dict[str, dict[str, int]]:
    """Query nvidia-smi for free memory, utilization, and temperature per GPU."""
    cached = _GPU_METRICS_CACHE.get("metrics")
    if cached is not None:
        return cached
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return {}
    metrics: dict[str, dict[str, int]] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx_s, free_s, util_s, temp_s = parts
            metrics[f"cuda:{int(idx_s)}"] = {
                "free_mb": int(free_s),
                "utilization": int(util_s),
                "temperature_c": int(temp_s),
            }
        except (ValueError, IndexError):
            continue
    _GPU_METRICS_CACHE.put("metrics", metrics)
    return metrics


def _device_score(device_info: dict[str, Any], busy: set[str], smi: dict[str, dict[str, int]]) -> float:
    """Score a GPU device: higher = better candidate for scheduling."""
    device_id = str(device_info["id"])
    if device_id in busy:
        return -1e12
    # Base score from total memory
    total_mb = float(device_info.get("memory_mb") or 0)
    # Prefer nvidia-smi free_mb if available
    smi_info = smi.get(device_id, {})
    free_mb = float(smi_info.get("free_mb", total_mb))
    util = float(smi_info.get("utilization", 0))
    temp = float(smi_info.get("temperature_c", 0))
    # Score: free memory is primary, penalize high utilization and temperature
    return free_mb - util * 80.0 - max(0.0, temp - 70.0) * 64.0


def normalize_torch_device_id(device_id: str) -> str:
    value = (device_id or "").strip().lower()
    if value in {"cpu", "mps", "auto"}:
        return value
    if value == "cuda":
        return "cuda:0"
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"
    raise ValueError("device must be one of cpu, mps, cuda, cuda:<index>")


_torch_devices_cache = ThreadSafeLRUCache(maxsize=1, ttl=10.0)


def list_torch_devices() -> list[dict[str, Any]]:
    cached = _torch_devices_cache.get("devices")
    if cached is not None:
        return cached

    devices: list[dict[str, Any]] = [
        {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True}
    ]
    try:
        import torch  # type: ignore
    except ImportError:
        _torch_devices_cache.put("devices", devices)
        return devices
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            devices.append({"id": "mps", "label": "Apple MPS", "kind": "mps", "available": True})
    except (RuntimeError, AttributeError):
        pass
    try:
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            for idx in range(count):
                label = f"CUDA:{idx}"
                memory_mb = None
                allocated_mb = None
                reserved_mb = None
                try:
                    props = torch.cuda.get_device_properties(idx)
                    memory_mb = int(props.total_memory // (1024 * 1024))
                    label = f"CUDA:{idx} {props.name} ({memory_mb}MB)"
                except (RuntimeError, AttributeError):
                    pass
                try:
                    allocated_mb = int(torch.cuda.memory_allocated(idx) // (1024 * 1024))
                    reserved_mb = int(torch.cuda.memory_reserved(idx) // (1024 * 1024))
                except (RuntimeError, AttributeError):
                    pass
                devices.append(
                    {
                        "id": f"cuda:{idx}",
                        "label": label,
                        "kind": "cuda",
                        "index": idx,
                        "memory_mb": memory_mb,
                        "allocated_mb": allocated_mb,
                        "reserved_mb": reserved_mb,
                        "available": True,
                    }
                )
    except (RuntimeError, AttributeError):
        pass
    _torch_devices_cache.put("devices", devices)
    return devices


def _is_exclusive_torch_device(device_id: str) -> bool:
    return device_id.startswith("cuda:") or device_id == "mps"


def active_torch_jobs() -> dict[str, dict[str, str]]:
    """Return active GPU jobs. Merges in-memory state with file-based locks."""
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        result = {device_id: dict(meta) for device_id, meta in _state.ACTIVE_TORCH_JOBS.items()}
    # Also check file locks for jobs surviving API restart
    disk_locks = _recover_locks_from_disk()
    for device_id, meta in disk_locks.items():
        if device_id not in result:
            result[device_id] = meta
    return result


def active_torch_job_for(device_id: str) -> dict[str, str] | None:
    resolved = resolve_torch_device_or_cpu(device_id)
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        meta = _state.ACTIVE_TORCH_JOBS.get(resolved)
        if meta is not None:
            return dict(meta)
    # Fallback: check file lock
    disk_meta = _read_device_lock(resolved)
    if disk_meta and not _lock_is_stale(disk_meta):
        return disk_meta
    return None


def claim_torch_device(
    requested_device: str,
    *,
    owner_kind: str,
    owner_id: str,
    project_id: str | None = None,
    wait: bool = False,
) -> str | None:
    resolved = resolve_torch_device_or_cpu(requested_device)
    if not _is_exclusive_torch_device(resolved):
        return resolved
    while True:
        meta = _try_claim_device_lock(
            resolved, owner_kind=owner_kind, owner_id=owner_id, project_id=project_id,
        )
        if meta is not None:
            with _state.ACTIVE_TORCH_JOBS_LOCK:
                _state.ACTIVE_TORCH_JOBS[resolved] = meta
            return resolved
        if not wait:
            return None
        time.sleep(0.2)


def release_torch_device(device_id: str, *, owner_id: str | None = None) -> None:
    resolved = resolve_torch_device_or_cpu(device_id)
    if not _is_exclusive_torch_device(resolved):
        return
    _release_device_lock(resolved, owner_id=owner_id)
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        current = _state.ACTIVE_TORCH_JOBS.get(resolved)
        if current is None:
            return
        if owner_id is not None and current.get("owner_id") != owner_id:
            return
        _state.ACTIVE_TORCH_JOBS.pop(resolved, None)


@contextmanager
def acquired_torch_device(
    requested_device: str,
    *,
    owner_kind: str,
    owner_id: str,
    project_id: str | None = None,
):
    resolved = claim_torch_device(
        requested_device,
        owner_kind=owner_kind,
        owner_id=owner_id,
        project_id=project_id,
        wait=False,
    )
    if resolved is None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "torch device busy",
                "device": resolve_torch_device_or_cpu(requested_device),
            },
        )
    try:
        yield resolved
    finally:
        release_torch_device(resolved, owner_id=owner_id)


def resolve_torch_device_or_cpu(requested_device: str) -> str:
    value = (requested_device or "").strip().lower()
    if value == "auto":
        devices = list_torch_devices()
        cuda_devices = [d for d in devices if str(d["id"]).startswith("cuda")]
        if cuda_devices:
            busy = set(active_torch_jobs().keys())
            smi = _query_nvidia_smi()
            # Merge nvidia-smi metrics into device info for scoring
            for d in cuda_devices:
                smi_info = smi.get(str(d["id"]), {})
                if smi_info:
                    d["free_mb"] = smi_info.get("free_mb")
                    d["utilization"] = smi_info.get("utilization")
                    d["temperature_c"] = smi_info.get("temperature_c")
            ranked = sorted(cuda_devices, key=lambda d: _device_score(d, busy, smi), reverse=True)
            if ranked and _device_score(ranked[0], busy, smi) > -1e11:
                return str(ranked[0]["id"])
            # All busy — return the first cuda device (will queue as reserved)
            if ranked:
                return str(ranked[0]["id"])
        available = {item["id"] for item in devices if item.get("available")}
        if "mps" in available:
            return "mps"
        return "cpu"
    try:
        normalized = normalize_torch_device_id(value)
    except ValueError:
        return "cpu"
    available = {item["id"] for item in list_torch_devices() if item.get("available")}
    if normalized in available:
        return normalized
    if normalized.startswith("cuda") and "cuda:0" in available:
        return "cuda:0"
    return "cpu"


def current_configured_torch_device() -> str:
    with _state.SETTINGS_LOCK:
        return _state.SELECTED_TORCH_DEVICE


def set_configured_torch_device(device_id: str) -> str:
    normalized = normalize_torch_device_id(device_id)
    if normalized != "auto":
        available = {item["id"] for item in list_torch_devices() if item.get("available")}
        if normalized not in available:
            raise HTTPException(
                status_code=400,
                detail=f"device '{normalized}' is not available. available={sorted(available)}",
            )
    with _state.SETTINGS_LOCK:
        _state.SELECTED_TORCH_DEVICE = normalized
        merge_runtime_settings({"torch_device": normalized})
    return normalized


def torch_device_state() -> dict[str, Any]:
    configured = current_configured_torch_device()
    resolved = resolve_torch_device_or_cpu(configured)
    devices = list_torch_devices()
    active = active_torch_jobs()
    smi = _query_nvidia_smi()
    out_devices = []
    for item in devices:
        view = dict(item)
        view["selected"] = bool(item.get("id") == resolved)
        view["busy"] = bool(item.get("id") in active)
        if item.get("id") in active:
            view["busy_owner_kind"] = active[item["id"]].get("owner_kind")
            view["busy_owner_id"] = active[item["id"]].get("owner_id")
        # Merge nvidia-smi metrics
        smi_info = smi.get(str(item.get("id")), {})
        if smi_info:
            view["free_mb"] = smi_info.get("free_mb")
            view["utilization"] = smi_info.get("utilization")
            view["temperature_c"] = smi_info.get("temperature_c")
        out_devices.append(view)
    return {
        "configured_device": configured,
        "selected_device": resolved,
        "devices": out_devices,
    }

