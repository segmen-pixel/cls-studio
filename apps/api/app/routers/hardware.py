# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..core.config import APP_BUILD_DATE, APP_VERSION
from ..core.torch_device import set_configured_torch_device, torch_device_state

router = APIRouter()


@router.get("/health")
def health_check():
    """Health endpoint with version, disk, and optional RAM info."""
    result: dict[str, Any] = {
        "status": "ok",
        "version": APP_VERSION,
        "build_date": APP_BUILD_DATE,
    }

    # Disk usage (stdlib)
    try:
        usage = shutil.disk_usage(Path.cwd())
        result["disk"] = {
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "used_pct": round((usage.used / usage.total) * 100, 1),
        }
    except Exception:
        result["disk"] = None

    # RAM (optional psutil)
    try:
        import psutil
        vm = psutil.virtual_memory()
        result["ram"] = {
            "total_gb": round(vm.total / (1024 ** 3), 1),
            "available_gb": round(vm.available / (1024 ** 3), 1),
            "used_pct": round(vm.percent, 1),
        }
    except Exception:
        result["ram"] = None

    # GPU VRAM (optional torch)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)
            result["gpu"] = {
                "name": props.name,
                "vram_total_mb": round(props.total_memory / (1024 ** 2)),
                "vram_allocated_mb": round(allocated / (1024 ** 2)),
                # ``reserved`` is what the caching allocator holds from the
                # driver — the number that actually moves when the arena is
                # released. ``allocated`` only counts live tensors and barely
                # budges. Read against the CONFIGURED device, not device 0.
                "vram_reserved_mb": _state_vram().get("reserved_mb"),
            }
        else:
            result["gpu"] = None
    except Exception:
        result["gpu"] = None

    return result


def _state_vram() -> dict[str, Any]:
    """This process's VRAM usage, or ``{"available": False}``. Never raises."""
    try:
        from ..core.cls_state import get_state

        return get_state().vram_stats()
    except Exception:  # noqa: BLE001 - a stats read must never 500 a route
        return {"available": False}


@router.get("/hardware/gpu/stats")
def gpu_stats():
    """GPU utilization, memory, temperature, and clocks via nvidia-smi. Multi-GPU."""
    import subprocess
    _QUERY = (
        "name,utilization.gpu,utilization.memory,temperature.gpu,"
        "memory.used,memory.total,fan.speed,power.draw,power.limit,"
        "clocks.current.graphics,clocks.current.memory"
    )
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return {"available": False, "error": r.stderr.strip(), "gpus": []}

        def _int(s: str) -> int | None:
            try:
                return int(s)
            except (ValueError, TypeError):
                return None

        def _float(s: str) -> float | None:
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "name": parts[0],
                "gpu_util": _int(parts[1]) or 0,
                "mem_util": _int(parts[2]) or 0,
                "temp_c": _int(parts[3]) or 0,
                "vram_used_mb": _int(parts[4]) or 0,
                "vram_total_mb": _int(parts[5]) or 0,
                "fan_pct": _int(parts[6]) if len(parts) > 6 else None,
                "power_w": _float(parts[7]) if len(parts) > 7 else None,
                "power_limit_w": _float(parts[8]) if len(parts) > 8 else None,
                "clock_graphics_mhz": _int(parts[9]) if len(parts) > 9 else None,
                "clock_memory_mhz": _int(parts[10]) if len(parts) > 10 else None,
            })
        # Backward compat: return first GPU as top-level fields + gpus array.
        # ``vram_used_mb`` here is MACHINE-WIDE — nvidia-smi cannot attribute
        # memory to a process under WDDM, so a training run on the same card
        # shows up in it. ``process`` is this server's own share; without it
        # the monitor invites the operator to blame cls-studio for a
        # neighbour's memory.
        first = gpus[0] if gpus else {}
        return {
            "available": bool(gpus),
            **first,
            "gpus": gpus,
            "process": _state_vram(),
        }
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    except Exception as e:
        return {"available": False, "error": str(e), "gpus": []}


@router.get("/hardware/torch/devices")
def get_torch_devices():
    return torch_device_state()


@router.put("/hardware/torch/device")
def put_torch_device(payload: dict[str, Any]):
    requested = payload.get("device")
    if not isinstance(requested, str) or not requested.strip():
        raise HTTPException(status_code=400, detail="device is required")
    selected = set_configured_torch_device(requested)
    # Drop the loaded backbone + cached tensors so the change takes effect
    # now — ensure_model() otherwise keeps the old device until restart.
    from ..core.cls_state import get_state
    get_state().reset_model()
    state = torch_device_state()
    state["selected_device"] = selected
    return state

@router.get("/hardware/coreml")
def coreml_capability() -> dict[str, Any]:
    """Can this machine produce a Core ML encoder, and if not, why not.

    Two flags rather than one. Converting is not the same as being able to run
    the result, and a conversion nobody can run is a conversion nobody can
    check against the bank it has to match -- so the UI disables the control
    rather than offering it and failing afterwards. On the Windows box this
    usually runs on, unavailable is the normal answer, not an error.
    """
    from clscore.coreml_export import coreml_availability

    return coreml_availability()


@router.get("/hardware/openvino")
def openvino_capability() -> dict[str, Any]:
    """Can this machine produce an OpenVINO IR.

    Same shape as /hardware/coreml so the UI reads one structure whichever
    backend it asks about -- but unlike Core ML the two flags move together,
    because OpenVINO converts and runs wherever it installs. Which is also why
    an IR gets checked against the bank on the machine that built both.
    """
    from clscore.openvino_export import openvino_availability

    return openvino_availability()
