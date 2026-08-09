# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""System-level settings (network binding, API token visibility).

Endpoints
---------
GET  /system/network — current bind host, LAN IPs, persisted opt-in, security flags
PUT  /system/network — persist {lan_access: bool} to runtime_settings.json
GET  /system/vram    — VRAM budget + idle-release settings and live usage
PUT  /system/vram    — persist them (budget applies to the next score)
POST /system/vram/release — hand the reserved arena back to the driver now

Changes take effect on the next server restart because uvicorn binds at startup
and cannot rebind mid-process.
"""
from __future__ import annotations

import os
import socket
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..core.config import ANNOTATION_BASE_URL, API_TOKEN, CVAT_BASE_URL
from ..core.runtime_compression import (
    read_compression_settings,
    save_compression_settings,
)
from ..core.runtime_settings import read_lan_access_setting, save_lan_access_setting
from ..core.runtime_vram import (
    MAX_IDLE_SECONDS,
    MIN_IDLE_SECONDS,
    read_vram_settings,
    save_vram_settings,
)

router = APIRouter()


def _current_bind_host() -> str:
    """Best-effort detection of the host uvicorn is bound to for this process."""
    raw = (os.getenv("CLS_HOST") or "").strip()
    return raw or "127.0.0.1"


def _list_lan_addresses() -> list[str]:
    """Return non-loopback IPv4 addresses for the local host (best-effort)."""
    addrs: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    # Fallback: open a dummy UDP socket to discover the primary outbound IP.
    if not addrs:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                primary = s.getsockname()[0]
                if primary and not primary.startswith("127."):
                    addrs.add(primary)
        except OSError:
            pass
    return sorted(addrs)


class NetworkSettingsUpdate(BaseModel):
    lan_access: bool


@router.get("/system/network")
def get_network_settings() -> dict[str, Any]:
    lan_access = read_lan_access_setting()
    current_host = _current_bind_host()
    expected_host = "0.0.0.0" if lan_access else "127.0.0.1"
    return {
        "lan_access": lan_access,
        "current_bind_host": current_host,
        "expected_bind_host": expected_host,
        "restart_required": current_host != expected_host,
        "lan_addresses": _list_lan_addresses(),
        "api_token_configured": bool(API_TOKEN),
        "cvat_proxy_configured": bool(CVAT_BASE_URL),
        "annotation_proxy_configured": bool(ANNOTATION_BASE_URL),
    }


@router.put("/system/network")
def update_network_settings(payload: NetworkSettingsUpdate) -> dict[str, Any]:
    save_lan_access_setting(payload.lan_access)
    return get_network_settings()


class CompressionSettingsUpdate(BaseModel):
    int8: bool
    ivf: bool
    ivf_nprobe: int = Field(8, ge=1, le=64)


@router.get("/system/compression")
def get_compression_settings() -> dict[str, Any]:
    """Bank-compression settings (int8 quantisation / IVF routing)."""
    return read_compression_settings()


@router.put("/system/compression")
def update_compression_settings(payload: CompressionSettingsUpdate) -> dict[str, Any]:
    """Persist compression settings; takes effect on the next score/eval.

    No restart needed: dropping the cached bank tensors makes the next
    scoring call rebuild them under the new settings. Eval caches and saved
    verdict recipes include these settings in their fingerprints, so stale
    numbers invalidate on their own.
    """
    out = save_compression_settings(payload.int8, payload.ivf, payload.ivf_nprobe)
    from ..core.cls_state import get_state

    state = get_state()
    with state.lock:
        state.mark_dirty()
    return out


class VramSettingsUpdate(BaseModel):
    budget_mb: int = Field(8192, ge=0, le=1_048_576)
    idle_release: bool = True
    idle_seconds: int = Field(60, ge=MIN_IDLE_SECONDS, le=MAX_IDLE_SECONDS)
    drop_bank: bool = False
    drop_model: bool = False


class VramReleaseRequest(BaseModel):
    drop_bank: bool | None = None
    drop_model: bool | None = None


@router.get("/system/vram")
def get_vram_settings() -> dict[str, Any]:
    """VRAM budget + idle-release settings, with this process's live usage."""
    from ..core.cls_state import get_state

    return {**read_vram_settings(), "usage": get_state().vram_stats()}


@router.put("/system/vram")
def update_vram_settings(payload: VramSettingsUpdate) -> dict[str, Any]:
    """Persist VRAM settings. The budget applies to the next score, no restart.

    Lowering the budget below what the process already holds does not shrink
    it on its own -- it only stops the next request from growing further.
    Follow with ``POST /system/vram/release`` to hand back what is already
    reserved.
    """
    out = save_vram_settings(
        payload.budget_mb,
        payload.idle_release,
        payload.idle_seconds,
        payload.drop_bank,
        payload.drop_model,
    )
    from ..core.cls_state import get_state

    return {**out, "usage": get_state().vram_stats()}


@router.post("/system/vram/release")
def release_vram_now(payload: VramReleaseRequest | None = None) -> dict[str, Any]:
    """Release now, without waiting for the idle timer.

    Returns 200 with ``{"released": false, "reason": "busy"}`` while work is
    in flight rather than an error status: forcing would not free anything
    anyway (``empty_cache`` cannot reclaim a referenced block) and an error
    toast is the wrong answer to "the server is working right now".
    """
    from ..core.cls_state import get_state

    cfg = read_vram_settings()
    req = payload or VramReleaseRequest()
    return get_state().release_vram(
        drop_bank=cfg["drop_bank"] if req.drop_bank is None else req.drop_bank,
        drop_model=cfg["drop_model"] if req.drop_model is None else req.drop_model,
    )
