# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Persisted runtime settings (``runtime_settings.json``) accessors.

One JSON file holds every user-adjustable runtime knob: the torch-device
choice, the LAN-access opt-in, and the ``bank_compression`` block managed by
:mod:`.runtime_compression`. All writers go through
:func:`merge_runtime_settings` so concurrent features never clobber each
other's keys. Extracted from ``torch_device.py`` so device selection/locking
and settings persistence evolve independently.
"""

from __future__ import annotations

import json
from typing import Any

from .config import RUNTIME_SETTINGS_PATH
from .paths import write_json


def read_runtime_settings() -> dict[str, Any]:
    if not RUNTIME_SETTINGS_PATH.exists():
        return {}
    try:
        raw = json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_runtime_settings(payload: dict[str, Any]) -> None:
    RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME_SETTINGS_PATH, payload)


def merge_runtime_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into runtime_settings.json without clobbering other keys."""
    current = read_runtime_settings()
    current.update(updates)
    save_runtime_settings(current)
    return current


def read_lan_access_setting() -> bool:
    """Return True when the user opted into binding the API on all interfaces."""
    return bool(read_runtime_settings().get("lan_access", False))


def save_lan_access_setting(lan_access: bool) -> bool:
    """Persist the LAN access opt-in to runtime_settings.json (merge-safe)."""
    merge_runtime_settings({"lan_access": bool(lan_access)})
    return bool(lan_access)
