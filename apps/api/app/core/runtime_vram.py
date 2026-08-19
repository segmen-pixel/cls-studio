# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""VRAM runtime settings: the process budget and the idle release.

Persisted under the ``vram`` key of ``runtime_settings.json`` (merge-safe,
same file as the torch-device choice and the ``bank_compression`` block).

Two knobs for one problem. ``budget_mb`` caps how much VRAM the scoring path
may size itself against: without it the distance-matrix chunk is grown to a
fraction of *whatever is free on the card*, which on an idle 24 GB GPU means
a single score reserves most of it and the caching allocator never gives it
back. ``idle_release`` then hands the reserved-but-unused arena back to the
driver once the server has gone quiet, so a neighbouring job (a training run
on the same GPU) can have it.

The release deliberately keeps the backbone and the device-resident bank:
they are small next to the scoring arena, and dropping them buys ~1 GB at
the price of a full int8 re-quantisation of every bank row on the next
score. ``drop_bank`` / ``drop_model`` are there for operators who need the
card empty and will accept that cost.
"""

from __future__ import annotations

from typing import Any

from .runtime_settings import merge_runtime_settings, read_runtime_settings

SETTINGS_KEY = "vram"
DEFAULTS: dict[str, Any] = {
    # 0 = unlimited (size against free VRAM, the pre-budget behaviour).
    "budget_mb": 8192,
    "idle_release": True,
    "idle_seconds": 60,
    "drop_bank": False,
    "drop_model": False,
}

# The reaper wakes at this cadence, so it also bounds how late a release can
# be. Kept below the smallest allowed idle_seconds so the timeout, not the
# poll, is what the operator observes.
POLL_SECONDS = 10.0
MIN_IDLE_SECONDS = 15
MAX_IDLE_SECONDS = 86400


def read_vram_settings() -> dict[str, Any]:
    """Current settings with defaults filled in; malformed values are ignored."""
    raw = read_runtime_settings().get(SETTINGS_KEY)
    out = dict(DEFAULTS)
    if isinstance(raw, dict):
        for key in ("idle_release", "drop_bank", "drop_model"):
            if isinstance(raw.get(key), bool):
                out[key] = raw[key]
        try:
            out["budget_mb"] = max(0, int(raw.get("budget_mb", out["budget_mb"])))
        except (TypeError, ValueError):
            pass
        try:
            out["idle_seconds"] = min(
                MAX_IDLE_SECONDS,
                max(MIN_IDLE_SECONDS, int(raw.get("idle_seconds", out["idle_seconds"]))),
            )
        except (TypeError, ValueError):
            pass
    return out


def save_vram_settings(
    budget_mb: int,
    idle_release: bool,
    idle_seconds: int,
    drop_bank: bool = False,
    drop_model: bool = False,
) -> dict[str, Any]:
    """Persist and return the resolved settings."""
    merge_runtime_settings(
        {
            SETTINGS_KEY: {
                "budget_mb": max(0, int(budget_mb)),
                "idle_release": bool(idle_release),
                "idle_seconds": min(
                    MAX_IDLE_SECONDS, max(MIN_IDLE_SECONDS, int(idle_seconds))
                ),
                "drop_bank": bool(drop_bank),
                "drop_model": bool(drop_model),
            }
        }
    )
    return read_vram_settings()
