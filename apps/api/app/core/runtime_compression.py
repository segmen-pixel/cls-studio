# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Bank-compression runtime settings (int8 quantisation + IVF routing).

Persisted under the ``bank_compression`` key of ``runtime_settings.json``
(merge-safe, same file as the torch-device choice). Both transforms default
to ON: the 2026-07 compression sweep found int8 verdict-neutral and IVF
nprobe=8 AUROC-neutral-or-better on every internal project, and running the
server compressed keeps its verdict thresholds transferable to compressed
edge exports. The eval-cache / verdict-recipe fingerprints include these
settings, so toggling them invalidates stale numbers automatically.
"""

from __future__ import annotations

from typing import Any

from .runtime_settings import merge_runtime_settings, read_runtime_settings

SETTINGS_KEY = "bank_compression"
DEFAULTS: dict[str, Any] = {"int8": True, "ivf": True, "ivf_nprobe": 8}


def read_compression_settings() -> dict[str, Any]:
    """Current settings with defaults filled in; malformed values are ignored."""
    raw = read_runtime_settings().get(SETTINGS_KEY)
    out = dict(DEFAULTS)
    if isinstance(raw, dict):
        for key in ("int8", "ivf"):
            if isinstance(raw.get(key), bool):
                out[key] = raw[key]
        try:
            out["ivf_nprobe"] = min(64, max(1, int(raw.get("ivf_nprobe", out["ivf_nprobe"]))))
        except (TypeError, ValueError):
            pass
    return out


def save_compression_settings(int8: bool, ivf: bool, ivf_nprobe: int) -> dict[str, Any]:
    """Persist and return the resolved settings."""
    merge_runtime_settings(
        {
            SETTINGS_KEY: {
                "int8": bool(int8),
                "ivf": bool(ivf),
                "ivf_nprobe": min(64, max(1, int(ivf_nprobe))),
            }
        }
    )
    return read_compression_settings()
