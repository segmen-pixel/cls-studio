# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Background thread that hands idle VRAM back to the driver.

Idle cannot be answered from the request path -- idle *means* no requests --
so this is a daemon thread rather than a dependency or a middleware hook. It
does nothing on a server that has never touched CUDA: it reads ``torch`` out
of ``sys.modules`` instead of importing it, so a poll can never be the thing
that creates a CUDA context on a machine that had none.

Started from ``_deferred_post_startup`` (after ``ready``, so it can never sit
in front of the UI build) and stopped from the app's shutdown hook.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from .runtime_vram import POLL_SECONDS, read_vram_settings

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_thread_lock = threading.Lock()
_stop = threading.Event()

THREAD_NAME = "vram-reaper"


def _tick() -> None:
    """One poll. Never raises."""
    if "torch" not in sys.modules:
        return  # nothing has touched CUDA in this process yet
    cfg = read_vram_settings()
    if not cfg["idle_release"]:
        return
    from .cls_state import get_state

    state = get_state()
    if state._gpu_inflight:
        return
    last_active = state._gpu_last_active
    if time.monotonic() - last_active < cfg["idle_seconds"]:
        return
    # Already released since the last real work: another flush would be a
    # synchronize + empty_cache that frees nothing, every poll, forever.
    if state._gpu_released_at >= last_active:
        return
    state.release_vram(
        drop_bank=bool(cfg["drop_bank"]), drop_model=bool(cfg["drop_model"])
    )


def _loop() -> None:
    while not _stop.wait(POLL_SECONDS):
        try:
            _tick()
        except Exception as exc:  # noqa: BLE001 - a reaper must never die
            logger.warning("vram reaper tick failed: %s", exc, exc_info=True)


def start_vram_reaper() -> bool:
    """Start the reaper once per process. Returns True if this call started it.

    Idempotent because the app object is a module-level singleton and every
    TestClient entry re-runs startup -- without the guard a test session
    accumulates one live reaper per client.
    """
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop.clear()
        _thread = threading.Thread(target=_loop, name=THREAD_NAME, daemon=True)
        _thread.start()
        logger.info("vram reaper started (poll %.0fs)", POLL_SECONDS)
        return True


def stop_vram_reaper(timeout: float = 2.0) -> None:
    """Signal the reaper and wait briefly. The join is advisory, not a promise.

    A release already in flight is not interruptible; the thread is a daemon
    so a slow flush can never hold up interpreter exit.
    """
    global _thread
    with _thread_lock:
        t = _thread
        _stop.set()
    if t is not None and t.is_alive():
        t.join(timeout)
        if t.is_alive():
            logger.info("vram reaper still finishing a release at shutdown")
    with _thread_lock:
        if _thread is t:
            _thread = None
