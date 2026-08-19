# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Deferred router registration (called from the background startup)."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .core.config import API_V1_PREFIX

logger = logging.getLogger("api")


def register_routers(app: FastAPI) -> None:
    """Import and register all routers.

    Each router is imported individually so that a missing optional dependency
    (e.g. cv2, torch) only disables the affected router instead of
    crashing the entire startup.
    """
    # Root-level routes (no versioned prefix): health, version, startup, UI redirect
    _root_router_modules = [
        ".routers.root",
    ]
    # API routes (registered under /api/v1 prefix): the platform routers
    # (project CRUD, hardware/health, system settings) followed by the
    # classification routers.
    _api_router_modules = [
        ".routers.hardware",
        ".routers.projects",
        ".routers.system",
        # --- classification routers ---
        ".routers.bank",
        ".routers.store",
        ".routers.score",
        ".routers.images",
        ".routers.staging",
        ".routers.captures",
        ".routers.inspections",
    ]
    # /v2 streaming API — no routers ship in this repository, so the list is
    # empty; scoring goes through POST /api/v1/score. The list and its loop are
    # kept because the optional CLS_API_TOKEN middleware (above) already
    # guards `/v2/` and `/ws/v2/`, so a streaming router can be added here
    # without revisiting the auth path.
    _v2_router_modules: list[str] = []
    # SECURITY NOTE: importlib is used here only with the hardcoded router
    # module names listed above.  No user input influences module resolution.
    # This deferred-import pattern avoids loading heavy dependencies (torch,
    # cv2, sklearn) at startup, cutting launch time from 15s+ to ~1s.
    import importlib
    for mod_name in _root_router_modules:
        try:
            mod = importlib.import_module(mod_name, package=__package__ or "app")
            app.include_router(mod.router)
        except Exception as exc:
            logger.warning("Skipping router %s: %s", mod_name, exc, exc_info=True)
    for mod_name in _api_router_modules:
        try:
            mod = importlib.import_module(mod_name, package=__package__ or "app")
            # Registered only under /api/v1 so the optional CLS_API_TOKEN
            # middleware (above) cannot be bypassed by hitting the prefix-less
            # path. Clients must use /api/v1/<route>.
            app.include_router(mod.router, prefix=API_V1_PREFIX)
        except Exception as exc:
            logger.warning("Skipping router %s: %s", mod_name, exc, exc_info=True)
    for mod_name in _v2_router_modules:
        try:
            mod = importlib.import_module(mod_name, package=__package__ or "app")
            # No prefix: routers define `/v2/...` and `/ws/v2/...` paths
            # themselves. The CLS_API_TOKEN middleware guards these explicitly
            # (see `_is_guarded_path` above).
            app.include_router(mod.router)
        except Exception as exc:
            logger.warning("Skipping router %s: %s", mod_name, exc, exc_info=True)
