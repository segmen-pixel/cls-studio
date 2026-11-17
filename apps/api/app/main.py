# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import faulthandler
import logging
import mimetypes
import os
import sys
import threading

faulthandler.enable()

# ---------------------------------------------------------------------------
# Reject launch from system Python (must use venv or bundled Python)
# ---------------------------------------------------------------------------
if not (getattr(sys, "real_prefix", None) or sys.prefix != sys.base_prefix):
    # Not inside a virtualenv — allow bundled installer Python (cls-studio dir)
    _exe = os.path.realpath(sys.executable).lower()
    _is_bundled = "cls-studio" in _exe
    if not _is_bundled and ("appdata" in _exe or "windowsapps" in _exe):
        print(
            "\n  [FATAL] System Python detected.\n"
            "  Use .venv-windows/Scripts/python.exe or the bundled launcher.\n"
            f"  sys.executable = {sys.executable}\n",
            file=sys.stderr,
        )
        sys.exit(1)

from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (secrets, config) before anything else
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass  # python-dotenv not installed --use OS env vars directly

# ---------------------------------------------------------------------------
# Logging (must come first)
# ---------------------------------------------------------------------------
from .core.logging_config import configure_logging  # noqa: E402

# Packaged builds: logs go to %LOCALAPPDATA%/cls-studio/logs (survives uninstall).
# Dev builds: logs go to <repo>/logs as before.
if (ROOT_DIR / "python" / "python.exe").exists():
    _log_dir = Path(os.environ.get("LOCALAPPDATA", str(ROOT_DIR))) / "cls-studio" / "logs"
else:
    _log_dir = ROOT_DIR / "logs"
configure_logging(log_dir=_log_dir)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# FastAPI & lightweight deps only --heavy imports deferred to background
# so uvicorn can bind the port and serve the loading page immediately.
# ---------------------------------------------------------------------------
from fastapi import FastAPI, Header, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# These imports are all lightweight (pathlib, json, threading constants):
from .core.config import (  # noqa: E402
    API_TOKEN,
    API_V1_PREFIX,
    APP_VERSION,
    PROJECTS_DIR,
    REGISTRY_DIR,
    TORCH_DEVICE_ENV_DEFAULT,
    UI_DIR,
    UI_SRC_DIR,
    require_token_for_nonlocal_bind,
)
from .core.state import SETTINGS_LOCK  # noqa: E402

# ---------------------------------------------------------------------------
# Environment setup (must happen before any router imports touch torch)
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

# clscore is installed via `pip install -e packages/clscore`.
# Fallback: add packages/clscore/ (the project dir that contains the nested
# clscore/ package) to sys.path if clscore is not installed.
try:
    import clscore  # noqa: F401
except ImportError:
    PKG_DIR = ROOT_DIR / "packages" / "clscore"
    sys.path.insert(0, str(PKG_DIR))

# ---------------------------------------------------------------------------
# App (minimal --routers are registered during background startup)
# ---------------------------------------------------------------------------
# Fail before binding, not after: a non-loopback bind with no token would serve
# every state-changing endpoint unauthenticated to the whole subnet.
require_token_for_nonlocal_bind()

app = FastAPI(title="cls-studio API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    # X-API-Token guards every /api/v1 call and X-Bank-Binding is the
    # multi-client safety check; both must survive CORS preflight or a
    # cross-origin browser client cannot talk to the API at all.
    allow_headers=["Content-Type", "Accept", "X-API-Token", "X-Bank-Binding"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-API-Version"] = APP_VERSION
    return response


from .core.security import (  # noqa: E402
    SESSION_COOKIE_NAME,
    WebSocketTokenGate,
    evaluate_request_guard,
    is_local_peer,
)

_GUARDED_PATH_PREFIXES = (API_V1_PREFIX + "/", "/v2/", "/ws/v2/")


def _is_guarded_path(path: str) -> bool:
    """Return True iff the request path is part of the authenticated API surface.

    Static UI assets, `/health`, `/docs`, `/openapi.json`, and CORS preflight
    requests stay open so the browser UI can still boot before the user has a
    token. Anything that mutates project state or exposes inference output is
    behind one of these prefixes.
    """
    return path.startswith(_GUARDED_PATH_PREFIXES)


# The guard runs unconditionally, not only when a token is configured. The
# token-less default install is exactly the case that needs it: without this,
# any page the operator visits can POST to http://127.0.0.1:8791 (CORS blocks
# reading the response, not sending the request) and clear a bank or delete a
# project, and a DNS-rebound host name grants full read access too.
@app.middleware("http")
async def guard_requests(request: Request, call_next):
    path = request.url.path
    if not _is_guarded_path(path) or path.startswith(API_V1_PREFIX + "/auth/"):
        return await call_next(request)
    verdict, reason = evaluate_request_guard(
        method=request.method,
        host_header=request.headers.get("host", ""),
        origin_header=request.headers.get("origin", ""),
        supplied_token=request.headers.get("X-API-Token", ""),
        configured_token=API_TOKEN,
        supplied_cookie=request.cookies.get(SESSION_COOKIE_NAME, ""),
        local_peer=is_local_peer(
            request.client.host if request.client else None,
            request.headers.keys(),
        ),
    )
    if verdict == "allow":
        return await call_next(request)
    return JSONResponse(
        status_code=401 if verdict == "unauthorized" else 403,
        content={"detail": reason},
    )


if API_TOKEN:
    # WebSocket handshakes bypass "http" middleware, so they are guarded
    # separately. Covers /ws/v2/*.
    app.add_middleware(WebSocketTokenGate, token=API_TOKEN, guard=_is_guarded_path)


# ---------------------------------------------------------------------------
# Structured error handlers (CLS-XXXX codes, correlation ID, unified format)
# ---------------------------------------------------------------------------
# Registered eagerly, not via the deferred router startup: a browser that has
# no credential yet must be able to reach /auth/status and /auth/session even
# while the heavy routers are still importing. These paths are exempt from the
# request guard for the same reason.
from .routers.auth import router as _auth_router  # noqa: E402

app.include_router(_auth_router, prefix=API_V1_PREFIX)

from .core.error_handlers import register_error_handlers  # noqa: E402

register_error_handlers(app)



# ---------------------------------------------------------------------------
# Startup loading screen
# ---------------------------------------------------------------------------
from .core.startup_state import (  # noqa: E402
    LOADING_HTML as _LOADING_HTML,
)
from .core.startup_state import (
    startup_state as _startup_state,
)


@app.middleware("http")
async def startup_loading_guard(request: Request, call_next):
    """Serve a loading page while startup is in progress.
    /startup-status is always available. Everything else under /ui
    gets the loading screen until ready."""
    if not _startup_state["ready"]:
        # Always let the polling endpoint through
        if request.url.path == "/startup-status":
            return await call_next(request)
        # API requests: return 503 so UI can retry gracefully
        if request.url.path.startswith(("/api/", "/v2/", "/ws/")):
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"detail": "Server is starting up", "status": "loading"},
                status_code=503,
            )
        # Serve loading page for browser requests (HTML-accepting)
        accept = request.headers.get("accept", "")
        if "text/html" in accept or request.url.path.startswith("/ui"):
            return HTMLResponse(_LOADING_HTML)
    return await call_next(request)


@app.get("/startup-status")
def get_startup_status():
    return _startup_state


# ---------------------------------------------------------------------------
# Router registration (deferred --called from background thread)
# ---------------------------------------------------------------------------
from .router_registry import register_routers  # noqa: E402


def _register_routers() -> None:
    """Back-compat alias; the registry lives in router_registry.py."""
    register_routers(app)


@app.get("/favicon.ico", include_in_schema=False)
async def _favicon() -> FileResponse:
    """Serve favicon from UI dist (or src public/) directory."""
    for base in (UI_DIR, UI_SRC_DIR / "public"):
        ico = base / "favicon.ico"
        if ico.exists():
            return FileResponse(ico, media_type="image/x-icon")
    return Response(status_code=204)


def _mount_static_files() -> None:
    """Mount UI static files (called after routers are registered)."""
    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
        assets_dir = UI_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")
    elif UI_SRC_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_SRC_DIR, html=True), name="ui")


# ---------------------------------------------------------------------------
# Startup tasks
# ---------------------------------------------------------------------------
from .core.startup_tasks import (  # noqa: E402
    _auto_build_ui,
    _check_inference_deps,
    _cleanup_orphan_project_dirs,
    _deferred_post_startup,
)


@app.on_event("startup")
def on_startup() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    # Heavy initialization runs in background so the server can serve
    # the loading page immediately.
    threading.Thread(target=_background_startup, daemon=True).start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    # Under pytest the app object is a module-level singleton and every
    # TestClient entry re-runs startup, so without this a session accumulates
    # a reaper thread per client.
    try:
        from .core.vram_reaper import stop_vram_reaper

        stop_vram_reaper(timeout=2.0)
    except Exception:  # noqa: BLE001 - shutdown must not raise
        pass


def _background_startup() -> None:
    """Run all heavy startup tasks in a background thread.

    Optimized for fast time-to-ready:
    - DB init runs in parallel with router registration (the heaviest step)
    - Dep check and health check are deferred to after ready (non-blocking)
    - UI build check uses a stamp file to skip mtime walks
    """
    import time as _time
    t0 = _time.monotonic()

    try:
        def _phase(label: str):
            """Log elapsed time since last phase."""
            now = _time.monotonic()
            dt = now - _phase.last  # type: ignore[attr-defined]
            _phase.last = now  # type: ignore[attr-defined]
            logger.info("  [startup] %s  %.1fs", label, dt)
        _phase.last = t0  # type: ignore[attr-defined]

        # --- Phase 0: UI build (must complete before static mount) ---
        _startup_state["current"] = "Checking UI build..."
        _auto_build_ui()
        _startup_state["steps"].append("UI build check")
        _phase("UI build check")

        # --- Phase 1: Parallel heavy work (routers + DB) ---
        _startup_state["current"] = "Loading modules..."

        db_result: dict = {}

        def _init_db_parallel():
            try:
                from .db import init_db
                init_db()
                db_result["ok"] = True
            except Exception as exc:
                db_result["error"] = exc

        db_thread = threading.Thread(target=_init_db_parallel, daemon=True)
        db_thread.start()

        _register_routers()
        _startup_state["steps"].append("Modules loaded")
        _phase("routers")

        # Wait for DB (usually finishes before routers)
        db_thread.join(timeout=30)
        if db_result.get("error"):
            raise db_result["error"]
        _startup_state["steps"].append("Database initialized")
        _phase("DB init")

        # --- Phase 2: Quick DB tasks ---
        # cls-studio keeps only the generic orphan-dir sweep. The cls-studio
        # training-run and annotation-mask cleanups (_cleanup_stale_runs_on_
        # startup / _cleanup_false_ok_masks) are dropped: cls-studio has no
        # training runs and no annotation masks.
        _startup_state["current"] = "Cleaning up projects..."
        _cleanup_orphan_project_dirs()
        _startup_state["steps"].append("Projects cleaned up")
        _phase("project cleanup")

        # --- Phase 3: Device setup ---
        _startup_state["current"] = "Configuring device..."
        from .core.runtime_settings import read_runtime_settings
        from .core.torch_device import resolve_torch_device_or_cpu
        logger.info("PROJECTS_DIR=%s", PROJECTS_DIR)
        saved = read_runtime_settings()
        configured = str(saved.get("torch_device", TORCH_DEVICE_ENV_DEFAULT))
        from .core import state as _state
        with SETTINGS_LOCK:
            _state.SELECTED_TORCH_DEVICE = configured
        logger.info("torch_device: configured=%s (resolving...)", configured)
        try:
            resolved = resolve_torch_device_or_cpu(configured)
            logger.info("torch_device resolved=%s", resolved)
        except Exception as exc:
            logger.warning("torch_device warmup error: %s", exc)
        _startup_state["steps"].append("Device configured")
        _phase("device setup")

        # --- Inference dependency check ---
        try:
            _check_inference_deps(resolved, _startup_state)
        except NameError:
            _check_inference_deps("cpu", _startup_state)

        # Mount static files right before marking ready
        _mount_static_files()

        elapsed = _time.monotonic() - t0
        _startup_state["current"] = ""
        _startup_state["ready"] = True
        logger.info("Startup complete in %.1fs", elapsed)

        # --- Phase 4: Deferred non-critical tasks (after ready) ---
        # These run after the UI is already accessible
        threading.Thread(target=_deferred_post_startup, daemon=True).start()

    except Exception:
        logger.exception("Startup failed")
        _startup_state["current"] = ""
        _startup_state["error"] = "Startup error --see server logs"
        _startup_state["ready"] = True  # mark ready so UI can load despite errors

