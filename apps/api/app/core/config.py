# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Reduce CUDA memory fragmentation on long-running trainer API processes.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Ensure JS modules are served with a browser-acceptable MIME type on Windows.
# ---------------------------------------------------------------------------
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# SECURITY NOTE: sys.path manipulation below adds only the local
# "packages/clscore/" project directory so that the monorepo's own clscore
# package takes precedence over any stale site-packages copies.  No external
# or user-controlled paths are added.  This is a standard monorepo pattern.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[4]
PKG_DIR = ROOT_DIR / "packages" / "clscore"
pkg_path = str(PKG_DIR)
if pkg_path in sys.path:
    sys.path.remove(pkg_path)
sys.path.insert(0, pkg_path)

# ---------------------------------------------------------------------------
# User data root (projects/). MUST live OUTSIDE the repo tree.
#
# History: during the v2 re-platforming a stray server whose CWD was the repo
# wrote thousands of user-data files into <repo>/projects/. To make that class
# of leak structurally impossible — not merely cleaned up after the fact —
# cls-studio (a) defaults the projects root to the user's Documents dir, never
# the repo, and (b) hard-refuses to start if the resolved root is inside the
# repo tree. Env var is CLS_PROJECTS_DIR; a legacy CLS_STATE_DIR
# (with /projects appended) is still honoured for older launch scripts.
# ---------------------------------------------------------------------------
_DEFAULT_PROJECTS_DIR = Path.home() / "Documents" / "ClsStudio" / "projects"


def _env(*names):
    """First env var that is set."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


_projects_override = _env("CLS_PROJECTS_DIR")
_legacy_state = _env("CLS_STATE_DIR", "CLS_DATA_DIR")
if _projects_override:
    PROJECTS_DIR = Path(_projects_override)
elif _legacy_state:
    PROJECTS_DIR = Path(_legacy_state) / "projects"
else:
    PROJECTS_DIR = _DEFAULT_PROJECTS_DIR

# Startup guard: a projects root inside the repo tree is a data-leak footgun.
# Refuse to run rather than silently write user data into a git checkout.
_resolved_projects = PROJECTS_DIR.resolve()
if _resolved_projects.is_relative_to(ROOT_DIR):
    raise RuntimeError(
        "Cls-Studio refuses to start: the projects directory "
        f"({_resolved_projects}) is inside the repository tree ({ROOT_DIR}). "
        "User data must live outside the repo. Set CLS_PROJECTS_DIR to a "
        f"path outside the repo, or unset it to use the default "
        f"({_DEFAULT_PROJECTS_DIR})."
    )

DEFAULT_MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR = Path(_env("CLS_MODELS_DIR")
                  or str(DEFAULT_MODELS_DIR))
REGISTRY_DIR = MODELS_DIR / "registry"
# CVAT / annotation reverse-proxy targets.
# Unset (the default) → the `/cvat/*` and `/annotate/*` routes are NOT mounted,
# so the trainer API never proxies outbound HTTP on the user's behalf. This
# closes the localhost-SSRF surface when `CLS_HOST=0.0.0.0` exposes the API
# to a LAN.  Set these env vars only when an upstream annotation service is
# actually running and intentionally fronted by the trainer API.
_CVAT_BASE_URL_RAW = os.getenv("CLS_CVAT_URL")
_ANNOTATION_BASE_URL_RAW = os.getenv("CLS_ANNOTATION_URL")
CVAT_BASE_URL: str | None = _CVAT_BASE_URL_RAW.rstrip("/") if _CVAT_BASE_URL_RAW else None
ANNOTATION_BASE_URL: str | None = _ANNOTATION_BASE_URL_RAW.rstrip("/") if _ANNOTATION_BASE_URL_RAW else None
UI_SRC_DIR = ROOT_DIR / "apps" / "ui"
UI_DIR = UI_SRC_DIR / "dist"
APP_VERSION = "0.1.0"
APP_BUILD_DATE = "2026-07-10"
TRAINER_BUILD_ID = APP_VERSION
API_V1_PREFIX = "/api/v1"

# Optional shared-secret for LAN / reverse-proxy deployments.
# Empty string (default) disables the check — safe for localhost-only
# operation. See SECURITY.md.
API_TOKEN = os.getenv("CLS_API_TOKEN", "").strip()

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})


def resolve_bind_host() -> str:
    """The interface uvicorn will bind to, resolved the way the launcher does.

    CLS_HOST wins if set; otherwise the persisted ``lan_access`` flag chooses
    between all-interfaces and loopback. Kept in sync with
    scripts/windows/_resolve_host.py so the auth gate reasons about the host
    that will actually be served.
    """
    env_host = (os.getenv("CLS_HOST") or "").strip()
    if env_host:
        return env_host
    try:
        import json as _json
        data = _json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and bool(data.get("lan_access", False)):
            return "0.0.0.0"
    except (OSError, ValueError):
        pass
    return "127.0.0.1"


def is_nonlocal_bind(host: str | None = None) -> bool:
    """True when the resolved bind host is reachable beyond this machine."""
    h = (host if host is not None else resolve_bind_host()).strip()
    return h not in _LOOPBACK_HOSTS


def require_token_for_nonlocal_bind() -> None:
    """Abort startup if the server is exposed off-box without a shared secret.

    A non-loopback bind with an empty token would serve every state-changing
    endpoint unauthenticated to anything that can reach the port. Rather than
    do that silently, stop with an actionable message. scripts/_lan_token.py
    mints and persists a token so the launchers never hit this.
    """
    if is_nonlocal_bind() and not API_TOKEN:
        host = resolve_bind_host()
        raise SystemExit(
            "Refusing to start: the server is bound to a non-loopback address "
            f"({host!r}) but CLS_API_TOKEN is empty, which would leave the API "
            "unauthenticated on the network. Either set CLS_API_TOKEN to a "
            "shared secret, or bind to 127.0.0.1 (unset CLS_HOST / disable LAN "
            "access) for local-only use."
        )

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
FIXED_INPUT_SIZE = [256, 256]
OUTPUT_STRIDE = 2
CLASS_ORDER = [0, 1]  # legacy default; use read_num_classes() for dynamic
NUM_CLASSES = len(CLASS_ORDER)   # legacy default; prefer dynamic lookup
IGNORE_INDEX = 255


def read_num_classes(classes_payload: dict) -> int:
    """Derive num_classes from classes payload: max(class_id) + 1.

    This ensures the model output dimension covers all class IDs present
    in masks (which use class_id as pixel value).
    """
    class_ids = [int(item.get("id", 0)) for item in classes_payload.get("classes", [])]
    if not class_ids:
        return NUM_CLASSES  # fallback to legacy default
    return max(class_ids) + 1


def read_class_ids(classes_payload: dict) -> list[int]:
    """Extract sorted class IDs from classes payload."""
    class_ids = [int(item.get("id", 0)) for item in classes_payload.get("classes", [])]
    return sorted(set(class_ids)) if class_ids else list(CLASS_ORDER)

NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
AUTO_CLASS_WEIGHT_FG_RATIO_LOW = 0.01
AUTO_CLASS_WEIGHT_FG_RATIO_HIGH = 0.12
AUTO_CLASS_WEIGHT_STRENGTH_SCALE = 0.80
AUTO_BG_WEIGHT_BOOST_MAX = 1.0
AUTO_VAL_TARGET_RATIO = 0.25
AUTO_VAL_MIN_COUNT = 6

# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------
RUNTIME_SETTINGS_PATH = PROJECTS_DIR / "runtime_settings.json"
TORCH_DEVICE_ENV_DEFAULT = os.getenv("CLS_TORCH_DEVICE", "auto").strip().lower() or "auto"

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# ---------------------------------------------------------------------------
# Aggregate caps for multi-file uploads (batch teach / staging drop).
# MAX_UPLOAD_BYTES bounds ONE part, so the *ingested upload bytes* of an N-part
# request are otherwise unbounded — in RAM (append_batch holds every image's
# bytes) and on disk (staging).
# Scope of the guarantee: the running total is checked after each part has been
# buffered, so the peak is MAX_UPLOAD_TOTAL_BYTES + one MAX_UPLOAD_BYTES part,
# and the decoded ndarrays append_batch keeps next to the bytes are NOT counted
# (a PNG decodes to many times its compressed size). The reverse proxy's
# request body limit is the hard ceiling — see docs/deployment.md.
# Bad env values fall back to the default instead of killing startup — same
# idiom as the bank capacity ceilings in routers/bank.py.
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


MAX_UPLOAD_TOTAL_BYTES = _env_int("CLS_MAX_UPLOAD_TOTAL_MB", 2048) * 1024 * 1024
MAX_UPLOAD_FILES = _env_int("CLS_MAX_UPLOAD_FILES", 1024)

# ---------------------------------------------------------------------------
# Archive imports (bank package, project package) get their own ceiling.
# MAX_UPLOAD_BYTES is a per-IMAGE bound; applying it to an archive meant a bank
# you had just exported could not be imported back — /bank/export's own
# docstring says the package "can run to several GB" while /banks/import
# rejected anything over 200 MB. A whole-project package is larger still.
# It stays a guard, not an invitation: the archive is streamed to disk, so the
# real limit is free space, and the reverse proxy's body limit is the hard
# ceiling. Same bad-value-falls-back-to-default idiom as above.
# ---------------------------------------------------------------------------
MAX_ARCHIVE_BYTES = _env_int("CLS_MAX_ARCHIVE_MB", 64 * 1024) * 1024 * 1024
