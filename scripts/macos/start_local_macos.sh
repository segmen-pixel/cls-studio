#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# cls-studio -- Start Local Services (macOS)
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

show_help() {
    echo ""
    echo "  cls-studio -- Start Local Services (macOS)"
    echo ""
    echo "  Usage:"
    echo "    bash scripts/macos/start_local_macos.sh [options]"
    echo ""
    echo "  Environment variables:"
    echo "    CLS_HOST=0.0.0.0            Bind to all interfaces (default: 127.0.0.1)"
    echo "    CLS_START_LABEL_STUDIO=1    Also start Label Studio"
    echo ""
    echo "  This script starts:"
    echo "    - cls-studio API on port 8791"
    echo "    - UI served by the API at http://localhost:8791/ui/"
    echo ""
    echo "  Prerequisites:"
    echo "    Run scripts/macos/install_macos.sh first."
    echo ""
    exit 0
}

for arg in "$@"; do
    case "$arg" in --help|-h) show_help ;; esac
done

# ── Locate repo root ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/apps/api/app/main.py" ]; then
            echo "$dir"
            return
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}
REPO_ROOT="$(find_repo_root "$SCRIPT_DIR")" || fail "Could not find repository root."
cd "$REPO_ROOT"

echo ""
echo "============================================================"
echo "  cls-studio -- Starting Services (macOS)"
echo "============================================================"
echo "  Repo: $REPO_ROOT"
echo ""

# ── Check venv ───────────────────────────────────────────────────
PYTHON=""
for venv in ".venv-macos" ".venv"; do
    if [ -x "$REPO_ROOT/$venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/$venv/bin/python"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    fail "Virtual environment not found. Run: bash scripts/macos/install_macos.sh"
fi
info "Using Python: $PYTHON"

# ── Force venv python for child processes (uvicorn workers) ──────
VENV_BIN="$(dirname "$(realpath "$PYTHON")")"
export VIRTUAL_ENV="$(dirname "$VENV_BIN")"
export PATH="$VENV_BIN:$PATH"

# ── Environment variables ────────────────────────────────────────
export CLS_PROJECTS_DIR="${CLS_PROJECTS_DIR:-$HOME/Documents/ClsStudio/projects}"
export CLS_DB_PATH="${CLS_DB_PATH:-$HOME/Documents/ClsStudio/projects/app.db}"
export CLS_MODELS_DIR="${CLS_MODELS_DIR:-$REPO_ROOT/models}"
export PYTHONDONTWRITEBYTECODE=1
# Annotation proxy target — only exported when Label Studio is opted in;
# leaving it unset keeps the /annotate/* proxy routes unmounted (SSRF surface).
if [ "${CLS_START_LABEL_STUDIO:-0}" = "1" ]; then
    export CLS_ANNOTATION_URL="${CLS_ANNOTATION_URL:-http://localhost:8081}"
fi

if [ -z "${CLS_HOST:-}" ]; then
  SETTINGS_PATH="$CLS_PROJECTS_DIR/runtime_settings.json"
  if [ -f "$SETTINGS_PATH" ] && "$PYTHON" -c "import json,sys; sys.exit(0 if bool(json.load(open(sys.argv[1],encoding='utf-8')).get('lan_access')) else 1)" "$SETTINGS_PATH" 2>/dev/null; then
    CLS_HOST="0.0.0.0"
  else
    CLS_HOST="127.0.0.1"
  fi
fi

# ── Ensure directories ──────────────────────────────────────────
mkdir -p "$HOME/Documents/ClsStudio/projects"
LOG_DIR="$REPO_ROOT/logs/macos"
mkdir -p "$LOG_DIR"
echo "[$(date)] start_local_macos.sh REPO_ROOT=$REPO_ROOT" >> "$LOG_DIR/start_local.log"

# ── Check port conflicts ────────────────────────────────────────
check_port() {
    local port=$1 name=$2
    if lsof -iTCP:"$port" -sTCP:LISTEN -t &>/dev/null; then
        local pid
        pid="$(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1)"
        warn "Port $port ($name) already in use (PID $pid)"
        return 1
    fi
    return 0
}
PORT_CONFLICT=0
check_port 8791 "cls-studio API" || PORT_CONFLICT=1
if [ "$PORT_CONFLICT" -eq 1 ]; then
    warn "Existing services may be running. Run stop_local_macos.sh first."
    echo ""
fi

# ── Start services ───────────────────────────────────────────────
info "Starting cls-studio API (port 8791, host=$CLS_HOST)"
nohup "$PYTHON" -m uvicorn apps.api.app.main:app \
    --host "$CLS_HOST" --port 8791 \
    >> "$LOG_DIR/trainer.log" 2>&1 &
echo $! > "$LOG_DIR/trainer.pid"

# ── UI dev server (optional) ────────────────────────────────────
if command -v npm &>/dev/null; then
    info "Starting Vite UI dev server (port 5173, host=$CLS_HOST)"
    nohup npm --prefix apps/ui run dev -- --host "$CLS_HOST" --port 5173 \
        >> "$LOG_DIR/ui_dev.log" 2>&1 &
    echo $! > "$LOG_DIR/ui_dev.pid"
elif [ -f "$REPO_ROOT/apps/ui/dist/index.html" ]; then
    info "npm not found, but UI build exists. Serving via API static mount."
else
    warn "npm not found and no UI build. UI unavailable."
fi

# ── Label Studio (optional, opt-in) ─────────────────────────────
# Requires user-supplied credentials; no defaults are provided.
if [ "${CLS_START_LABEL_STUDIO:-0}" = "1" ]; then
    if [ -z "${LABEL_STUDIO_USERNAME:-}" ] || [ -z "${LABEL_STUDIO_PASSWORD:-}" ] || [ -z "${LABEL_STUDIO_EMAIL:-}" ]; then
        warn "CLS_START_LABEL_STUDIO=1 requires LABEL_STUDIO_USERNAME, LABEL_STUDIO_PASSWORD and LABEL_STUDIO_EMAIL. Skipping Label Studio."
    else
        info "Starting Label Studio (port 8081)"
        LABEL_STUDIO_USERNAME="$LABEL_STUDIO_USERNAME" LABEL_STUDIO_PASSWORD="$LABEL_STUDIO_PASSWORD" LABEL_STUDIO_EMAIL="$LABEL_STUDIO_EMAIL" \
        nohup "$PYTHON" -m label_studio.server start --port 8081 --no-browser \
            >> "$LOG_DIR/label_studio.log" 2>&1 &
        echo $! > "$LOG_DIR/label_studio.pid"
    fi
fi

echo ""
echo "============================================================"
echo "  Services started successfully"
echo "============================================================"
echo ""
echo "  cls-studio UI  : http://$CLS_HOST:8791/ui/"
echo "  cls-studio API : http://$CLS_HOST:8791/docs"
echo "  Logs        : $LOG_DIR"
echo ""
echo "  To stop all: bash scripts/macos/stop_local_macos.sh"
echo ""

# ── Wait for API ready, then open browser ────────────────────────
info "Waiting for API to be ready..."
READY=0
for i in $(seq 1 30); do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8791/startup-status', timeout=2)" &>/dev/null; then
        READY=1
        break
    fi
    sleep 2
done

if [ "$READY" -eq 1 ]; then
    ok "API is ready. Opening browser..."
else
    warn "API did not respond within 60s. Opening browser anyway..."
fi
open "http://$CLS_HOST:8791/ui/"
