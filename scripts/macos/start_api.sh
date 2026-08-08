#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# cls-studio -- Start cls-studio API only (macOS)
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
fail() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

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

# ── Resolve Python ───────────────────────────────────────────────
PYTHON=""
for venv in ".venv-macos" ".venv"; do
    if [ -x "$REPO_ROOT/$venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/$venv/bin/python"
        break
    fi
done
[ -z "$PYTHON" ] && fail "Virtual environment not found. Run: bash scripts/macos/install_macos.sh"

# ── Force venv python for child processes (uvicorn workers) ──────
VENV_BIN="$(dirname "$(realpath "$PYTHON")")"
export VIRTUAL_ENV="$(dirname "$VENV_BIN")"
export PATH="$VENV_BIN:$PATH"

# ── Environment ──────────────────────────────────────────────────
export CLS_PROJECTS_DIR="${CLS_PROJECTS_DIR:-$HOME/Documents/ClsStudio/projects}"
export CLS_DB_PATH="${CLS_DB_PATH:-$HOME/Documents/ClsStudio/projects/app.db}"
export CLS_MODELS_DIR="${CLS_MODELS_DIR:-$REPO_ROOT/models}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HOME/Documents/ClsStudio/projects"

if [ -z "${CLS_HOST:-}" ]; then
  SETTINGS_PATH="$CLS_PROJECTS_DIR/runtime_settings.json"
  if [ -f "$SETTINGS_PATH" ] && "$PYTHON" -c "import json,sys; sys.exit(0 if bool(json.load(open(sys.argv[1],encoding='utf-8')).get('lan_access')) else 1)" "$SETTINGS_PATH" 2>/dev/null; then
    CLS_HOST="0.0.0.0"
  else
    CLS_HOST="127.0.0.1"
  fi
fi

info "Starting cls-studio API (port 8791, host=$CLS_HOST)"
info "Python: $PYTHON"
exec "$PYTHON" -m uvicorn apps.api.app.main:app \
    --host "$CLS_HOST" --port 8791
