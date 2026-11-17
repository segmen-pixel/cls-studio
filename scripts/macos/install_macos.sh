#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# cls-studio -- macOS Install Script
# Usage: bash scripts/macos/install_macos.sh [--skip-ui] [--help]
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

show_help() {
    echo ""
    echo "  cls-studio -- macOS Installer"
    echo ""
    echo "  Usage:"
    echo "    bash scripts/macos/install_macos.sh [options]"
    echo ""
    echo "  Options:"
    echo "    --skip-ui              Skip UI (npm) build"
    echo "    --with-coreml          Also install coremltools, for Core ML export"
    echo "    --help, -h             Show this help"
    echo ""
    exit 0
}

# ── Parse arguments ──────────────────────────────────────────────
SKIP_UI=0
for arg in "$@"; do
    case "$arg" in
        --skip-ui)            SKIP_UI=1 ;;
        --with-coreml)        WITH_COREML=1 ;;
        --help|-h)            show_help ;;
        *) warn "Unknown option: $arg" ;;
    esac
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

# ── Log setup ────────────────────────────────────────────────────
LOG_DIR="$REPO_ROOT/logs/macos"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install_macos.log"
echo "=== install_macos.sh $(date) ===" > "$LOG_FILE"
echo "Repo root: $REPO_ROOT" >> "$LOG_FILE"

# ── Architecture detection ───────────────────────────────────────
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon (arm64) detected — MPS GPU acceleration available"
else
    info "Intel Mac ($ARCH) detected — CPU only"
fi

echo ""
echo "============================================================"
echo "  cls-studio macOS Installer"
echo "============================================================"
echo "  Repo root  : $REPO_ROOT"
echo "  Architecture: $ARCH"
echo "  Log file   : $LOG_FILE"
echo "============================================================"
echo ""

# ============================================================
#  STEP 1: Prerequisites check
# ============================================================
info "[STEP 1/6] Checking prerequisites..."
PREREQ_OK=1

# Python.
#
# The upper bound is not a whim: apps/api/requirements.txt pins a torch
# version, and torch publishes wheels for a bounded set of interpreters. Pip
# cannot install a pinned torch into a Python it has no wheel for, so an
# interpreter above the bound fails several minutes into STEP 3 with an error
# that reads like a network problem. Checking it here costs nothing.
#
# Keep PY_MAX_MINOR in step with the torch pin: as of torch 2.6 the macOS
# arm64 wheels stop at cp313.
PY_MIN_MINOR=10
PY_MAX_MINOR=13

py_minor_of() {
    "$1" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null
}

py_in_range() {
    local exe="$1" major minor
    major="$("$exe" -c 'import sys; print(sys.version_info.major)' 2>/dev/null)" || return 1
    minor="$(py_minor_of "$exe")" || return 1
    [ "$major" = "3" ] && [ "$minor" -ge "$PY_MIN_MINOR" ] && [ "$minor" -le "$PY_MAX_MINOR" ]
}

PY=""
if [ -n "${CLS_PYTHON:-}" ]; then
    # An explicit choice wins, and a bad one is an error rather than a silent
    # fallback: someone who set this wants to know it was ignored.
    if [ ! -x "$CLS_PYTHON" ]; then
        fail "CLS_PYTHON is not executable: $CLS_PYTHON"
    elif py_in_range "$CLS_PYTHON"; then
        PY="$CLS_PYTHON"
    else
        fail "CLS_PYTHON is Python $("$CLS_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo '?') — need 3.$PY_MIN_MINOR to 3.$PY_MAX_MINOR"
    fi
elif command -v python3 &>/dev/null && py_in_range "$(command -v python3)"; then
    PY="$(command -v python3)"
else
    # The default python3 is out of range (or missing). Newest supported
    # first: a machine with several Pythons should get the best one, not the
    # oldest one.
    for minor in $(seq "$PY_MAX_MINOR" -1 "$PY_MIN_MINOR"); do
        if command -v "python3.$minor" &>/dev/null; then
            PY="$(command -v "python3.$minor")"
            break
        fi
    done
fi

if [ -n "$PY" ]; then
    PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    ok "Python $PY_VER ($PY)"
    if [ "$PY" != "$(command -v python3 2>/dev/null)" ]; then
        info "Using $PY rather than the default python3"
    fi
else
    PREREQ_OK=0
    FOUND_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 'none')"
    echo "  Python ...... need 3.$PY_MIN_MINOR to 3.$PY_MAX_MINOR, found $FOUND_VER"
    echo ""
    echo "  The pinned PyTorch build has no wheel for Python $FOUND_VER."
    echo "  Install a supported one and re-run:"
    echo "      brew install python@3.$PY_MAX_MINOR"
    echo "  Or point the installer at one you already have:"
    echo "      CLS_PYTHON=/path/to/python3.$PY_MAX_MINOR bash install-macos.sh"
fi

# Git
if command -v git &>/dev/null; then
    ok "git $(git --version | awk '{print $3}')"
else
    PREREQ_OK=0; warn "git not found — install via: brew install git"
fi

# npm (optional for UI)
if command -v npm &>/dev/null; then
    ok "npm $(npm --version)"
else
    warn "npm not found — UI build will be skipped"
    SKIP_UI=1
fi

[ "$PREREQ_OK" -eq 0 ] && fail "Missing prerequisites. Install them and re-run."
echo ""

# ============================================================
#  STEP 2: Create virtual environment
# ============================================================
info "[STEP 2/6] Setting up virtual environment..."
VENV_DIR="$REPO_ROOT/.venv-macos"

if [ -d "$VENV_DIR" ]; then
    info "Existing venv found at $VENV_DIR"
    # An existing venv built on an out-of-range interpreter would fail the
    # same way a fresh one would, and the cause would be invisible.
    if [ -x "$VENV_DIR/bin/python" ] && ! py_in_range "$VENV_DIR/bin/python"; then
        fail "$VENV_DIR was built with Python $("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])') — remove it and re-run"
    fi
else
    # "$PY", not python3: the prerequisite check may have picked a different
    # interpreter than the one on PATH.
    "$PY" -m venv "$VENV_DIR" 2>&1 | tee -a "$LOG_FILE"
    ok "Created venv: $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
PIP="$PYTHON -m pip"

# Upgrade pip
$PIP install --upgrade pip setuptools wheel >> "$LOG_FILE" 2>&1
ok "pip upgraded"
echo ""

# ============================================================
#  STEP 3: Install Python dependencies
# ============================================================
info "[STEP 3/6] Installing Python dependencies..."

# One install from the lockfile, which is where the torch pin lives. Fetching
# torch separately first only downloaded a build the next line replaced —
# gigabytes of traffic for nothing. The default PyPI wheel carries MPS on
# Apple Silicon, so no extra index is needed here.
info "Installing dependencies from the lockfile (PyTorch included — MPS on Apple Silicon)"
$PIP install -r "$REPO_ROOT/apps/api/requirements.txt" 2>&1 | tee -a "$LOG_FILE"



# clscore (local package, editable)
$PIP install -e "$REPO_ROOT/packages/clscore" 2>&1 | tee -a "$LOG_FILE"

# Core ML export, opt-in. Its lockfile is compiled against the main one, so it
# adds coremltools and its own dependencies without moving a pin the banks
# were built with -- unconstrained it wanted numpy 2.5.2 over the pinned 2.4.4.
# Only macOS gets this at all: coremltools ships no Windows wheel, and the
# sdist pip falls back to there builds without the extensions that write an
# mlpackage or run a prediction.
if [ "${WITH_COREML:-0}" = "1" ]; then
    info "Installing Core ML export dependencies"
    $PIP install -r "$REPO_ROOT/apps/api/requirements-coreml.txt" 2>&1 | tee -a "$LOG_FILE"
fi

ok "Python dependencies installed"
echo ""

# ============================================================
#  STEP 4: Verify PyTorch + MPS
# ============================================================
info "[STEP 4/6] Verifying PyTorch installation..."

"$PYTHON" -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  MPS available:   {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
" 2>&1 | tee -a "$LOG_FILE"

ok "PyTorch verified"
echo ""

# ============================================================
#  STEP 5: Build UI
# ============================================================
if [ "$SKIP_UI" -eq 0 ]; then
    info "[STEP 5/6] Building UI..."
    UI_DIR="$REPO_ROOT/apps/ui"
    if [ -f "$UI_DIR/package.json" ]; then
        (cd "$UI_DIR" && npm install && npm run build) 2>&1 | tee -a "$LOG_FILE"
        ok "UI built"
    else
        warn "No package.json found at $UI_DIR"
    fi
else
    info "[STEP 5/6] Skipping UI build (--skip-ui)"
fi
echo ""

# ============================================================
#  STEP 6: Summary
# ============================================================
info "[STEP 6/6] Installation complete!"
echo ""
echo "============================================================"
echo "  Installation Summary"
echo "============================================================"
echo "  Venv       : $VENV_DIR"
echo "  Python     : $("$PYTHON" --version 2>&1)"
echo "  Architecture: $ARCH"
echo ""
echo "  To start cls-studio:"
echo "    bash scripts/macos/start_local_macos.sh"
echo ""
echo "  Or manually:"
echo "    source .venv-macos/bin/activate"
echo "    python -m uvicorn apps.api.app.main:app --port 8791"
echo "    Then open: http://localhost:8791/ui/"
echo "============================================================"
