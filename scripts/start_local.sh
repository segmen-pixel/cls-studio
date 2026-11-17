#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Locate venv python: try repo-local venvs, then system python
VENV_PY=""
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  VENV_PY="$REPO_ROOT/.venv/bin/python"
elif [ -x "$REPO_ROOT/.venv-windows-cu130/Scripts/python.exe" ]; then
    VENV_PY="$REPO_ROOT/.venv-windows-cu130/Scripts/python.exe"
elif [ -x "$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe" ]; then
  VENV_PY="$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe"
elif [ -x "$REPO_ROOT/.venv-windows/Scripts/python.exe" ]; then
  VENV_PY="$REPO_ROOT/.venv-windows/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
  VENV_PY="python3"
elif command -v python &>/dev/null; then
  VENV_PY="python"
fi

if [ -z "$VENV_PY" ]; then
  echo "No python found. Create a venv at $REPO_ROOT/.venv or install python3." >&2
  exit 1
fi

# Force venv python for all child processes (uvicorn workers).
# Without this, workers may pick up a system python from PATH.
VENV_BIN_DIR="$(dirname "$(realpath "$VENV_PY")")"
export VIRTUAL_ENV="$(dirname "$VENV_BIN_DIR")"
export PATH="$VENV_BIN_DIR:$PATH"

cd "$REPO_ROOT"

export CLS_PROJECTS_DIR="$HOME/Documents/ClsStudio/projects"
export CLS_DB_PATH="$HOME/Documents/ClsStudio/projects/app.db"

# Host binding — default to localhost only; the GUI Settings dialog can opt
# into LAN access by persisting `lan_access: true` in runtime_settings.json.
# An explicit CLS_HOST env var still wins.
if [ -z "${CLS_HOST:-}" ]; then
  SETTINGS_PATH="$CLS_PROJECTS_DIR/runtime_settings.json"
  if [ -f "$SETTINGS_PATH" ] && "$VENV_PY" -c "import json,sys; sys.exit(0 if bool(json.load(open(sys.argv[1],encoding='utf-8')).get('lan_access')) else 1)" "$SETTINGS_PATH" 2>/dev/null; then
    CLS_HOST="0.0.0.0"
  else
    CLS_HOST="127.0.0.1"
  fi
fi
export CLS_HOST

# Start Label Studio (annotation tool) — opt-in only.
# Set CLS_START_LABEL_STUDIO=1 AND supply your own credentials via
# LABEL_STUDIO_USERNAME / LABEL_STUDIO_PASSWORD / LABEL_STUDIO_EMAIL.
# No default credentials are provided.
if [ "${CLS_START_LABEL_STUDIO:-0}" = "1" ]; then
  if [ -z "${LABEL_STUDIO_USERNAME:-}" ] || [ -z "${LABEL_STUDIO_PASSWORD:-}" ] || [ -z "${LABEL_STUDIO_EMAIL:-}" ]; then
    echo "CLS_START_LABEL_STUDIO=1 requires LABEL_STUDIO_USERNAME, LABEL_STUDIO_PASSWORD and LABEL_STUDIO_EMAIL to be set. Skipping Label Studio." >&2
  elif "$VENV_PY" -m pip show label-studio >/dev/null 2>&1; then
    # Derive label-studio bin from the same venv
    VENV_BIN_DIR="$(dirname "$VENV_PY")"
    LABEL_STUDIO_BIN="$VENV_BIN_DIR/label-studio"
    export LABEL_STUDIO_USERNAME LABEL_STUDIO_PASSWORD LABEL_STUDIO_EMAIL
    if [ -x "$LABEL_STUDIO_BIN" ]; then
      nohup "$LABEL_STUDIO_BIN" start \
        --port 8081 \
        --no-browser \
        --username "$LABEL_STUDIO_USERNAME" \
        --password "$LABEL_STUDIO_PASSWORD" \
        > /tmp/seg_labelstudio.log 2>&1 &
    else
      nohup "$VENV_PY" -m label_studio.server start --port 8081 --no-browser > /tmp/seg_labelstudio.log 2>&1 &
    fi
    echo "Started Label Studio on 8081"
  else
    echo "Label Studio not installed. Install with: $VENV_PY -m pip install label-studio" >&2
  fi
fi

# Start cls-studio API
nohup "$VENV_PY" -m uvicorn apps.api.app.main:app --host "$CLS_HOST" --reload --port 8791 > /tmp/seg_trainer.log 2>&1 &

# Serve UI if build exists
if [ -d "$REPO_ROOT/apps/ui/dist" ]; then
  nohup "$VENV_PY" -m http.server 5173 --directory "$REPO_ROOT/apps/ui/dist" > /tmp/seg_ui.log 2>&1 &
else
  echo "cls-studio UI build not found. Run: (cd apps/ui && npm install && npm run build)" >&2
fi


