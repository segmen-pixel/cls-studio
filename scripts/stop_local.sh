#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
set -euo pipefail

pkill -f "uvicorn apps.api.app.main" || true
pkill -f "http.server 5173" || true
pkill -f "label-studio" || true

echo "Stopped local services."
