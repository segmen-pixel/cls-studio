# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import threading

RUN_FLAGS: dict[str, threading.Event] = {}
TRAIN_GUARDS: dict[str, threading.Lock] = {}
TRAIN_GUARDS_LOCK = threading.Lock()
ACTIVE_TORCH_JOBS: dict[str, dict[str, str]] = {}
ACTIVE_TORCH_JOBS_LOCK = threading.Lock()

# Mutable device setting (mutate via: import core.state as _state; _state.SELECTED_TORCH_DEVICE = ...)
from .config import TORCH_DEVICE_ENV_DEFAULT

SELECTED_TORCH_DEVICE: str = TORCH_DEVICE_ENV_DEFAULT
SETTINGS_LOCK = threading.Lock()
