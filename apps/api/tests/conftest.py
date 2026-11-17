# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Shared fixtures for api unit tests.

Environment variables CLS_PROJECTS_DIR and CLS_DB_PATH are set at module
level (before any app import) so that the FastAPI app uses a temporary
directory instead of the real project store.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Temp directory + env-var override  (MUST happen before app imports)
# ---------------------------------------------------------------------------
_TEST_ROOT = tempfile.mkdtemp(prefix="seg_test_")
os.environ["CLS_PROJECTS_DIR"] = _TEST_ROOT
os.environ["CLS_DB_PATH"] = os.path.join(_TEST_ROOT, "test.db")
# TestClient sends Host "testserver"; declare it so the request guard
# (loopback-only Host allowlist for the tokenless case) admits the harness.
os.environ.setdefault("CLS_ALLOWED_HOSTS", "testserver")

# 2. Add api to sys.path so `from app.main import app` works
_TRAINER_API_DIR = str(Path(__file__).resolve().parents[1])
if _TRAINER_API_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_API_DIR)

# Also ensure the project root is on sys.path (for packages/clscore)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 3. Now safe to import app modules
# ---------------------------------------------------------------------------
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Importing the models is what registers them on SQLModel.metadata. Without
# it, create_all() below has nothing to create and the first migration's
# ALTER TABLE fails with "no such table: project". A whole-directory run
# happened to work because some other collected module imported them first,
# which left every test file unrunnable on its own.
from app import models as _models  # noqa: F401
from app.db import init_db
from app.main import app


# ---------------------------------------------------------------------------
# Session-scoped: initialise test DB once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _init_test_env():
    """Create the test DB and clean up when the session ends."""
    Path(_TEST_ROOT).mkdir(parents=True, exist_ok=True)
    init_db()
    yield
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    """Session-scoped FastAPI TestClient (one startup for the whole run).

    The app is a module-level singleton and entering TestClient re-runs
    the startup event, which re-registers every router on the same app.
    A per-test client therefore grows app.routes each test until startup
    eventually fails; one shared client avoids the re-entry entirely.
    """
    import time as _time
    # Give the client a loopback peer: the request guard distinguishes a
    # local browser from a LAN caller by the TCP peer address, and
    # TestClient's default 'testclient' host is neither.
    with TestClient(app, client=("127.0.0.1", 50000)) as c:
        # Wait for background startup (router registration) to complete
        from app.core.startup_state import startup_state as _startup_state
        deadline = _time.monotonic() + 30
        while not _startup_state.get("ready") and _time.monotonic() < deadline:
            _time.sleep(0.1)
        yield c


@pytest.fixture
def project_id(client):
    """Create a temporary project and delete it after the test."""
    resp = client.post("/api/v1/projects", json={"name": "pytest-tmp"})
    assert resp.status_code == 200
    pid = resp.json()["id"]
    yield pid
    client.delete(f"/api/v1/projects/{pid}")


@pytest.fixture
def sample_image_bytes():
    """16x16 red PNG image as raw bytes."""
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_mask_png(arr: np.ndarray) -> bytes:
    """Encode a uint8 numpy array as a PNG byte string (single-channel)."""
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
