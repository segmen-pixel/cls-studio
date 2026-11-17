# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""``routers/projects.py`` must not drag torch in, and neither must bank_images.

The projects router serves the project list, which is the first thing the app
renders. It reaches only fastapi/sqlmodel/PIL/numpy today, deliberately —
``annotate_index.py`` even keeps its ``clscore.image_io`` import function-local
to preserve the property. ``bank_images`` is imported by that router, so it
inherits the constraint: one ``from clscore.store import STORE_SUBDIR`` in it
would execute ``clscore/__init__.py`` and load torch on the project list.

A subprocess is required. By the time this test runs, the pytest session has
imported torch through other modules, so an in-process ``sys.modules`` check
proves nothing at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_API_DIR = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[3]


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_API_DIR), str(_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_bank_images_pulls_in_no_heavy_dependency():
    r = _run(
        "import sys\n"
        "import app.core.bank_images\n"
        "heavy = [m for m in ('torch', 'clscore', 'cv2', 'numpy') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    assert r.returncode == 0, f"bank_images imported {r.stderr.strip()}"


def test_the_projects_router_still_reaches_no_torch():
    r = _run(
        "import sys\n"
        "import app.routers.projects\n"
        "assert 'torch' not in sys.modules, 'the project list now loads torch'\n"
    )
    if r.returncode != 0 and "ModuleNotFoundError" in r.stderr:
        pytest.skip(f"router not importable standalone: {r.stderr.strip()[-200:]}")
    assert r.returncode == 0, r.stderr.strip()[-2000:]
