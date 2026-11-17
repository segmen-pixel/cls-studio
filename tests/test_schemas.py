# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
"""Unit tests for Pydantic schema validation in api."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_API_DIR = str(_REPO_ROOT / "apps" / "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from pydantic import ValidationError

from app.schemas import ClassesPayload, ClassItem, ProjectCreate


# ===================================================================
# ProjectCreate
# ===================================================================
class TestProjectCreate:
    def test_valid(self):
        p = ProjectCreate(name="test project")
        assert p.name == "test project"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ProjectCreate(name="")

    def test_optional_fields(self):
        p = ProjectCreate(name="x")
        assert p.description is None
        assert p.memo is None


# ===================================================================
# ClassesPayload
# ===================================================================
class TestClassesPayload:
    def test_valid(self):
        payload = ClassesPayload(
            version=1,
            ignore_index=255,
            classes=[
                ClassItem(id=0, name="bg", color=[0, 0, 0], active=True),
                ClassItem(id=1, name="defect", color=[255, 0, 0], active=True),
            ],
        )
        assert len(payload.classes) == 2

    def test_empty_classes(self):
        payload = ClassesPayload(version=1, ignore_index=255, classes=[])
        assert len(payload.classes) == 0
