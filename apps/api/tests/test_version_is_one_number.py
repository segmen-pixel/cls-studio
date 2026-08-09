# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The product's version is written down three times; they must agree.

``config.APP_VERSION`` sat at "0.1.0" through the 0.2.0 and 0.2.1 releases.
Nothing caught it, because nothing compared it to anything: the root
pyproject and the UI package were bumped by the release commit and this one
was not. The cost is not cosmetic -- ``/api/v1/health`` is the endpoint you
reach for to confirm WHICH build is live (it is what a deploy check reads),
and ``TRAINER_BUILD_ID`` stamps the same string onto trained artifacts, so a
model file recorded a provenance that was two releases stale.

This is the same shape as ``test_bank_layout_constants.py``: a fact restated
in more than one file, pinned so the copies cannot drift apart in silence.
The CHANGELOG is checked too, because a release whose notes are filed under
the wrong heading is a release nobody can find the notes for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.config import APP_VERSION, TRAINER_BUILD_ID

# tests/ -> api/ -> apps/ -> cls-studio/
REPO = Path(__file__).resolve().parents[3]


def test_the_api_agrees_with_the_root_project():
    """Read with a regex, NOT tomllib: the project declares requires-python
    >=3.10 and tomllib is stdlib only from 3.11, so importing it here would
    make the version check itself the thing that breaks on a supported
    interpreter."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    body = text.split("[project]", 1)[1]
    m = re.search(r'^version\s*=\s*"([^"]+)"', body, re.MULTILINE)
    assert m, "no version in the root pyproject's [project] table"
    assert APP_VERSION == m.group(1)


def test_the_api_agrees_with_the_ui_package():
    data = json.loads((REPO / "apps" / "ui" / "package.json").read_text(encoding="utf-8"))
    assert APP_VERSION == data["version"]


def test_the_newest_changelog_release_is_this_version():
    """The topmost dated heading, ignoring an empty [Unreleased]."""
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[([^\]]+)\]", text, re.MULTILINE)
    dated = [h for h in headings if h != "Unreleased"]
    assert dated, "CHANGELOG has no released version heading"
    assert dated[0] == APP_VERSION


def test_the_trainer_stamp_is_the_app_version():
    """Artifacts record the build that produced them, not a frozen literal."""
    assert TRAINER_BUILD_ID == APP_VERSION
