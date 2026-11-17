# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Project ids are a path-length budget, not just an identifier.

Every file in a project sits under its id, so the id's length is paid once per
path and Windows stops at 260 characters. These tests pin the two properties
that follow: new ids stay short, and directories named the old way keep being
recognised -- they hold user data that cannot be regenerated.
"""
from __future__ import annotations

import re

from app.core import paths


def test_new_ids_are_short_and_hex():
    for _ in range(50):
        pid = paths.new_project_id()
        assert re.fullmatch(r"[0-9a-f]{12}", pid), pid


def test_new_ids_do_not_repeat():
    ids = {paths.new_project_id() for _ in range(200)}
    assert len(ids) == 200


def test_new_id_avoids_a_directory_that_already_exists(monkeypatch, tmp_path):
    """The id has to be free on disk, not merely unique in memory."""
    taken = "abcdef012345"
    (tmp_path / taken).mkdir()
    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path)
    seq = iter([taken, taken, "0123456789ab"])

    class _FakeUUID:
        def __init__(self, value):
            self.hex = value

    monkeypatch.setattr(paths.uuid, "uuid4", lambda: _FakeUUID(next(seq)))
    assert paths.new_project_id() == "0123456789ab"


def test_directories_written_by_earlier_versions_are_still_recognised():
    # Projects created before the id was shortened are full UUIDs. Failing to
    # recognise one would make the startup sweep skip adopting it.
    assert paths.looks_like_project_id("06897bd2-fff7-4fb7-9da5-6efb90a83182")
    assert paths.looks_like_project_id("06897BD2-FFF7-4FB7-9DA5-6EFB90A83182")


def test_current_ids_are_recognised():
    assert paths.looks_like_project_id("0123456789ab")


def test_unrelated_directories_are_not_mistaken_for_projects():
    for name in ("", "banks", "..", "0123456789", "0123456789abc",
                 "zzzzzzzzzzzz", "_tmp", "0123456789ab.deleted"):
        assert not paths.looks_like_project_id(name), name


def test_id_length_leaves_room_under_max_path():
    # The budget this change exists to protect: a taught image's path is
    # <projects root>/<id>/banks/<bank>/_images/<tier>/<user filename>.
    assert paths.PROJECT_ID_LEN <= 12
