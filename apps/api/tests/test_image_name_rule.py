# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""One rule decides what a taught image is called on disk.

The writer and the reader used to decide it separately: ``save_source_image``
replaced every character outside ``[A-Za-z0-9._-]`` with ``_``, and
``image_path`` *rejected* any name that contained one. Those agree only for a
name the writer produced — and the name every caller holds is the ORIGINAL,
recorded in ``bank_meta.bank_images`` and in the store index.

So a real project of 635 images called ``img001 (1)_豆_1.png`` had every single
thumbnail request refused as "invalid image name", and the refusal happened
one line BEFORE the store fallback that exists precisely so the viewer is not
left on a black rectangle. Both halves now go through
``ClsStudioState.safe_image_name``.
"""

from __future__ import annotations

import pytest

from app.core.cls_state import ClsStudioState
from app.core.exceptions import PathTraversalError

# The shape that broke it: a space, parentheses, and a non-ASCII character.
REAL_NAME = "img001 (1)_豆_1.png"


def test_the_writers_rule_is_the_readers_rule():
    """A name the writer produced must round-trip unchanged."""
    written = ClsStudioState.safe_image_name(REAL_NAME)
    assert written == "img001__1____1.png"
    assert ClsStudioState.safe_image_name(written) == written


@pytest.mark.parametrize("name", [
    REAL_NAME,
    "a photo.png",
    "テスト.png",       # wholly non-ASCII
    "img (2).JPG",
])
def test_original_names_resolve_instead_of_raising(client, project_id, name):
    """The reader maps the original name; it must not refuse it.

    Refusing is what made the thumbnail route fail before it could reach the
    store fallback, so this is the assertion that keeps the fallback reachable.
    """
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    state = ClsStudioState.instance() if hasattr(ClsStudioState, "instance") else None
    if state is None:
        from app.core.cls_state import get_state
        state = get_state()
    p = state.image_path("normal", name)
    assert p.name == ClsStudioState.safe_image_name(name)
    assert p.parent == state.images_dir("normal")


def test_traversal_is_still_refused(client, project_id):
    """The mapping keeps "." and "-", so ".." still reaches the real guard."""
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    from app.core.cls_state import get_state
    state = get_state()
    with pytest.raises(PathTraversalError):
        state.image_path("normal", "..")


@pytest.mark.parametrize("name", ["../evil.png", "..\\evil.png", "/etc/passwd"])
def test_separators_cannot_escape_the_tier_directory(client, project_id, name):
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    from app.core.cls_state import get_state
    state = get_state()
    p = state.image_path("normal", name)
    assert p.parent == state.images_dir("normal")


def test_the_thumbnail_route_no_longer_422s_on_a_japanese_name(client, project_id):
    """404 (nothing taught yet) is the right answer; 422 was the bug."""
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    r = client.get(f"/api/v1/bank/images/normal/{REAL_NAME}")
    assert r.status_code == 404, r.text
    assert "invalid image name" not in r.text
