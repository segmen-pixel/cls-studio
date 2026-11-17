# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The project card's thumbnail, for a bank built the current way.

``_first_bank_image`` looked in ``_images/<tier>/`` and ``_staging/``. Only the
retired ``/bank/append`` path ever wrote ``_images/``, so a project imported
into the store, labelled and assembled had neither — and its card said "no
image" while holding several hundred. On a real 635-image project every other
card on the screen had a thumbnail and that one did not.

The viewer already had this fallback (``get_bank_image`` serves the store's
copy when the bank has no ``_images/``); the card did not.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app.routers import projects as projects_mod

API = "/api/v1"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(3, 4, 5)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def bank_dir(client, project_id) -> Path:
    r = client.post(f"{API}/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    return Path(r.json()["bank_dir"])


# The layout-constant check that used to live here now covers all eleven names
# in test_bank_layout_constants.py. This one pinned IMAGES_SUBDIR and not
# STORE_SUBDIR, so renaming half the path broke the card with a green suite.


def test_a_store_only_bank_still_has_a_card_thumbnail(client, project_id, bank_dir):
    """No _images/, no _staging/ — exactly the shape assemble-from-store leaves."""
    assert not (bank_dir / "_images").exists()
    img_dir = bank_dir / "store" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "000000.png").write_bytes(_png())

    found = projects_mod._first_bank_image(project_id)
    assert found is not None, "the store's own copy is the card's last resort"
    assert found.name == "000000.png"

    r = client.get(f"{API}/projects/{project_id}/bank-thumbnail")
    assert r.status_code == 200, r.text
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def _summary_row(client, project_id: str) -> dict:
    rows = client.get(f"{API}/projects/summary").json()
    rows = rows if isinstance(rows, list) else rows.get("projects", [])
    return next(p for p in rows if p["id"] == project_id)


def test_the_summary_reports_the_thumbnail_for_a_store_only_bank(client, project_id, bank_dir):
    img_dir = bank_dir / "store" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "000000.png").write_bytes(_png())

    assert _summary_row(client, project_id)["has_bank_thumbnail"] is True


def test_the_summary_reports_the_count_for_a_store_only_bank(client, project_id, bank_dir):
    """The half this file never asserted, and the half that was broken.

    The thumbnail arm learned about the store; the counting arm did not, so
    this card painted a picture above the words "0 images".
    """
    store_dir = bank_dir / "store"
    (store_dir / "img").mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(3):
        (store_dir / "img" / f"{i:06d}.png").write_bytes(_png())
        entries.append(
            {"id": f"{i:06d}", "name": f"a{i}.png", "image_ref": f"store/img/{i:06d}.png"}
        )
    (store_dir / "store_index.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8"
    )

    row = _summary_row(client, project_id)
    assert row["has_bank_thumbnail"] is True
    assert row["image_count"] == 3, "an ingested project is not an empty one"


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"])
def test_every_extension_the_store_writer_emits_can_be_a_thumbnail(
    client, project_id, bank_dir, ext: str
):
    """.webp was missing from the card's list, so a WebP project had no thumbnail."""
    img_dir = bank_dir / "store" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"000000{ext}").write_bytes(_png())

    found = projects_mod._first_bank_image(project_id)
    assert found is not None and found.suffix == ext


def test_a_taught_copy_still_wins_over_the_store(client, project_id, bank_dir):
    """Order matters: _images/ is tier-ordered, so it names a normal image.

    The store's first blob is whatever was ingested first, which may be a
    defect — a worse choice for a card, so the store stays the last resort.
    """
    taught = bank_dir / "_images" / "normal"
    taught.mkdir(parents=True, exist_ok=True)
    (taught / "aaa.png").write_bytes(_png())
    img_dir = bank_dir / "store" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "000000.png").write_bytes(_png())

    found = projects_mod._first_bank_image(project_id)
    assert found is not None and found.name == "aaa.png"


def test_an_empty_project_has_no_thumbnail(client, project_id, bank_dir):
    assert projects_mod._first_bank_image(project_id) is None
    r = client.get(f"{API}/projects/{project_id}/bank-thumbnail")
    assert r.status_code == 404, r.text
