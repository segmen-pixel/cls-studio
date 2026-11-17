# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Server-side staging (upload-on-drop).

2026-07-18: dropped images lived only in the browser and evaporated on every
reload. They now land in ``<bank>/_staging/`` immediately, labels persist
next to them, a fresh listing (= reload) restores everything, and teach
consumes only the files that actually reached the bank.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.core.cls_state import get_state
from app.routers import bank as bank_mod


def _png(color=(0, 128, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_staging_upload_label_restore_delete(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    files = [("files", (f"s{i}.png", _png(), "image/png")) for i in range(3)]
    r = client.post("/api/v1/bank/staging/upload", files=files)
    assert r.status_code == 200
    names = [it["name"] for it in r.json()["items"]]
    assert len(names) == 3

    # Labels persist server-side (they must survive a reload too).
    r = client.post("/api/v1/bank/staging/label", json={"names": names[:2], "tier": "normal"})
    assert r.status_code == 200
    assert [it["tier"] for it in r.json()["items"]] == ["normal", "normal", None]

    # A fresh listing (= what a reload does) returns the same state.
    r = client.get("/api/v1/bank/staging")
    assert len(r.json()["items"]) == 3
    assert [it["tier"] for it in r.json()["items"]] == ["normal", "normal", None]

    # The staged file itself streams back for the thumbnail.
    assert client.get(f"/api/v1/bank/staging/file/{names[0]}").status_code == 200

    r = client.post("/api/v1/bank/staging/delete", json={"names": [names[2]]})
    assert len(r.json()["items"]) == 2


def test_summary_counts_bank_and_staged_images(client, project_id):
    """The project card must reflect drops immediately: staged files count
    into image_count (labeled ones into mask_count) and provide the
    thumbnail fallback — the annotate-dataset counts are always 0 for
    cls-studio projects (2026-07-18 report: '0 images' after teaching)."""
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    files = [("files", (f"c{i}.png", _png(), "image/png")) for i in range(2)]
    assert client.post("/api/v1/bank/staging/upload", files=files).status_code == 200
    assert client.post(
        "/api/v1/bank/staging/label", json={"names": ["c0.png"], "tier": "critical"},
    ).status_code == 200

    rows = client.get("/api/v1/projects/summary").json()
    row = next(r for r in rows if r["id"] == project_id)
    assert row["image_count"] == 2, "staged files must count as project images"
    assert row["mask_count"] == 1, "labeled staged files must count as labeled"
    assert row["has_bank_thumbnail"] is True, "staged file must provide the card thumbnail"
    assert client.get(f"/api/v1/projects/{project_id}/bank-thumbnail").status_code == 200


def test_staged_teach_consumes_only_taught(client, project_id, monkeypatch):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    state = get_state()
    rng = np.random.default_rng(0)
    # The teach path streams per image (one image's features live at a time)
    # rather than collecting the group, so the stub is a generator too.
    monkeypatch.setattr(
        bank_mod, "iter_images_features_batched",
        lambda model, arrs, device, max_batch=0: (
            (i, rng.standard_normal((4, 32)).astype(np.float16)) for i, _ in enumerate(arrs)
        ),
    )
    monkeypatch.setattr(state, "ensure_model", lambda: (object(), "cpu", None))

    files = [("files", (f"t{i}.png", _png((i * 30, 0, 0)), "image/png")) for i in range(3)]
    names = [it["name"] for it in client.post("/api/v1/bank/staging/upload", files=files).json()["items"]]

    r = client.post(
        "/api/v1/bank/staging/teach",
        json={"names": names + ["ghost.png"], "tier": "normal"},
    )
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["taught"]) == sorted(names)
    # Unknown / unreadable names are reported failed, never invented as taught.
    assert body["failed"] == ["ghost.png"]
    assert body["appended_patches"] == 12
    assert body["bank"]["normal"] == 12

    # Taught files are consumed; nothing lingers to be re-taught by accident.
    assert client.get("/api/v1/bank/staging").json()["items"] == []
