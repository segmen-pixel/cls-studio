# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Whole-project export / import.

/bank/export only ever packaged ONE bank of one project, so a project's
images, masks, inspection log and any second bank could not leave the machine
at all — there was no way to move a project or to back one up whole.

The import always creates a NEW project: it must never overwrite an existing
one, and the id inside the package (the exporting machine's) must not survive
into the row that now owns the copy.
"""
from __future__ import annotations

import io
import json
import zipfile

from app.core.paths import project_dir


def _export(client, pid: str) -> zipfile.ZipFile:
    r = client.get(f"/api/v1/projects/{pid}/export")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    return zipfile.ZipFile(io.BytesIO(r.content))


def test_export_carries_the_whole_project(client, project_id):
    root = project_dir(project_id)
    (root / "banks" / "default").mkdir(parents=True, exist_ok=True)
    (root / "banks" / "default" / "bank_meta.json").write_text("{}", encoding="utf-8")
    (root / "banks" / "second").mkdir(parents=True, exist_ok=True)
    (root / "banks" / "second" / "bank_meta.json").write_text("{}", encoding="utf-8")
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "images" / "a.png").write_bytes(b"not-really-a-png")
    (root / "leftover.tmp").write_bytes(b"in-flight")

    with _export(client, project_id) as zf:
        names = set(zf.namelist())
    assert "project.json" in names
    assert "banks/default/bank_meta.json" in names
    assert "banks/second/bank_meta.json" in names, "every bank, not just the active one"
    assert "images/a.png" in names
    assert "leftover.tmp" not in names, "atomic-write leftovers are not data"


def test_export_unknown_project_is_404(client):
    assert client.get("/api/v1/projects/does-not-exist/export").status_code == 404


def test_round_trip_creates_a_separate_project(client, project_id):
    root = project_dir(project_id)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "images" / "keep.png").write_bytes(b"payload")

    r = client.get(f"/api/v1/projects/{project_id}/export")
    assert r.status_code == 200
    imported = client.post(
        "/api/v1/projects/import",
        files={"archive": ("p.clsproj.zip", r.content, "application/zip")},
        data={"name": "restored"},
    )
    assert imported.status_code == 200, imported.text
    body = imported.json()
    assert body["name"] == "restored"
    assert body["id"] != project_id, "import must not reuse the exporting machine's id"

    # The original is untouched and both are listed.
    ids = {p["id"] for p in client.get("/api/v1/projects").json()}
    assert {project_id, body["id"]} <= ids

    new_root = project_dir(body["id"])
    assert (new_root / "images" / "keep.png").read_bytes() == b"payload"
    # ...and the on-disk project.json agrees with the row that owns it now.
    on_disk = json.loads((new_root / "project.json").read_text(encoding="utf-8"))
    assert on_disk["id"] == body["id"]
    assert on_disk["name"] == "restored"


def test_import_keeps_the_packaged_name_when_none_is_given(client, project_id):
    r = client.get(f"/api/v1/projects/{project_id}/export")
    imported = client.post(
        "/api/v1/projects/import",
        files={"archive": ("p.zip", r.content, "application/zip")},
    )
    assert imported.status_code == 200
    original = client.get(f"/api/v1/projects/{project_id}").json()
    assert imported.json()["name"] == original["name"]


def test_import_rejects_something_that_is_not_a_project(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("random.txt", "hello")
    r = client.post(
        "/api/v1/projects/import",
        files={"archive": ("x.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
    assert "project.json" in r.text


def test_import_rejects_a_path_escaping_the_project_dir(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("project.json", json.dumps({"name": "evil"}))
        zf.writestr("../escaped.txt", "nope")
    r = client.post(
        "/api/v1/projects/import",
        files={"archive": ("x.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
    assert "unsafe path" in r.text


def test_failed_import_leaves_nothing_behind(client):
    before = {p["id"] for p in client.get("/api/v1/projects").json()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("project.json", "{not json")
    r = client.post(
        "/api/v1/projects/import",
        files={"archive": ("x.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
    after = {p["id"] for p in client.get("/api/v1/projects").json()}
    assert after == before, "a rejected package must not leave a half-made project"
