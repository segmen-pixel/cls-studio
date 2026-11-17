# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Tests for /projects CRUD endpoints."""
from __future__ import annotations


def test_create_project(client):
    resp = client.post("/api/v1/projects", json={"name": "unit-test-project"})
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body
    assert body["name"] == "unit-test-project"
    # cleanup
    client.delete(f"/api/v1/projects/{body['id']}")


def test_list_projects_contains_created(client, project_id):
    resp = client.get("/api/v1/projects")
    assert resp.status_code == 200
    ids = [p["id"] for p in resp.json()]
    assert project_id in ids


def test_get_project(client, project_id):
    resp = client.get(f"/api/v1/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "pytest-tmp"


def test_update_project(client, project_id):
    resp = client.put(f"/api/v1/projects/{project_id}", json={"name": "renamed"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "renamed"
    # Confirm persistence
    resp2 = client.get(f"/api/v1/projects/{project_id}")
    assert resp2.json()["name"] == "renamed"


def test_projects_summary(client, project_id):
    resp = client.get("/api/v1/projects/summary")
    assert resp.status_code == 200
    summaries = resp.json()
    match = [s for s in summaries if s["id"] == project_id]
    assert len(match) == 1
    assert match[0]["image_count"] == 0
    assert match[0]["first_filename"] is None


def test_delete_project_returns_404(client):
    resp = client.post("/api/v1/projects", json={"name": "to-delete"})
    pid = resp.json()["id"]
    del_resp = client.delete(f"/api/v1/projects/{pid}")
    assert del_resp.status_code == 200
    get_resp = client.get(f"/api/v1/projects/{pid}")
    assert get_resp.status_code == 404


def test_rename_project(client, project_id):
    """Rename via PUT should work and update timestamp."""
    import time

    proj_before = client.get(f"/api/v1/projects/{project_id}").json()
    before_ts = proj_before["updated_at"]
    assert proj_before["name"] == "pytest-tmp"
    time.sleep(0.05)  # ensure timestamp differs

    resp = client.put(f"/api/v1/projects/{project_id}", json={"name": "new-name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new-name"

    proj_after = client.get(f"/api/v1/projects/{project_id}").json()
    assert proj_after["name"] == "new-name"
    assert proj_after["updated_at"] >= before_ts


def _age_project_dir(path, seconds: int = 600) -> None:
    """Back-date a dir past the orphan sweep's freshly-modified skip window
    (the sweep ignores dirs touched in the last few minutes because a
    project mid-creation has its dirs on disk before its DB row commits)."""
    import os as _os
    import time as _time

    past = _time.time() - seconds
    _os.utime(path, (past, past))


def test_orphan_cleanup_ignores_tombstoned_dir():
    """Dir with .deleted tombstone must NOT be resurrected on startup sweep.

    Uses DB / filesystem directly to avoid spinning up another TestClient
    (lifespan churn is a known source of RecursionError in this suite).
    """
    import shutil as _shutil
    import uuid

    from sqlmodel import Session, select

    from app.core.paths import PROJECTS_DIR
    from app.core.startup_tasks import _cleanup_orphan_project_dirs
    from app.db import get_engine
    from app.models import Project

    pid = str(uuid.uuid4())
    path = PROJECTS_DIR / pid
    path.mkdir(parents=True, exist_ok=True)
    (path / ".deleted").write_text("", encoding="utf-8")
    _age_project_dir(path)
    try:
        _cleanup_orphan_project_dirs()
        with Session(get_engine()) as session:
            ids = {p.id for p in session.exec(select(Project)).all()}
        assert pid not in ids, "tombstoned dir must not be resurrected in DB"
        assert not path.exists(), "tombstoned dir must be purged"
    finally:
        _shutil.rmtree(path, ignore_errors=True)


def test_orphan_cleanup_purges_stub_dir():
    """Stub dir (no project.json, no content) must NOT be resurrected."""
    import shutil as _shutil
    import uuid

    from sqlmodel import Session, select

    from app.core.paths import PROJECTS_DIR
    from app.core.startup_tasks import _cleanup_orphan_project_dirs
    from app.db import get_engine
    from app.models import Project

    pid = str(uuid.uuid4())
    stub_path = PROJECTS_DIR / pid
    stub_path.mkdir(parents=True, exist_ok=True)
    (stub_path / "classes.json").write_text("{}", encoding="utf-8")
    _age_project_dir(stub_path)
    try:
        _cleanup_orphan_project_dirs()
        with Session(get_engine()) as session:
            ids = {p.id for p in session.exec(select(Project)).all()}
        assert pid not in ids, "stub dir must not be resurrected"
        assert not stub_path.exists(), "stub dir must be purged"
    finally:
        _shutil.rmtree(stub_path, ignore_errors=True)


def test_orphan_cleanup_adopts_imported_project():
    """External project with project.json + content must still be adopted."""
    import json as _json
    import shutil as _shutil
    import uuid

    from sqlmodel import Session, select
    from sqlmodel import delete as _delete

    from app.core.paths import PROJECTS_DIR
    from app.core.startup_tasks import _cleanup_orphan_project_dirs
    from app.db import get_engine
    from app.models import Project

    pid = str(uuid.uuid4())
    imported = PROJECTS_DIR / pid
    imported.mkdir(parents=True, exist_ok=True)
    (imported / "images").mkdir()
    (imported / "project.json").write_text(
        _json.dumps({"name": "imported-one", "description": "via subtree"}),
        encoding="utf-8",
    )
    _age_project_dir(imported)
    try:
        _cleanup_orphan_project_dirs()
        with Session(get_engine()) as session:
            adopted = session.exec(select(Project).where(Project.id == pid)).first()
        assert adopted is not None, "imported project must be adopted"
        assert adopted.name == "imported-one"
    finally:
        with Session(get_engine()) as session:
            session.exec(_delete(Project).where(Project.id == pid))
            session.commit()
        _shutil.rmtree(imported, ignore_errors=True)


def test_reorder_projects_not_shadowed(client):
    """PUT /projects/reorder must be reachable and persist sort_order.

    Regression test: the parameterized PUT /projects/{project_id} route used
    to be registered before the static PUT /projects/reorder route, so
    Starlette matched "reorder" as a project id and returned 404. The static
    route must stay registered first.
    """
    pid_a = client.post("/api/v1/projects", json={"name": "reorder-a"}).json()["id"]
    pid_b = client.post("/api/v1/projects", json={"name": "reorder-b"}).json()["id"]
    try:
        resp = client.put("/api/v1/projects/reorder", json={"order": [pid_b, pid_a]})
        # 404 here means the parameterized route swallowed the request again
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok"}

        projects = {p["id"]: p for p in client.get("/api/v1/projects").json()}
        assert projects[pid_b]["sort_order"] == 0
        assert projects[pid_a]["sort_order"] == 1
    finally:
        client.delete(f"/api/v1/projects/{pid_a}")
        client.delete(f"/api/v1/projects/{pid_b}")
