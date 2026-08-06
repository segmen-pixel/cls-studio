# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Project deletion vs the process-wide active bank (2026-07-17 incident).

DELETE /projects/{id} used to leave ClsStudioState pointing into the removed
tree: teaches kept "succeeding" into the deleted directory and the next
startup orphan purge destroyed everything taught in the meantime. These
tests pin the whole defence:

  - deleting the active project deactivates the in-memory bank,
  - bank routes 409 cleanly afterwards instead of resurrecting the tree,
  - re-selecting a deleted project id 404s (DB check),
  - the startup purge adopts a banks-bearing dir instead of purging it.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path


def _age_dir(d: Path, seconds: int = 600) -> None:
    """Back-date a dir so the orphan sweep's freshly-modified skip ignores it."""
    past = time.time() - seconds
    os.utime(d, (past, past))


def _select(client, pid: str):
    return client.post("/api/v1/bank/select", json={"project_id": pid})


def test_delete_active_project_deactivates_bank(client):
    pid = client.post("/api/v1/projects", json={"name": "doomed"}).json()["id"]
    r = _select(client, pid)
    assert r.status_code == 200
    project_root = Path(r.json()["bank_dir"]).parent.parent

    assert client.delete(f"/api/v1/projects/{pid}").status_code == 200
    # The tree is gone and no bank route may quietly re-create it.
    assert not project_root.exists()
    for resp in (
        client.get("/api/v1/bank"),
        client.post("/api/v1/banks/create", json={"name": "ghost"}),
        client.post("/api/v1/bank/save"),
    ):
        assert resp.status_code == 409, resp.text
    assert not project_root.exists()


def test_delete_non_active_project_keeps_bank(client):
    keep = client.post("/api/v1/projects", json={"name": "keep"}).json()["id"]
    doomed = client.post("/api/v1/projects", json={"name": "doomed2"}).json()["id"]
    assert _select(client, keep).status_code == 200
    assert client.delete(f"/api/v1/projects/{doomed}").status_code == 200
    # Unrelated delete leaves the active bank alone.
    assert client.get("/api/v1/bank").status_code == 200
    client.delete(f"/api/v1/projects/{keep}")


def test_select_deleted_project_404(client):
    pid = client.post("/api/v1/projects", json={"name": "gone"}).json()["id"]
    assert client.delete(f"/api/v1/projects/{pid}").status_code == 200
    # A stale client (sessionStorage, second browser) re-selecting the id
    # must get a 404, not a silently re-created ghost tree.
    assert _select(client, pid).status_code == 404


def test_external_dir_removal_turns_routes_into_409(client):
    pid = client.post("/api/v1/projects", json={"name": "yanked"}).json()["id"]
    r = _select(client, pid)
    assert r.status_code == 200
    project_root = Path(r.json()["bank_dir"]).parent.parent
    # Simulate the race: the directory vanishes underneath the active state
    # (delete from another process, manual cleanup, ...).
    shutil.rmtree(project_root)
    for resp in (
        client.get("/api/v1/bank"),
        client.post("/api/v1/banks/create", json={"name": "ghost"}),
    ):
        assert resp.status_code == 409, resp.text
    assert not project_root.exists()
    client.delete(f"/api/v1/projects/{pid}")


def test_bank_binding_mismatch_409(client):
    """X-Bank-Binding pins a mutation to the (project, bank) the client saw;
    a mismatch means another LAN client re-bound the global active bank and
    the write would land in the wrong bank."""
    pid = client.post("/api/v1/projects", json={"name": "bind"}).json()["id"]
    r = _select(client, pid)
    assert r.status_code == 200
    good = f"{pid}/{r.json()['bank_id']}"
    wrong = f"{pid}/other-bank"
    r = client.post(
        "/api/v1/bank/clear/critical", headers={"X-Bank-Binding": wrong},
    )
    assert r.status_code == 409
    r = client.post(
        "/api/v1/bank/clear/critical", headers={"X-Bank-Binding": good},
    )
    assert r.status_code == 200
    # Unbound requests (older clients, curl) keep working.
    assert client.post("/api/v1/bank/clear/critical").status_code == 200
    client.delete(f"/api/v1/projects/{pid}")


def test_clear_normal_tier_422(client):
    pid = client.post("/api/v1/projects", json={"name": "clr"}).json()["id"]
    assert _select(client, pid).status_code == 200
    assert client.post("/api/v1/bank/clear/normal").status_code == 422
    client.delete(f"/api/v1/projects/{pid}")


def test_orphan_purge_adopts_banks_bearing_dir(client):
    """A DB-orphan dir holding taught banks is user data — adopt, never purge."""
    from app.core.config import PROJECTS_DIR
    from app.core.startup_tasks import _cleanup_orphan_project_dirs

    ghost = Path(PROJECTS_DIR) / str(uuid.uuid4())
    (ghost / "banks" / "default").mkdir(parents=True)
    (ghost / "banks" / "default" / "bank.npy").write_bytes(b"stub")
    _age_dir(ghost)
    # No project.json, no tombstone — pre-fix this dir was purged.
    _cleanup_orphan_project_dirs()
    assert ghost.exists(), "banks-bearing orphan dir must be adopted, not purged"
    # Adopted into the DB — a later purge pass must leave it alone too.
    _cleanup_orphan_project_dirs()
    assert ghost.exists()
    # Clean up the adopted row + dir so other tests see a stable world.
    pid = ghost.name
    client.delete(f"/api/v1/projects/{pid}")


def test_orphan_purge_still_removes_tombstoned_dir(client):
    from app.core.config import PROJECTS_DIR
    from app.core.startup_tasks import _cleanup_orphan_project_dirs

    ghost = Path(PROJECTS_DIR) / str(uuid.uuid4())
    (ghost / "banks").mkdir(parents=True)
    (ghost / ".deleted").write_text("", encoding="utf-8")
    _age_dir(ghost)
    _cleanup_orphan_project_dirs()
    assert not ghost.exists(), "tombstoned remnants must still be purged"
