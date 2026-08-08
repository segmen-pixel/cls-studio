# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Bank deletion: prune a whole bank, always land the selector on a valid one.

``POST /banks/delete`` mirrors ``banks/create``'s missing inverse. Deleting a
non-active bank leaves the active one alone; deleting the active bank falls
through to a remaining bank; deleting the last bank re-creates a fresh empty
``default`` so the selector is never empty.
"""
from __future__ import annotations

from pathlib import Path


def _bank_ids(client) -> list[str]:
    r = client.get("/api/v1/banks")
    assert r.status_code == 200
    return [b["id"] for b in r.json()["banks"]]


def test_delete_non_active_bank_keeps_active(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    line_b = client.post("/api/v1/banks/create", json={"name": "line-b"}).json()["bank_id"]
    # line-b is active now; delete the (non-active) default.
    r = client.post("/api/v1/banks/delete", json={"bank_id": "default"})
    assert r.status_code == 200
    # Still on line-b, and default is gone from the listing.
    assert r.json()["bank_id"] == line_b
    assert _bank_ids(client) == [line_b]


def test_delete_active_bank_falls_through(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    line_b = client.post("/api/v1/banks/create", json={"name": "line-b"}).json()["bank_id"]
    # line-b is active — deleting it must re-select the remaining default.
    r = client.post("/api/v1/banks/delete", json={"bank_id": line_b})
    assert r.status_code == 200
    assert r.json()["bank_id"] == "default"
    assert _bank_ids(client) == ["default"]


def test_delete_last_bank_recreates_empty_default(client, project_id):
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    bank_dir = Path(r.json()["bank_dir"])
    assert _bank_ids(client) == ["default"]
    # Deleting the only bank lands on a fresh empty default so the selector
    # always has an entry.
    r = client.post("/api/v1/banks/delete", json={"bank_id": "default"})
    assert r.status_code == 200
    assert r.json()["bank_id"] == "default"
    assert r.json()["bank"]["normal"] == 0
    assert _bank_ids(client) == ["default"]
    # The directory was really removed and re-created (not left with stale rows).
    assert not (bank_dir / ".deleted").exists()


def test_delete_removes_directory_from_disk(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    r = client.post("/api/v1/banks/create", json={"name": "ephemeral"})
    eph_dir = Path(r.json()["bank_dir"])
    assert eph_dir.is_dir()
    client.post("/api/v1/banks/delete", json={"bank_id": r.json()["bank_id"]})
    # Nothing held the dir open in-process, so rmtree fully removed it.
    assert not eph_dir.exists()


def test_delete_unknown_bank_404(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    r = client.post("/api/v1/banks/delete", json={"bank_id": "does-not-exist"})
    assert r.status_code == 404
