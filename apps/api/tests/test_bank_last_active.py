# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Project re-selection without a bank id must keep the last-active bank.

Tab activations re-select the project with ``bank_id: null``; before the
last-active marker this silently snapped the active bank back to the
alphabetically first one, so verdict configs and exports could target a
different bank than the one shown in the UI.
"""
from __future__ import annotations


def test_select_without_bank_id_keeps_last_active(client, project_id):
    # First bind lands on the empty default bank.
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200
    assert r.json()["bank_id"] == "default"

    # Create a second bank — it becomes active.
    r = client.post("/api/v1/banks/create", json={"name": "line-b"})
    assert r.status_code == 200
    line_b = r.json()["bank_id"]
    assert line_b != "default"

    # Re-selecting the project WITHOUT a bank id (what tab switches do)
    # must stay on line-b, not snap back to the first bank.
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200
    assert r.json()["bank_id"] == line_b

    # An explicit choice still wins and is remembered afterwards.
    r = client.post("/api/v1/bank/select", json={"project_id": project_id, "bank_id": "default"})
    assert r.status_code == 200
    assert r.json()["bank_id"] == "default"
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.json()["bank_id"] == "default"


def test_last_active_marker_survives_bank_deletion(client, project_id):
    """A stale marker (bank deleted on disk) falls back to the first bank."""
    from pathlib import Path

    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    banks_root = Path(r.json()["bank_dir"]).parent
    r = client.post("/api/v1/banks/create", json={"name": "ephemeral"})
    assert r.status_code == 200
    # Simulate an out-of-band deletion of the remembered bank.
    import shutil
    shutil.rmtree(Path(r.json()["bank_dir"]))
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200
    assert r.json()["bank_id"] == "default"
    assert (banks_root / ".last_active").read_text(encoding="utf-8").strip() == "default"
