# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Bank runtime package: verdict-config persistence + export/import round trip.

The bank directory is the deployable artifact — these tests build a tiny
real bank on disk (no model needed), save a runtime config, export the
package, re-import it as a new bank and check that features, marks and the
verdict recipe all survive the trip.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np


def _seed_bank(client, project_id: str) -> Path:
    """Select ``project_id`` and write a small real bank into its default bank."""
    from clscore.bank import Bank

    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200
    bank_dir = Path(r.json()["bank_dir"])
    rng = np.random.default_rng(0)
    b = Bank(normal=rng.standard_normal((8, 16)).astype(np.float32))
    b.append("critical", rng.standard_normal((4, 16)).astype(np.float32),
             label="scratch", image_name="x.png")
    b.set_image_annotation("critical", "scratch", "x.png", [1, 3],
                           rects=[{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}])
    b.save(bank_dir)
    # Re-select so the app state loads the seeded bank from disk.
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200
    assert r.json()["bank"]["normal"] == 8
    return bank_dir


def test_runtime_config_null_before_first_save(client, project_id):
    _seed_bank(client, project_id)
    r = client.get("/api/v1/bank/runtime-config")
    assert r.status_code == 200
    assert r.json() is None


def test_runtime_config_round_trip(client, project_id):
    bank_dir = _seed_bank(client, project_id)
    put = client.put("/api/v1/bank/runtime-config", json={
        "topk": 10, "k": 5, "alpha": 500.0, "beta": 0.0,
        "exemplar_alpha": True, "threshold": 38.2,
    })
    assert put.status_code == 200
    assert put.json()["saved_at"]  # server-stamped
    assert (bank_dir / "runtime_config.json").exists()
    got = client.get("/api/v1/bank/runtime-config").json()
    assert got["alpha"] == 500.0
    assert got["threshold"] == 38.2
    assert got["metric"] == "topk_mean"


def test_export_import_round_trip(client, project_id):
    _seed_bank(client, project_id)
    client.put("/api/v1/bank/runtime-config", json={
        "topk": 10, "k": 5, "alpha": 750.0, "beta": 0.0,
        "exemplar_alpha": True, "threshold": 41.0,
    })

    r = client.get("/api/v1/bank/export")
    assert r.status_code == 200
    blob = r.content
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
    assert {"bank.npy", "bank_meta.json", "runtime_config.json",
            "critical/scratch.npy", "critical/scratch.meta.npz"} <= names

    imp = client.post(
        "/api/v1/banks/import",
        files={"archive": ("line-a.clsbank.zip", blob, "application/zip")},
    )
    assert imp.status_code == 200
    body = imp.json()
    assert body["bank_id"] != "default"  # landed as a NEW bank
    assert body["bank"]["normal"] == 8
    assert body["bank"]["critical"] == 4

    # The verdict recipe and the defect marks travelled with the package.
    cfg = client.get("/api/v1/bank/runtime-config").json()
    assert cfg["alpha"] == 750.0 and cfg["threshold"] == 41.0
    from clscore.bank import Bank
    from clscore.incident import SEVERITY_HEAVY
    b2 = Bank.load(Path(body["bank_dir"]))
    sev = b2.critical_meta["scratch"].severity
    assert sev[1] == SEVERITY_HEAVY and sev[3] == SEVERITY_HEAVY

    # Restore the default bank as active for any tests that follow.
    client.post("/api/v1/bank/select", json={"project_id": project_id})


def test_import_rejects_non_bank_zip(client, project_id):
    _seed_bank(client, project_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "not a bank")
    r = client.post(
        "/api/v1/banks/import",
        files={"archive": ("junk.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422


def test_import_rejects_path_traversal(client, project_id):
    _seed_bank(client, project_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("bank.npy", b"x")
        zf.writestr("bank_meta.json", "{}")
        zf.writestr("../evil.txt", "escape")
    r = client.post(
        "/api/v1/banks/import",
        files={"archive": ("evil.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
