# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Persisted inspection results: log round-trip, file serving, cap pruning.

The Operator tab restores these after a browser reload — results that
finished server-side must survive the page's death.
"""
from __future__ import annotations

from app.core.cls_state import get_state
from app.routers.inspections import append_inspection


def _append(name: str, topk: float = 1.5) -> None:
    append_inspection(
        get_state(),
        name=name, topk_score=topk, max_score=2.0, p99_score=1.8,
        n_exemplar_rows=0, alpha=0.0, server_ms=12.3,
        orig_jpeg=b"jpeg:" + name.encode(), heat_png=b"png:" + name.encode(),
    )


def test_inspection_log_roundtrip(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    _append("a.tiff")

    r = client.get("/api/v1/inspections")
    assert r.status_code == 200
    es = r.json()["entries"]
    assert len(es) == 1
    assert es[0]["name"] == "a.tiff"
    assert es[0]["topk_score"] == 1.5

    fr = client.get(f"/api/v1/inspections/file/{es[0]['orig']}")
    assert fr.status_code == 200 and fr.content == b"jpeg:a.tiff"
    fr = client.get(f"/api/v1/inspections/file/{es[0]['heat']}")
    assert fr.status_code == 200 and fr.content == b"png:a.tiff"

    # Traversal / junk names are rejected, not resolved.
    assert client.get("/api/v1/inspections/file/..%2Flog.json").status_code in (400, 404)

    r = client.delete("/api/v1/inspections")
    assert r.status_code == 200 and r.json()["entries"] == []
    assert client.get("/api/v1/inspections").json()["entries"] == []


def test_delete_single_inspection(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    _append("keep.png", topk=1.0)
    _append("drop.png", topk=2.0)
    es = client.get("/api/v1/inspections").json()["entries"]
    victim = next(e for e in es if e["name"] == "drop.png")

    r = client.delete(f"/api/v1/inspections/{victim['id']}")
    assert r.status_code == 200
    assert [e["name"] for e in r.json()["entries"]] == ["keep.png"]
    # Its files are gone; the survivor's stay.
    d = get_state().inspections_dir()
    assert not (d / victim["orig"]).exists() and not (d / victim["heat"]).exists()
    survivor = r.json()["entries"][0]
    assert (d / survivor["orig"]).exists()
    # Idempotent: deleting again is a 200 no-op; junk ids are rejected.
    assert client.delete(f"/api/v1/inspections/{victim['id']}").status_code == 200
    assert client.delete("/api/v1/inspections/not-a-real-id!").status_code == 400


def test_inspection_log_cap_prunes_oldest(client, project_id, monkeypatch):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    monkeypatch.setenv("CLS_INSPECTION_LOG_CAP", "3")
    for i in range(5):
        _append(f"img{i}.png", topk=float(i))

    es = client.get("/api/v1/inspections").json()["entries"]
    assert [e["name"] for e in es] == ["img2.png", "img3.png", "img4.png"]

    # The pruned entries' files are gone from disk too.
    d = get_state().inspections_dir()
    kept = {e["orig"] for e in es} | {e["heat"] for e in es}
    on_disk = {p.name for p in d.iterdir() if p.name != "log.json"}
    assert on_disk == kept
