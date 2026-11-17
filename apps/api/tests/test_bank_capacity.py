# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Runtime memory-bank size budget (small / medium / large).

The normal tensor is GPU-resident for scoring, so its total patch count sets
the resident VRAM floor. The capacity tier caps that at teach time — and it is
append-only, so an over-budget teach is truncated/rejected but existing rows
are never evicted.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.core.cls_state import get_state
from app.routers import bank as bank_mod


def _png(color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_capacity_defaults_to_medium_and_persists(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})

    body = client.get("/api/v1/bank/capacity").json()
    assert body["capacity"] == "medium"
    assert body["ceiling"] == bank_mod._DEFAULT_CAPACITY_CEILINGS["medium"]
    assert body["normal"] == 0 and body["pct"] == 0.0

    r = client.put("/api/v1/bank/capacity", json={"capacity": "small"})
    assert r.status_code == 200
    assert r.json()["capacity"] == "small"
    assert r.json()["ceiling"] == bank_mod._DEFAULT_CAPACITY_CEILINGS["small"]

    # Persisted across reads.
    assert client.get("/api/v1/bank/capacity").json()["capacity"] == "small"


def test_put_capacity_preserves_verdict_recipe(client, project_id):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    client.put("/api/v1/bank/runtime-config", json={"topk": 7, "k": 3, "threshold": 1.5})

    client.put("/api/v1/bank/capacity", json={"capacity": "large"})

    cfg = client.get("/api/v1/bank/runtime-config").json()
    assert cfg["bank_capacity"] == "large"
    # The verdict recipe rode along untouched.
    assert cfg["topk"] == 7 and cfg["k"] == 3 and cfg["threshold"] == 1.5


def test_teach_stops_at_capacity_ceiling(client, project_id, monkeypatch):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    client.put("/api/v1/bank/capacity", json={"capacity": "small"})

    # Tiny ceiling + no per-image cap so two synthetic teaches straddle it.
    monkeypatch.setenv("CLS_CAPACITY_SMALL", "100")
    monkeypatch.setenv("CLS_MAX_PATCHES_PER_IMAGE", "0")

    state = get_state()
    rng = np.random.default_rng(0)

    def _fake_extract(model, arr, device, max_batch=0):
        return rng.standard_normal((60, 32)).astype(np.float16)

    monkeypatch.setattr(bank_mod, "extract_image_features_for_bank", _fake_extract)
    monkeypatch.setattr(state, "ensure_model", lambda: (object(), "cpu", None))

    def _teach_one():
        return client.post(
            "/api/v1/bank/append/normal",
            files=[("image", ("x.png", _png(), "image/png"))],
        )

    r1 = _teach_one()
    assert r1.status_code == 200
    assert r1.json()["bank"]["normal"] == 60           # first image: all 60 rows fit

    r2 = _teach_one()
    assert r2.status_code == 200
    assert r2.json()["bank"]["normal"] == 100          # second: clipped to the 40-row headroom
    assert r2.json()["appended_patches"] == 40

    r3 = _teach_one()
    assert r3.status_code == 409                        # bank full → rejected, no orphan rows
    assert client.get("/api/v1/bank/capacity").json()["normal"] == 100
