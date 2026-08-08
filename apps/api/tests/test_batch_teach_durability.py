# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Bulk teach must persist per group, not once at the end of the batch.

2026-07-17: a native crash between the last taught image and the single
end-of-batch save orphaned all 24 images of the batch (thumbnails on disk,
rows gone). Saving per group bounds the loss to one group.
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


def test_append_batch_saves_once_per_group(client, project_id, monkeypatch):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    state = get_state()
    rng = np.random.default_rng(0)

    def _fake_extract_batched(model, arrs, device, max_batch=0):
        # Generator, matching the streaming contract the teach path consumes:
        # each image's features arrive as soon as its windows are through.
        for i, _ in enumerate(arrs):
            yield i, rng.standard_normal((5, 32)).astype(np.float16)

    monkeypatch.setattr(bank_mod, "iter_images_features_batched", _fake_extract_batched)
    monkeypatch.setattr(state, "ensure_model", lambda: (object(), "cpu", None))
    monkeypatch.setattr(bank_mod, "_BATCH_TEACH_GROUP", 2)

    saves: list[int] = []
    orig_save = state.save_bank

    def _spy_save(parts=None):
        # Record how many rows are durable at each save: every group's rows
        # must hit disk before the next group starts.
        saves.append(int(state.bank.normal.shape[0]))
        return orig_save(parts=parts)

    monkeypatch.setattr(state, "save_bank", _spy_save)

    files = [("images", (f"img{i}.png", _png((i * 40, 0, 0)), "image/png")) for i in range(5)]
    r = client.post("/api/v1/bank/append_batch/normal", files=files)
    assert r.status_code == 200
    assert r.json()["appended_patches"] == 25

    # 5 images / group-of-2 = 3 groups → 3 saves, at 10, 20 and 25 rows.
    assert saves == [10, 20, 25]
