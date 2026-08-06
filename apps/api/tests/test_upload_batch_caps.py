# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Aggregate caps for multi-file uploads (batch teach / staging drop).

MAX_UPLOAD_BYTES bounds ONE multipart part, so before these caps a single
request could ingest N x 200 MB into RAM (``append_batch`` holds every image's
bytes) or onto disk (staging). The caps are ``MAX_UPLOAD_FILES`` /
``MAX_UPLOAD_TOTAL_BYTES`` in ``app.core.config``, enforced by
``app.core.security.check_upload_batch`` with the same 413 shape as the
per-file cap.

They bound the *ingested upload bytes*, not the decoded footprint: the total is
checked after each part is buffered (peak = total cap + one part) and the
ndarrays ``append_batch`` decodes from those bytes are several times larger and
uncounted. The reverse proxy body limit is the hard ceiling (docs/deployment.md).
"""
from __future__ import annotations

import io
import os

from PIL import Image

from app.core import config as config_mod
from app.core import security as security_mod
from app.routers import bank as bank_mod
from app.routers import staging as staging_mod


def _png(color=(0, 128, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_defaults_are_sane():
    """Defaults must not fire for the documented workload ("dozens of images").

    These are the values SECURITY.md promises, so they are asserted — but only
    when the operator has not overridden them in the environment.
    """
    if "CLS_MAX_UPLOAD_FILES" not in os.environ:
        assert config_mod.MAX_UPLOAD_FILES == 1024
    if "CLS_MAX_UPLOAD_TOTAL_MB" not in os.environ:
        assert config_mod.MAX_UPLOAD_TOTAL_BYTES == 2048 * 1024 * 1024
    # A single part is still bounded independently.
    assert config_mod.MAX_UPLOAD_BYTES == 200 * 1024 * 1024


def test_both_multi_file_routes_share_the_guard():
    """Both multi-part routes must resolve the public alias, not a local copy.

    ``check_upload_batch`` is exported as an alias of ``_check_upload_batch``;
    a router importing the underscored name (or defining its own) would drop
    silently out of the cap, and the monkeypatching below would stop biting.
    """
    assert staging_mod.check_upload_batch is security_mod.check_upload_batch
    assert bank_mod.check_upload_batch is security_mod.check_upload_batch


def test_env_int_falls_back_instead_of_killing_startup(monkeypatch):
    """A bad env value must not raise at import time (the API would not start)."""
    monkeypatch.setenv("CLS_TEST_CAP", "not-a-number")
    assert config_mod._env_int("CLS_TEST_CAP", 7) == 7
    monkeypatch.setenv("CLS_TEST_CAP", "0")
    assert config_mod._env_int("CLS_TEST_CAP", 7) == 1  # clamped, never 0
    monkeypatch.setenv("CLS_TEST_CAP", "12")
    assert config_mod._env_int("CLS_TEST_CAP", 7) == 12
    monkeypatch.delenv("CLS_TEST_CAP")
    assert config_mod._env_int("CLS_TEST_CAP", 7) == 7


def test_staging_upload_rejects_too_many_files(client, project_id, monkeypatch):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    monkeypatch.setattr(security_mod, "MAX_UPLOAD_FILES", 2)

    files = [("files", (f"many{i}.png", _png(), "image/png")) for i in range(3)]
    r = client.post("/api/v1/bank/staging/upload", files=files)
    assert r.status_code == 413
    assert "too many files" in r.json()["detail"]

    # The count is checked before anything is read, so nothing reaches disk.
    assert client.get("/api/v1/bank/staging").json()["items"] == []


def test_staging_upload_rejects_oversized_batch_but_keeps_what_landed(
    client, project_id, monkeypatch,
):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    png = _png()
    # Room for exactly one part: the second one trips the running total.
    monkeypatch.setattr(security_mod, "MAX_UPLOAD_TOTAL_BYTES", len(png))

    files = [("files", (f"big{i}.png", png, "image/png")) for i in range(3)]
    r = client.post("/api/v1/bank/staging/upload", files=files)
    assert r.status_code == 413
    assert "upload batch too large" in r.json()["detail"]

    # Files already written are flushed to staging.json before the rejection —
    # a listing (= what a reload does) must not lag behind the disk.
    names = [it["name"] for it in client.get("/api/v1/bank/staging").json()["items"]]
    assert names == ["big0.png"]


def test_append_batch_rejects_too_many_files(client, project_id, monkeypatch):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    monkeypatch.setattr(security_mod, "MAX_UPLOAD_FILES", 2)

    files = [("images", (f"teach{i}.png", _png((i * 40, 0, 0)), "image/png")) for i in range(3)]
    r = client.post("/api/v1/bank/append_batch/normal", files=files)
    assert r.status_code == 413
    assert "too many files" in r.json()["detail"]

    # Rejected before the model is touched: the bank stays empty.
    assert client.get("/api/v1/bank").json()["normal"] == 0


def test_append_batch_rejects_oversized_batch(client, project_id, monkeypatch):
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    png = _png()
    monkeypatch.setattr(security_mod, "MAX_UPLOAD_TOTAL_BYTES", len(png))

    files = [("images", (f"heavy{i}.png", png, "image/png")) for i in range(3)]
    r = client.post("/api/v1/bank/append_batch/normal", files=files)
    assert r.status_code == 413
    assert "upload batch too large" in r.json()["detail"]
    assert client.get("/api/v1/bank").json()["normal"] == 0
