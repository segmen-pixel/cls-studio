# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""``POST /store/ingest_zip`` — a zip of images as the transport for an ingest.

A zip is a hostile input in a way a multipart part is not: the member names are
attacker-chosen and the central directory can lie about the sizes. Every guard
here therefore runs *before* a single member is read, and before the backbone is
touched at all — which is why the guard tests below need no model stub, and why
a 413/422 from one of them proves both the status code and that nothing
expensive happened.

The happy paths do need a stub. ``store.py`` imports ``ingest_decoded`` by name,
so replacing it on the router module bites; the stub writes entries through
``FeatureStore.add`` exactly as ``test_store_routes._seed_store`` does, so the
suite still never loads DINOv2.

The zip-slip and drive-letter members are built with an explicit ``ZipInfo`` and
``ZIP_STORED`` — ``zipfile`` normalises hostile names away when it compresses a
path you hand it, so a test that writes them the easy way silently tests
nothing.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core import security as security_mod
from app.routers import bank as bank_mod
from app.routers import projects as projects_mod
from app.routers import store as store_mod
from clscore.store import FeatureStore, StoreMeta

API = "/api/v1"
DIM = 16


@pytest.fixture
def bank_dir(client, project_id) -> Path:
    r = client.post(f"{API}/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    return Path(r.json()["bank_dir"])


def _png(color=(0, 128, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _zip(members: dict[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, data in members.items():
            # An explicit ZipInfo keeps a hostile name verbatim; zf.writestr
            # with a plain string would let zipfile sanitise it first. The
            # compress_type has to be set on the info as well -- a fresh
            # ZipInfo defaults to STORED and silently ignores the ZipFile's
            # compression, which is how the zip-bomb test first shipped a 4 MB
            # "compressed" member and proved nothing.
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            zf.writestr(info, data)
    return buf.getvalue()


def _post(client, blob: bytes, **kw):
    return client.post(
        f"{API}/store/ingest_zip",
        files={"archive": ("images.zip", blob, "application/zip")},
        **kw,
    )


@pytest.fixture
def fake_ingest(monkeypatch, bank_dir):
    """Stand in for the backbone; record what the route decoded and handed on."""
    seen: list[str] = []

    def _fake(state, store, decoded, *, max_patches=None):
        rng = np.random.default_rng(0)
        entries = []
        for name, _data, _arr in decoded:
            seen.append(name)
            entries.append(
                store.add(rng.random((4, DIM), dtype=np.float32), name=name,
                          grid_rows=4, height=16, width=16)
            )
        store.save_index()
        return entries

    monkeypatch.setattr(store_mod, "ingest_decoded", _fake)
    # A store the stub can append to, with a dim the route never re-derives.
    FeatureStore(bank_dir / "store", meta=StoreMeta(dim=DIM)).save_index()
    return seen


def _names(client) -> list[str]:
    return [i["name"] for i in client.get(f"{API}/store").json()["images"]]


# ---- happy paths -----------------------------------------------------------


def test_zip_import_ingests_every_image_in_the_archive(client, bank_dir, fake_ingest):
    blob = _zip({"a.png": _png(), "sub/b.png": _png((1, 2, 3)),
                 "sub/deep/c.png": _png((4, 5, 6))})
    r = _post(client, blob)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ingested"] == 3
    assert body["failed"] == []
    assert sorted(_names(client)) == ["a.png", "b.png", "c.png"]


def test_zip_import_flattens_nested_directories_to_basenames(client, bank_dir, fake_ingest):
    """Two lots, one filename. Both land, both keep the basename, ids differ.

    Renaming collisions apart (seg-studio appends _1/_2) would be wrong here:
    store ids are opaque and two entries are allowed to share a name.
    """
    blob = _zip({"lot1/a.png": _png(), "lot2/a.png": _png((9, 9, 9))})
    r = _post(client, blob)
    assert r.status_code == 200, r.text
    assert r.json()["ingested"] == 2
    images = client.get(f"{API}/store").json()["images"]
    assert [i["name"] for i in images] == ["a.png", "a.png"]
    assert len({i["id"] for i in images}) == 2


def test_zip_import_skips_macosx_and_dotfiles(client, bank_dir, fake_ingest):
    """Skipped, not failed — a mac zip carries one AppleDouble per image."""
    blob = _zip({
        "__MACOSX/._a.png": b"resource fork",
        ".DS_Store": b"junk",
        ".hidden.png": b"junk",
        "a.png": _png(),
    })
    r = _post(client, blob)
    assert r.status_code == 200, r.text
    assert r.json()["ingested"] == 1
    assert r.json()["failed"] == []


def test_zip_import_skips_non_image_members(client, bank_dir, fake_ingest):
    blob = _zip({"readme.txt": b"hello", "labels.json": b"{}", "a.png": _png()})
    r = _post(client, blob)
    assert r.status_code == 200, r.text
    assert r.json()["ingested"] == 1
    assert r.json()["failed"] == []


def test_zip_import_reports_undecodable_images_as_failed(client, bank_dir, fake_ingest):
    blob = _zip({"broken.png": b"not a png", "good.png": _png()})
    r = _post(client, blob)
    assert r.status_code == 200, r.text
    assert r.json()["ingested"] == 1
    assert r.json()["failed"] == ["broken.png"]


def test_zip_import_prefills_the_tier(client, bank_dir, fake_ingest):
    blob = _zip({"a.png": _png(), "b.png": _png((7, 7, 7))})
    r = _post(client, blob, data={"tier": "normal"})
    assert r.status_code == 200, r.text
    assert r.json()["status"]["assigned"] == 2
    assert {i["tier"] for i in client.get(f"{API}/store").json()["images"]} == {"normal"}


# ---- rejections ------------------------------------------------------------


def test_zip_import_with_no_images_is_a_422(client, bank_dir):
    r = _post(client, _zip({"readme.txt": b"hello"}))
    assert r.status_code == 422, r.text
    assert "no images" in r.text


def test_zip_import_of_only_undecodable_images_is_a_422(client, bank_dir):
    r = _post(client, _zip({"a.png": b"not a png"}))
    assert r.status_code == 422, r.text
    assert "no decodable images" in r.text


def test_zip_import_rejects_a_non_zip_upload(client, bank_dir):
    """422, explicitly not 500.

    ``import_bank`` does not catch BadZipFile, so junk there escapes to the
    global handler as a 500. This route catches it; the assertion on the exact
    code is the point of the test.
    """
    r = client.post(
        f"{API}/store/ingest_zip",
        files={"archive": ("x.zip", b"not a zip at all", "application/zip")},
    )
    assert r.status_code == 422, r.text
    assert "not a zip archive" in r.text


def test_zip_import_rejects_zip_slip(client, bank_dir):
    blob = _zip({"../evil.png": _png()}, compression=zipfile.ZIP_STORED)
    r = _post(client, blob)
    assert r.status_code == 422, r.text
    assert "unsafe path" in r.text
    assert not (bank_dir.parent / "evil.png").exists()


def test_zip_import_rejects_an_absolute_member_path(client, bank_dir):
    blob = _zip({"/tmp/evil.png": _png()}, compression=zipfile.ZIP_STORED)
    r = _post(client, blob)
    assert r.status_code == 422, r.text
    assert "unsafe path" in r.text


def test_zip_import_rejects_a_windows_drive_letter_member(client, bank_dir):
    """The ``":" in parts[0]`` arm — a Windows-authored ``C:/...`` member.

    ``is_absolute()`` misses it because the backslashes collapse into one
    component. Neither repo had a test for this arm before.
    """
    blob = _zip({"C:/evil.png": _png()}, compression=zipfile.ZIP_STORED)
    r = _post(client, blob)
    assert r.status_code == 422, r.text
    assert "unsafe path" in r.text


def test_zip_import_rejects_a_zip_bomb_by_ratio(client, bank_dir, monkeypatch):
    """The floor is a named constant so this test does not need 128 MB of zeros."""
    monkeypatch.setattr(security_mod, "ARCHIVE_EXPANSION_FLOOR_BYTES", 1024)
    blob = _zip({"a.png": b"\0" * (4 * 1024 * 1024)})
    assert len(blob) * 8 < 4 * 1024 * 1024, "member must out-expand 8x the upload"
    r = _post(client, blob)
    assert r.status_code == 413, r.text
    assert "expands to" in r.text


def test_zip_import_rejects_a_single_hyper_compressed_member(client, bank_dir, monkeypatch):
    """A lying central directory must not get past the per-member cap.

    ``_check_archive_bounds`` trusts ``file_size``; only the streaming read
    bounds what actually comes out.
    """
    monkeypatch.setattr(security_mod, "ARCHIVE_EXPANSION_FLOOR_BYTES", 1 << 40)
    monkeypatch.setattr(security_mod, "MAX_UPLOAD_BYTES", 1024)
    blob = _zip({"a.png": b"\0" * (64 * 1024)})
    r = _post(client, blob)
    assert r.status_code == 413, r.text
    assert "file too large" in r.text


def test_zip_import_rejects_too_many_images(client, bank_dir, monkeypatch):
    monkeypatch.setattr(security_mod, "MAX_UPLOAD_FILES", 2)
    blob = _zip({"a.png": _png(), "b.png": _png((1, 1, 1)), "c.png": _png((2, 2, 2))})
    r = _post(client, blob)
    assert r.status_code == 413, r.text
    assert "too many files" in r.text
    assert client.get(f"{API}/store").json()["images"] == []


def test_zip_import_rejects_an_oversized_batch(client, bank_dir, monkeypatch):
    png = _png()
    monkeypatch.setattr(security_mod, "MAX_UPLOAD_TOTAL_BYTES", len(png))
    blob = _zip({"a.png": png, "b.png": _png((1, 1, 1))})
    r = _post(client, blob)
    assert r.status_code == 413, r.text
    assert "upload batch too large" in r.text


def test_zip_import_rejects_an_oversized_archive(client, bank_dir, monkeypatch):
    """Patched on the router module, which imported the name."""
    monkeypatch.setattr(store_mod, "MAX_ARCHIVE_BYTES", 64)
    r = _post(client, _zip({"a.png": _png()}))
    assert r.status_code == 413, r.text
    assert "file too large" in r.text


def test_zip_import_honours_the_bank_binding(client, bank_dir):
    r = _post(client, _zip({"a.png": _png()}),
              headers={"X-Bank-Binding": "other-project/default"})
    assert r.status_code == 409, r.text


# ---- the guards stay shared ------------------------------------------------


def test_zip_guards_are_shared_by_every_archive_route():
    """Every archive route must resolve the public alias, not a local copy.

    A router that kept its own inline guard would drop silently out of the
    shared rule and the monkeypatching above would stop biting — the same
    reasoning as ``test_upload_batch_caps.test_both_multi_file_routes_share``.
    """
    for mod in (store_mod, bank_mod, projects_mod):
        assert mod.check_archive_bounds is security_mod.check_archive_bounds
        assert mod.check_archive_paths is security_mod.check_archive_paths
    assert store_mod.read_zip_member is security_mod.read_zip_member


def test_the_existing_package_import_still_rejects_traversal(client, project_id, bank_dir):
    """Regression on the guard lift: the bank package route keeps its 422.

    ``bank_dir`` is requested for its side effect — /banks/import answers 409
    until a bank is active, which would pass the status assertion for the
    wrong reason.
    """
    blob = _zip({"bank.npy": b"x", "bank_meta.json": b"{}", "../evil.txt": b"nope"},
                compression=zipfile.ZIP_STORED)
    r = client.post(
        f"{API}/banks/import",
        files={"archive": ("b.clsbank.zip", blob, "application/zip")},
        data={"project_id": project_id},
    )
    assert r.status_code == 422, r.text
    assert "unsafe path" in r.text
