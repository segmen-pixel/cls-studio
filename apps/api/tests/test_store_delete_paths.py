# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""``POST /store/delete`` must only ever unlink blobs the store owns.

The route used to glob on the request body::

    for entry_id in body.ids:
        for p in store.images_dir().glob(f"{entry_id}.*"):
            p.unlink(missing_ok=True)

``ids`` is caller text, and ``glob("../../*.*")`` resolves out of ``store/img/``
into the bank root. Sent alongside one real id — so the ``if removed:`` guard
passes — it answered 200 and took ``bank.npy``, ``bank_meta.json`` and
``assembly_state.json`` with it. Authenticated callers only, which is the
difference between "fix it today" and "fix it now", not a reason to leave it.

The paths now come from the store's own entries and are containment-checked, so
nothing built from caller text reaches ``unlink`` at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clscore.store import FeatureStore, StoreMeta

API = "/api/v1"
DIM = 16


@pytest.fixture
def bank_dir(client, project_id) -> Path:
    r = client.post(f"{API}/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    return Path(r.json()["bank_dir"])


def _seed(bank_dir: Path, names: list[str]) -> FeatureStore:
    """A store with real blobs on disk, without going near the backbone."""
    st = FeatureStore(bank_dir / "store", meta=StoreMeta(dim=DIM))
    rng = np.random.default_rng(0)
    st.images_dir().mkdir(parents=True, exist_ok=True)
    for name in names:
        e = st.add(rng.random((4, DIM), dtype=np.float32), name=name,
                   grid_rows=4, height=16, width=16)
        blob = st.images_dir() / f"{e.id}.png"
        blob.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        e.image_ref = f"store/img/{e.id}.png"
    st.save_index()
    return st


def test_a_traversing_id_cannot_reach_outside_the_image_directory(client, bank_dir):
    st = _seed(bank_dir, ["a.png", "b.png"])
    real_id = st.entries[0].id

    decoy = bank_dir / "bank.npy"
    decoy.write_bytes(b"not really a bank, but it stands where one does")
    meta = bank_dir / "bank_meta.json"
    meta.write_text("{}", encoding="utf-8")

    # The real id makes `removed` truthy, which is what let the crafted one
    # through the guard before.
    r = client.post(f"{API}/store/delete", json={"ids": [real_id, "../../*"]})
    assert r.status_code in (200, 422), r.text

    assert decoy.exists(), "a delete must not be able to reach the bank file"
    assert meta.exists(), "nor the bank metadata"


@pytest.mark.parametrize("hostile", ["../../*", "..", "../*", "*", "a/../../*"])
def test_hostile_ids_never_unlink_anything(client, bank_dir, hostile):
    _seed(bank_dir, ["a.png"])
    sibling = bank_dir / "keep-me.txt"
    sibling.write_text("x", encoding="utf-8")

    client.post(f"{API}/store/delete", json={"ids": [hostile]})
    assert sibling.exists()
    # And the one legitimate blob is untouched, because no entry matched.
    assert list((bank_dir / "store" / "img").glob("*.png"))


def test_a_normal_delete_still_removes_its_own_blob(client, bank_dir):
    st = _seed(bank_dir, ["a.png", "b.png"])
    victim, survivor = st.entries[0].id, st.entries[1].id
    img = bank_dir / "store" / "img"
    assert (img / f"{victim}.png").exists()

    r = client.post(f"{API}/store/delete", json={"ids": [victim]})
    assert r.status_code == 200, r.text

    assert not (img / f"{victim}.png").exists(), "the deleted entry's blob goes"
    assert (img / f"{survivor}.png").exists(), "and only that one"
    assert [i["id"] for i in r.json()["images"]] == [survivor]


def test_a_migrated_entrys_source_image_is_reclaimed_too(client, bank_dir):
    """Containment was store/img/ ALONE, narrower than the refs we produce.

    Migrating an append-era bank mints refs into ``_images/<tier>/``, so
    deleting such an image reclaimed nothing: the file stayed on disk and the
    project card kept thumbnailing a picture the operator had deleted.
    """
    st = _seed(bank_dir, ["a.png"])
    taught = bank_dir / "_images" / "normal"
    taught.mkdir(parents=True, exist_ok=True)
    (taught / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    st.entries[0].image_ref = "_images/normal/a.png"
    st.save_index()

    r = client.post(f"{API}/store/delete", json={"ids": [st.entries[0].id]})
    assert r.status_code == 200, r.text
    assert not (taught / "a.png").exists(), "the migrated copy goes too"


def test_a_ref_naming_the_bank_file_is_still_refused(client, bank_dir):
    """Widening containment must not widen it to the whole bank directory."""
    st = _seed(bank_dir, ["a.png"])
    decoy = bank_dir / "bank.npy"
    decoy.write_bytes(b"not really a bank, but it stands where one does")
    st.entries[0].image_ref = "bank.npy"
    st.save_index()

    r = client.post(f"{API}/store/delete", json={"ids": [st.entries[0].id]})
    assert r.status_code == 200, r.text
    assert decoy.exists(), "a hand-edited ref must not reach the bank file"


def test_a_missing_blob_still_drops_its_cached_renditions(client, bank_dir):
    """Existence is checked by the unlink, not by the admission rule.

    An entry whose blob is already gone used to be skipped entirely, so its
    downscaled renditions stayed in store/cache/ forever.
    """
    st = _seed(bank_dir, ["a.png"])
    eid = st.entries[0].id
    (bank_dir / "store" / "img" / f"{eid}.png").unlink()
    cache = bank_dir / "store" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rendition = cache / f"{eid}_thumb.jpg"
    rendition.write_bytes(b"\xff\xd8\xff")

    r = client.post(f"{API}/store/delete", json={"ids": [eid]})
    assert r.status_code == 200, r.text
    assert not rendition.exists(), "the orphaned rendition goes with the entry"
