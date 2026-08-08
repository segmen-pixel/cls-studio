# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Two long operations, one process-global active bank.

``check_binding`` proves the caller was right when the request *arrived*.
``state.bank_dir`` is then re-resolved at write time, minutes later. Both
teach paths already snapshot the bank's identity and 409 if it moved; the
assembly did not, and it is the longest-running write of the three.

The failure it left was not subtle: ``Bank.save`` unlinks every tier file in
the target directory that the incoming bank does not carry, so project A's
assembly landing in project B deletes B's labelled tiers. Two tabs is all it
takes — opening a project fires ``/bank/select`` on mount.

The store index has the same shape of problem between ``/store/delete`` and a
running ingest: two snapshots of one manifest, written back whole.
"""

from __future__ import annotations

import shutil
import threading

import numpy as np
import pytest

from app.core import cls_store as cls_store_mod
from app.core.cls_state import get_state
from app.core.exceptions import ValidationError
from clscore.labelset import DEFAULT_LABELSET_ID, LabelSet
from clscore.store import FeatureStore, StoreMeta

API = "/api/v1"
DIM = 16
MOUNT_REQUESTS = 4


@pytest.fixture
def two_projects(client):
    ids = []
    for name in ("race-a", "race-b"):
        r = client.post(f"{API}/projects", json={"name": name})
        assert r.status_code == 200, r.text
        ids.append(r.json()["id"])
    yield ids
    for pid in ids:
        client.delete(f"{API}/projects/{pid}")


def _select(client, pid: str):
    r = client.post(f"{API}/bank/select", json={"project_id": pid})
    assert r.status_code == 200, r.text
    return r.json()


def test_assembly_refuses_to_land_in_a_bank_that_was_rebound_under_it(client, two_projects):
    """The snapshot-and-refuse both teach paths have, applied to assembly."""
    a, b = two_projects
    sel = _select(client, a)
    state = get_state()

    # A store with one entry, so assembly has something to build from.
    store = FeatureStore(cls_store_mod.store_dir(state), meta=StoreMeta(dim=DIM))
    rng = np.random.default_rng(0)
    e = store.add(rng.random((4, DIM), dtype=np.float32), name="a.png",
                  grid_rows=4, height=16, width=16)
    store.save_index()
    ls = cls_store_mod.active_labelset(state)
    ls.assign(e.id, "normal", label="")
    ls.save(cls_store_mod.labelsets_dir(state))

    # Re-bind to the other project part-way through, the way a second tab
    # mounting on another project does.
    real_assemble = cls_store_mod.assemble_bank

    def _assemble_then_rebind(*args, **kwargs):
        out = real_assemble(*args, **kwargs)
        _select(client, b)
        return out

    cls_store_mod.assemble_bank = _assemble_then_rebind
    try:
        with pytest.raises(ValidationError) as excinfo:
            cls_store_mod.assemble_active_bank(state)
        assert "changed during assembly" in str(excinfo.value)
    finally:
        cls_store_mod.assemble_bank = real_assemble

    assert sel["project_id"] == a


def test_the_store_index_is_read_and_written_under_one_lock(client, project_id):
    """A delete must not persist a manifest taken before a concurrent write.

    Rather than racing real threads (flaky, and the ingest needs the backbone),
    this asserts the property that makes the race impossible: the delete route
    reads the index inside ``state.lock``, so a writer holding the lock cannot
    interleave between its read and its write.
    """
    import inspect

    from app.routers import store as store_mod

    src = inspect.getsource(store_mod.delete_from_store)
    lock_at = src.index("with state.lock:")
    load_at = src.index("load_store(state)")
    assert load_at > lock_at, (
        "load_store must sit inside `with state.lock:` — outside it, this route "
        "and a running ingest each hold a different snapshot of one manifest"
    )

    ingest_src = inspect.getsource(cls_store_mod.ingest_decoded)
    save_at = ingest_src.index("store.save_index()")
    lock_before = ingest_src.rfind("with state.lock:", 0, save_at)
    assert lock_before != -1, "ingest_decoded must save the index under the lock"


def test_a_normal_assembly_still_succeeds(client, project_id):
    """The guard must not fire when nothing rebinds."""
    _select(client, project_id)
    state = get_state()
    store = FeatureStore(cls_store_mod.store_dir(state), meta=StoreMeta(dim=DIM))
    rng = np.random.default_rng(1)
    # One label-set reference for every assign: active_labelset() hands back a
    # fresh object per call, so assigning through separate ones and saving the
    # last persists only that one's work.
    ls = cls_store_mod.active_labelset(state)
    for name in ("a.png", "b.png"):
        e = store.add(rng.random((4, DIM), dtype=np.float32), name=name,
                      grid_rows=4, height=16, width=16)
        ls.assign(e.id, "normal", label="")
    store.save_index()
    ls.save(cls_store_mod.labelsets_dir(state))

    bank = cls_store_mod.assemble_active_bank(state)
    assert int(bank.normal.shape[0]) > 0


def test_a_fresh_project_creates_its_default_label_set_exactly_once(
    client, project_id, monkeypatch
):
    """Every request the Bank tab fires on mount creates that label set lazily.

    ``/bank/select``, ``/labelsets`` and ``/store`` all call ``active_labelset``,
    FastAPI runs those sync endpoints in a threadpool, and on a project nobody
    has opened yet every one of them took the creation branch. They then wrote
    ``standard.json`` through one shared temp name: the first ``replace`` moved
    it away and the rest raised WinError 2, which v0.2.2 surfaced as

        bank select failed: An internal error occurred.

    Unique temp names (``clscore.fsio``) stop the crash. The lock is what stops
    N threads from each creating the same file and the last one winning.
    """
    _select(client, project_id)
    state = get_state()
    d = cls_store_mod.labelsets_dir(state)
    shutil.rmtree(d, ignore_errors=True)  # back to a project nobody has opened

    saved: list[str] = []
    real_save = LabelSet.save

    def counting_save(self, directory):
        saved.append(self.id)
        return real_save(self, directory)

    monkeypatch.setattr(LabelSet, "save", counting_save)

    gate = threading.Barrier(MOUNT_REQUESTS, timeout=30)
    got: list[str] = []
    errors: list[BaseException] = []

    def mount_request() -> None:
        try:
            gate.wait()
            got.append(cls_store_mod.active_labelset(state).id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=mount_request) for _ in range(MOUNT_REQUESTS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a mount request never finished"

    assert errors == [], f"a concurrent mount failed: {errors!r}"
    assert got == [DEFAULT_LABELSET_ID] * MOUNT_REQUESTS, got
    assert saved == [DEFAULT_LABELSET_ID], f"created {len(saved)} times, not once"
    assert not list(d.glob("*.tmp")), "temp files left in the label-sets directory"
