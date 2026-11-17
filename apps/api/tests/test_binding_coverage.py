# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Every mutating route reads ``X-Bank-Binding``.

The active bank is process-global. ``check_binding`` is what stops a second
browser tab — which re-binds the server the moment it opens another project —
from redirecting the first tab's writes into that project.

Four routes declared no header at all while the client sent one on all of them:

* ``/labelsets/create``, ``/labelsets/select``, ``/labelsets/delete``
* ``PUT /bank/capacity``

``/labelsets/delete`` was the destructive one: it unlinks from the global
active bank, and ``DEFAULT_LABELSET_ID`` is the literal ``"standard"`` that
every project gets on first read, so the ids collide essentially always.
``/bank/capacity`` was the quiet one: the ceiling feeds ``/bank/assemble``, so
an unbound write coresets another project's normal tier on its next assemble.

The signature test below is the one that matters — it fails when a NEW mutating
route forgets the header, which is how all four of these arrived.
"""

from __future__ import annotations

import inspect

import pytest

from app.routers import bank as bank_mod
from app.routers import store as store_mod

API = "/api/v1"

# Every route that writes something owned by the active bank.
MUTATORS = [
    (store_mod, "post_create_labelset"),
    (store_mod, "post_select_labelset"),
    (store_mod, "post_delete_labelset"),
    (store_mod, "post_assign"),
    (store_mod, "delete_from_store"),
    (store_mod, "ingest_images"),
    (store_mod, "ingest_zip"),
    (bank_mod, "put_bank_capacity"),
]


@pytest.mark.parametrize("mod,name", MUTATORS, ids=[n for _, n in MUTATORS])
def test_every_mutating_route_declares_the_binding_header(mod, name):
    fn = getattr(mod, name, None)
    assert fn is not None, f"{name} moved or was renamed; update this list"
    params = inspect.signature(fn).parameters
    assert "binding" in params, f"{name} does not accept X-Bank-Binding"
    default = params["binding"].default
    assert getattr(default, "alias", None) == "X-Bank-Binding", (
        f"{name}'s binding parameter is not bound to the header"
    )


@pytest.mark.parametrize("path,body", [
    ("/labelsets/create", {"name": "other", "copy_active": True}),
    ("/labelsets/select", {"id": "standard"}),
    ("/labelsets/delete", {"id": "standard"}),
])
def test_labelset_routes_refuse_a_foreign_binding(client, project_id, path, body):
    r = client.post(f"{API}/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    r = client.post(f"{API}{path}", json=body,
                    headers={"X-Bank-Binding": "some-other-project/default"})
    assert r.status_code == 409, r.text


def test_capacity_refuses_a_foreign_binding(client, project_id):
    r = client.post(f"{API}/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    r = client.put(f"{API}/bank/capacity", json={"capacity": "small"},
                   headers={"X-Bank-Binding": "some-other-project/default"})
    assert r.status_code == 409, r.text


def test_capacity_still_works_with_the_right_binding(client, project_id):
    r = client.post(f"{API}/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    bank_id = r.json()["bank_id"]
    r = client.put(f"{API}/bank/capacity", json={"capacity": "small"},
                   headers={"X-Bank-Binding": f"{project_id}/{bank_id}"})
    assert r.status_code == 200, r.text
