# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Store / label-set / assemble routes.

The store is seeded on disk rather than through ``/store/ingest`` so the
suite never loads DINOv2: everything these routes do after extraction is
bookkeeping and numpy, and that is exactly the part worth testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clscore.bank import Bank, BankMeta
from clscore.incident import SEVERITY_HEAVY
from clscore.store import FeatureStore, StoreMeta
from clscore.sw import DINO_PATCH, WINDOW_SIZE, expected_rows

DIM = 16
API = "/api/v1"


@pytest.fixture
def bank_dir(client, project_id) -> Path:
    r = client.post(f"{API}/bank/select", json={"project_id": project_id})
    assert r.status_code == 200, r.text
    return Path(r.json()["bank_dir"])


def _seed_store(bank_dir: Path, images: list[tuple[str, int]], *, size: int = 0) -> FeatureStore:
    """Write a store straight to disk, bypassing the backbone."""
    st = FeatureStore(bank_dir / "store", meta=StoreMeta(dim=DIM))
    rng = np.random.default_rng(0)
    for name, rows in images:
        st.add(
            rng.random((rows, DIM), dtype=np.float32),
            name=name,
            grid_rows=rows,
            height=size,
            width=size,
        )
    st.save_index()
    return st


def _ids(client) -> list[str]:
    return [i["id"] for i in client.get(f"{API}/store").json()["images"]]


# ---- listing ---------------------------------------------------------------


def test_store_is_empty_before_migration(client, bank_dir):
    body = client.get(f"{API}/store").json()
    assert body["images"] == []
    assert body["total_rows"] == 0
    st = client.get(f"{API}/bank/assembly").json()
    assert st["migrated"] is False
    # Nothing to assemble yet, so nothing is stale -- telling the operator to
    # rebuild a bank the store cannot reproduce would be worse than silence.
    assert st["stale"] is False


def test_store_lists_seeded_images_as_unassigned(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5), ("b.png", 7)])
    body = client.get(f"{API}/store").json()
    assert [i["name"] for i in body["images"]] == ["a.png", "b.png"]
    assert [i["tier"] for i in body["images"]] == ["", ""]
    assert body["total_rows"] == 12
    assert body["dim"] == DIM
    st = client.get(f"{API}/bank/assembly").json()
    assert (st["store_images"], st["assigned"], st["unassigned"]) == (2, 0, 2)


def _seed_store_with_images(
    bank_dir: Path, images: list[tuple[str, int]],
) -> dict[str, bytes]:
    """A store whose entries carry a real source image, as ingest leaves them."""
    from PIL import Image

    src = bank_dir / "store" / "src"
    src.mkdir(parents=True, exist_ok=True)
    st = FeatureStore(bank_dir / "store", meta=StoreMeta(dim=DIM))
    rng = np.random.default_rng(0)
    written: dict[str, bytes] = {}
    for i, (name, rows) in enumerate(images):
        p = src / name
        Image.new("RGB", (8, 8), (i * 10 % 256, 0, 0)).save(p)
        written[name] = p.read_bytes()
        st.add(
            rng.random((rows, DIM), dtype=np.float32),
            name=name,
            grid_rows=rows,
            height=8,
            width=8,
            image_ref=f"store/src/{name}",
        )
    st.save_index()
    return written


def test_an_assembled_bank_still_serves_its_images(client, bank_dir):
    """Assembling writes no ``_images/`` copy — only the retired
    ``/bank/append`` path did — so the endpoint has to fall back to the
    original the store kept. Without it the Teach tab viewer is a black
    rectangle and the heatmap prefetch reports a 404."""
    written = _seed_store_with_images(bank_dir, [("ok.png", 5)])
    (entry_id,) = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": [entry_id], "tier": "normal"})
    assert client.post(f"{API}/bank/assemble").status_code == 200

    assert not (bank_dir / "_images" / "normal" / "ok.png").exists()
    r = client.get(f"{API}/bank/images/normal/ok.png")
    assert r.status_code == 200, r.text
    assert r.content == written["ok.png"]


def test_the_store_fallback_does_not_cross_tiers(client, bank_dir):
    """Same filename in two tiers must not serve the other tier's picture —
    that would put a defect where the viewer is showing a good part."""
    written = _seed_store_with_images(bank_dir, [("same.png", 5), ("same.png", 7)])
    assert len(written) == 1  # one file on disk; two entries reference sizes
    good, bad = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": [good], "tier": "normal"})
    client.post(
        f"{API}/labelsets/assign",
        json={"ids": [bad], "tier": "critical", "label": "scratch"},
    )
    assert client.post(f"{API}/bank/assemble").status_code == 200

    for tier in ("normal", "critical"):
        r = client.get(f"{API}/bank/images/{tier}/same.png")
        assert r.status_code == 200, r.text
    # An image labelled into neither tier has nothing to serve.
    assert client.get(f"{API}/bank/images/negative/same.png").status_code == 404


# ---- assignment ------------------------------------------------------------


def test_assign_then_assemble_builds_the_bank(client, bank_dir):
    _seed_store(bank_dir, [("ok.png", 5), ("ng.png", 7)])
    a, b = _ids(client)

    r = client.post(f"{API}/labelsets/assign", json={"ids": [a], "tier": "normal"})
    assert r.status_code == 200, r.text
    assert r.json()["status"]["stale"] is True
    client.post(
        f"{API}/labelsets/assign",
        json={"ids": [b], "tier": "critical", "label": "scratch"},
    )

    r = client.post(f"{API}/bank/assemble")
    assert r.status_code == 200, r.text
    assert r.json()["bank"]["normal"] == 5
    assert r.json()["bank"]["critical_by_label"] == {"scratch": 7}
    assert r.json()["status"]["stale"] is False
    # And it is really on disk, not just in memory.
    assert Bank.load(bank_dir).normal.shape == (5, DIM)


def test_reassigning_moves_rows_without_re_extracting(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5), ("b.png", 7)])
    a, b = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": [a, b], "tier": "normal"})
    client.post(f"{API}/bank/assemble")
    assert client.get(f"{API}/bank").json()["normal"] == 12

    client.post(
        f"{API}/labelsets/assign",
        json={"ids": [b], "tier": "critical", "label": "scratch"},
    )
    client.post(f"{API}/bank/assemble")
    bank = client.get(f"{API}/bank").json()
    assert bank["normal"] == 5
    assert bank["critical_by_label"] == {"scratch": 7}
    # The features never moved.
    assert client.get(f"{API}/store").json()["total_rows"] == 12


def test_unassign_returns_an_image_to_the_pool(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5)])
    a = _ids(client)[0]
    client.post(f"{API}/labelsets/assign", json={"ids": [a], "tier": "normal"})
    r = client.post(f"{API}/labelsets/unassign", json={"ids": [a]})
    assert r.json()["changed"] == 1
    assert r.json()["status"]["unassigned"] == 1
    assert client.get(f"{API}/store").json()["images"][0]["tier"] == ""


def test_assigning_an_unknown_id_is_a_404(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5)])
    r = client.post(f"{API}/labelsets/assign", json={"ids": ["ffffff"], "tier": "normal"})
    assert r.status_code == 404


def test_assemble_without_a_store_is_a_409(client, bank_dir):
    r = client.post(f"{API}/bank/assemble")
    assert r.status_code == 409


# ---- marks -----------------------------------------------------------------


def test_marks_are_stored_as_grid_patches_and_reach_the_bank(client, bank_dir):
    grid = expected_rows(WINDOW_SIZE, WINDOW_SIZE)
    _seed_store(bank_dir, [("ng.png", grid)], size=WINDOW_SIZE)
    a = _ids(client)[0]
    client.post(
        f"{API}/labelsets/assign", json={"ids": [a], "tier": "critical", "label": "scratch"}
    )
    # One patch square in the top-left corner.
    frac = DINO_PATCH / WINDOW_SIZE
    r = client.post(
        f"{API}/labelsets/mark",
        json={"id": a, "rects": [{"x": 0.0, "y": 0.0, "w": frac, "h": frac}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["marks"] == 1
    assert client.get(f"{API}/store").json()["images"][0]["marks"] == 1

    client.post(f"{API}/bank/assemble")
    sev = Bank.load(bank_dir).critical_meta["scratch"].severity
    assert list(np.flatnonzero(sev == SEVERITY_HEAVY)) == [0]


def test_marking_an_unassigned_image_is_a_409(client, bank_dir):
    _seed_store(bank_dir, [("ng.png", 4)], size=WINDOW_SIZE)
    a = _ids(client)[0]
    r = client.post(
        f"{API}/labelsets/mark",
        json={"id": a, "rects": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}]},
    )
    assert r.status_code == 409


def test_marking_an_image_of_unknown_size_is_refused(client, bank_dir):
    """Resolving rectangles without the pixel size would mark the wrong patches."""
    _seed_store(bank_dir, [("ng.png", 4)])  # size=0
    a = _ids(client)[0]
    client.post(f"{API}/labelsets/assign", json={"ids": [a], "tier": "critical"})
    r = client.post(
        f"{API}/labelsets/mark",
        json={"id": a, "rects": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}]},
    )
    assert r.status_code == 409
    assert "pixel size" in r.json()["detail"]


# ---- label sets ------------------------------------------------------------


def test_a_forked_label_set_starts_from_the_active_one(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5), ("b.png", 5)])
    a, b = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": [a, b], "tier": "normal"})

    r = client.post(f"{API}/labelsets/create", json={"name": "what if b is NG"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["active_id"] != "standard"
    forked = body["active_id"]
    assert client.get(f"{API}/store").json()["images"][1]["tier"] == "normal"

    client.post(
        f"{API}/labelsets/assign", json={"ids": [b], "tier": "critical", "label": "scratch"}
    )
    # The original judgement is untouched.
    client.post(f"{API}/labelsets/select", json={"id": "standard"})
    assert client.get(f"{API}/store").json()["images"][1]["tier"] == "normal"
    client.post(f"{API}/labelsets/select", json={"id": forked})
    assert client.get(f"{API}/store").json()["images"][1]["tier"] == "critical"


def test_an_empty_fork_starts_from_nothing(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5)])
    a = _ids(client)[0]
    client.post(f"{API}/labelsets/assign", json={"ids": [a], "tier": "normal"})
    client.post(f"{API}/labelsets/create", json={"name": "fresh", "copy_active": False})
    assert client.get(f"{API}/store").json()["images"][0]["tier"] == ""


def test_deleting_the_active_label_set_moves_the_marker(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5)])
    client.get(f"{API}/store")  # materialises "standard"
    created = client.post(f"{API}/labelsets/create", json={"name": "temp"}).json()
    victim = created["active_id"]
    r = client.post(f"{API}/labelsets/delete", json={"id": victim})
    assert r.status_code == 200
    assert r.json()["active_id"] == "standard"
    assert victim not in [x["id"] for x in r.json()["labelsets"]]


def test_selecting_a_missing_label_set_is_a_404(client, bank_dir):
    r = client.post(f"{API}/labelsets/select", json={"id": "nope"})
    assert r.status_code == 404


# ---- deletion --------------------------------------------------------------


def test_deleting_from_the_store_drops_the_assignment_too(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5), ("b.png", 5)])
    a, b = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": [a, b], "tier": "normal"})
    r = client.post(f"{API}/store/delete", json={"ids": [a]})
    assert r.status_code == 200
    assert [i["name"] for i in r.json()["images"]] == ["b.png"]
    st = client.get(f"{API}/bank/assembly").json()
    assert (st["store_images"], st["assigned"]) == (1, 1)


# ---- migration -------------------------------------------------------------


def _write_bank(bank_dir: Path) -> Bank:
    rng = np.random.default_rng(3)
    bank = Bank(normal=np.zeros((0, DIM), dtype=np.float16), meta=BankMeta(dim=DIM))
    for i in range(2):
        bank.append("normal", rng.random((6, DIM)).astype(np.float16), image_name=f"ok{i}.png")
    bank.append(
        "critical", rng.random((4, DIM)).astype(np.float16),
        label="scratch", image_name="ng0.png",
    )
    bank.save(bank_dir)
    return bank


def test_migration_carves_the_loaded_bank(client, bank_dir):
    original = _write_bank(bank_dir)
    assert client.post(f"{API}/bank/reload").json()["normal"] == 12

    r = client.post(f"{API}/store/migrate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["images"] == 3
    assert body["rows"] == 16
    assert body["status"]["counts"] == {"normal": 2, "critical": 1, "negative": 0}
    assert body["status"]["stale"] is False  # the loaded bank already matches

    # And re-assembling reproduces it.
    client.post(f"{API}/bank/assemble")
    rebuilt = Bank.load(bank_dir)
    assert np.array_equal(rebuilt.normal, original.normal)
    assert np.array_equal(rebuilt.critical["scratch"], original.critical["scratch"])


def test_migrating_twice_is_a_409(client, bank_dir):
    _write_bank(bank_dir)
    client.post(f"{API}/bank/reload")
    assert client.post(f"{API}/store/migrate").status_code == 200
    assert client.post(f"{API}/store/migrate").status_code == 409


# ---- images ----------------------------------------------------------------


def test_store_image_is_404_without_a_source(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5)])
    a = _ids(client)[0]
    assert client.get(f"{API}/store/image/{a}").status_code == 404


def test_store_image_is_served_when_present(client, bank_dir, sample_image_bytes):
    st = _seed_store(bank_dir, [("a.png", 5)])
    entry = st.entries[0]
    st.images_dir().mkdir(parents=True, exist_ok=True)
    (st.images_dir() / f"{entry.id}.png").write_bytes(sample_image_bytes)
    entry.image_ref = f"store/img/{entry.id}.png"
    st.save_index()
    r = client.get(f"{API}/store/image/{entry.id}")
    assert r.status_code == 200
    assert r.content == sample_image_bytes


def test_store_image_refuses_to_escape_the_bank(client, bank_dir):
    """The ref is data off disk, so traversal has to be checked, not assumed."""
    st = _seed_store(bank_dir, [("a.png", 5)])
    st.entries[0].image_ref = "../../../../etc/hosts"
    st.save_index()
    assert client.get(f"{API}/store/image/{st.entries[0].id}").status_code == 404


# ---- grouping ---------------------------------------------------------------


def test_group_preview_splits_by_filename_date(client, bank_dir):
    _seed_store(
        bank_dir,
        [
            ("OK_170104_094937.png", 3),
            ("OK_170104_101122.png", 3),
            ("OK_170105_083010.png", 3),
            ("plate.png", 3),
        ],
    )
    body = client.get(f"{API}/store/groups", params={"mode": "datetime"}).json()
    assert body["groups"]["170104"] == ["OK_170104_094937.png", "OK_170104_101122.png"]
    assert body["groups"]["170105"] == ["OK_170105_083010.png"]
    # The one the rule could not place is reported, not hidden in a group.
    assert body["ungrouped"] == 1
    assert body["grouped"] == 3


def test_group_preview_rejects_an_unknown_mode(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 3)])
    assert client.get(f"{API}/store/groups", params={"mode": "vibes"}).status_code == 422


def test_manual_group_is_stored_on_the_image(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 3), ("b.png", 3)])
    a, b = _ids(client)
    r = client.post(f"{API}/store/group", json={"ids": [a, b], "group": "lot-1"})
    assert r.status_code == 200
    assert [i["group"] for i in r.json()["images"]] == ["lot-1", "lot-1"]
    preview = client.get(f"{API}/store/groups", params={"mode": "manual"}).json()
    assert preview["groups"]["lot-1"] == ["a.png", "b.png"]
    # And it survives a reload, because it lives in the store index.
    assert [i["group"] for i in client.get(f"{API}/store").json()["images"]] == [
        "lot-1", "lot-1",
    ]


def test_clearing_a_manual_group(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 3)])
    a = _ids(client)[0]
    client.post(f"{API}/store/group", json={"ids": [a], "group": "lot-1"})
    r = client.post(f"{API}/store/group", json={"ids": [a], "group": ""})
    assert r.json()["images"][0]["group"] == ""


def test_grouping_an_unknown_id_is_a_404(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 3)])
    r = client.post(f"{API}/store/group", json={"ids": ["ffffff"], "group": "x"})
    assert r.status_code == 404


def test_grouped_evaluation_excludes_the_whole_lot(client, bank_dir):
    """The twin-image problem: leave-one-out leaves the twin behind."""
    rng = np.random.default_rng(1)
    # Every row identical inside the lot, so the default k=5 neighbours are
    # all twins and the leave-one-out distance really is ~0. With distinct
    # rows the score would just be the spread within the image.
    row = rng.random((1, DIM), dtype=np.float32)
    twin = np.repeat(row, 6, axis=0)
    far = np.repeat(row + 10.0, 6, axis=0).astype(np.float32)
    st = FeatureStore(bank_dir / "store", meta=StoreMeta(dim=DIM))
    a1 = st.add(twin, name="A1.png", grid_rows=6)
    a2 = st.add(twin.copy(), name="A2.png", grid_rows=6)
    b = st.add(far, name="B.png", grid_rows=6)
    st.save_index()
    client.post(
        f"{API}/labelsets/assign", json={"ids": [a1.id, a2.id, b.id], "tier": "normal"}
    )
    client.post(f"{API}/bank/assemble")

    loo = client.get(
        f"{API}/bank/images/evaluate", params={"tier": "normal", "name": "A1.png"}
    ).json()
    # Its identical twin is still in the bank, so the image scores as nominal
    # no matter what it actually looks like.
    # Not exactly 0: the resident bank is int8-quantised while the query rows
    # are read back at full precision.
    assert loo["score_mean"] < 0.5

    client.post(f"{API}/store/group", json={"ids": [a1.id, a2.id], "group": "lot-1"})
    grouped = client.get(
        f"{API}/bank/images/evaluate",
        params={"tier": "normal", "name": "A1.png", "group_mode": "manual"},
    ).json()
    assert grouped["score_mean"] > 10.0

    # The grouped score must not have displaced the cached leave-one-out one:
    # the exemplar and projection paths read that cache by image name.
    cached = client.get(f"{API}/bank/evaluation/cached").json()
    entry = next(c for c in cached if c["name"] == "A1.png")
    assert entry["score_mean"] == pytest.approx(loo["score_mean"], abs=1e-6)


def test_grouped_evaluation_rejects_an_unknown_mode(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 4), ("b.png", 4)])
    a, b = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": [a, b], "tier": "normal"})
    client.post(f"{API}/bank/assemble")
    r = client.get(
        f"{API}/bank/images/evaluate",
        params={"tier": "normal", "name": "a.png", "group_mode": "vibes"},
    )
    assert r.status_code == 422


# ---- export ----------------------------------------------------------------


def _archive_names(resp) -> list[str]:
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return zf.namelist()


def test_export_carries_the_store_so_the_receiver_can_relabel(client, bank_dir):
    _write_bank(bank_dir)
    client.post(f"{API}/bank/reload")
    client.post(f"{API}/store/migrate")
    names = _archive_names(client.get(f"{API}/bank/export"))
    assert "bank.npy" in names
    assert any(n.startswith("store/feat/") for n in names)
    assert any(n.startswith("labelsets/") for n in names)


def test_export_can_leave_the_store_behind_for_a_scoring_only_deploy(client, bank_dir):
    _write_bank(bank_dir)
    client.post(f"{API}/bank/reload")
    client.post(f"{API}/store/migrate")
    names = _archive_names(
        client.get(f"{API}/bank/export", params={"include_store": "false"})
    )
    assert "bank.npy" in names
    assert not any(n.startswith("store/") for n in names)
    # The judgement is a few KB and is never worth dropping.
    assert any(n.startswith("labelsets/") for n in names)
