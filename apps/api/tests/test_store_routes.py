# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Store / label-set / assemble routes.

The store is seeded on disk rather than through ``/store/ingest`` so the
suite never loads DINOv2: everything these routes do after extraction is
bookkeeping and numpy, and that is exactly the part worth testing.
"""

from __future__ import annotations

import json
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


def test_deleting_several_assigned_images_drops_every_assignment(client, bank_dir):
    """One id per request cannot tell any() from all().

    ``any(ls.unassign(i) for i in ids)`` stopped at the first assigned id, so
    a batch delete left the rest assigned to entries that were gone. The Bank
    tab then read "0 unassigned" over images nobody had labelled, ticked step
    2 green, and assembling produced an empty bank.
    """
    _seed_store(bank_dir, [(f"a{i}.png", 5) for i in range(6)])
    ids = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": ids[:4], "tier": "normal"})
    r = client.post(f"{API}/store/delete", json={"ids": ids[:4]})
    assert r.status_code == 200

    st = client.get(f"{API}/bank/assembly").json()
    assert st["store_images"] == 2
    assert st["assigned"] == 0, "every deleted image's assignment must go"
    assert st["unassigned"] == 2, "the two survivors are genuinely unlabelled"
    assert st["stale_assignments"] == 0


def test_the_unassigned_count_survives_a_stale_assignment(client, bank_dir):
    """Counted set-wise, so a label set corrupted by the shipped build reads right.

    The old arithmetic was ``max(0, len(store) - sum(counts))``: it
    under-reported by exactly the stale count and floored at zero, which is
    how "0 unassigned" appeared over four unlabelled images.
    """
    _seed_store(bank_dir, [(f"a{i}.png", 5) for i in range(4)])
    ids = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": ids, "tier": "normal"})
    # Forge the divergence the old delete path produced: assignments naming
    # entries the store no longer has.
    ls_path = next((bank_dir / "labelsets").glob("*.json"))
    doc = json.loads(ls_path.read_text(encoding="utf-8"))
    for i in range(5):
        doc["assignments"][f"deadbeef{i}"] = {"tier": "critical", "label": "kizu"}
    ls_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    st = client.get(f"{API}/bank/assembly").json()
    assert st["assigned"] == 4
    assert st["unassigned"] == 0
    assert st["stale_assignments"] == 5, "surfaced, not absorbed"


# ---- serving the source image ----------------------------------------------


def _assembled(client, bank_dir: Path, images: list[str], tier: str = "normal",
               label: str = "") -> list[str]:
    """Ingest, assign, assemble — and write a blob for each entry."""
    _seed_store(bank_dir, [(n, 5) for n in images])
    ids = _ids(client)
    img_dir = bank_dir / "store" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    st = FeatureStore.load(bank_dir / "store")
    for e in st.entries:
        (img_dir / f"{e.id}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        e.image_ref = f"store/img/{e.id}.png"
    st.save_index()
    body = {"ids": ids, "tier": tier}
    if label:
        body["label"] = label
    client.post(f"{API}/labelsets/assign", json=body)
    assert client.post(f"{API}/bank/assemble", json={}).status_code == 200
    return ids


def test_a_relabelled_image_still_serves_its_bytes(client, bank_dir):
    """The bank index freezes the tier; the label set is live. They diverge.

    Re-labelling in the Bank tab does NOT assemble -- that is the whole point
    of the store/bank split -- so the viewer asked for the frozen tier while
    the resolver checked the current one and answered 404 for the thumbnail,
    the centre image and the heatmap alike.
    """
    _assembled(client, bank_dir, ["a.png"])
    row = client.get(f"{API}/bank/images").json()["images"][0]
    assert client.get(f"{API}{row['url']}").status_code == 200

    ids = _ids(client)
    client.post(f"{API}/labelsets/assign",
                json={"ids": ids, "tier": "critical", "label": "kizu"})
    # No assemble: the bank still says normal.
    r = client.get(f"{API}{row['url']}")
    assert r.status_code == 200, "re-labelling must not blank the viewer"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_an_unassigned_image_still_serves_its_bytes(client, bank_dir):
    _assembled(client, bank_dir, ["a.png"])
    row = client.get(f"{API}/bank/images").json()["images"][0]
    client.post(f"{API}/labelsets/unassign", json={"ids": _ids(client)})
    assert client.get(f"{API}{row['url']}").status_code == 200


def test_a_hash_in_a_filename_survives_the_round_trip(client, bank_dir):
    """`#` truncated the request at the fragment, `%` began an escape."""
    _assembled(client, bank_dir, ["lot#3 50%.png"])
    row = client.get(f"{API}/bank/images").json()["images"][0]
    assert row["name"] == "lot#3 50%.png"
    assert "%23" in row["url"] and "%25" in row["url"]
    assert client.get(f"{API}{row['url']}").status_code == 200


def test_duplicate_filenames_serve_their_own_bytes(client, bank_dir):
    """Two photographs, one filename: the id in the URL tells them apart."""
    _assembled(client, bank_dir, ["img001.png", "img001.png"])
    rows = client.get(f"{API}/bank/images").json()["images"]
    assert len(rows) == 2
    assert rows[0]["url"] != rows[1]["url"], "the entry id disambiguates them"
    for row in rows:
        assert client.get(f"{API}{row['url']}").status_code == 200


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


# ---- the operator's own words -----------------------------------------------


def test_the_store_listing_returns_the_label_the_operator_typed(client, bank_dir):
    """The chips are built from this field and a chip click sends it back.

    Returning the on-disk stem meant a Japanese defect name never round-tripped
    to what was typed. BankTab's comment claimed it "fills in on the
    reconcile"; the reconcile reads THIS field, so it never did.
    """
    _seed_store(bank_dir, [("ng.png", 5)])
    ids = _ids(client)
    client.post(f"{API}/labelsets/assign",
                json={"ids": ids, "tier": "critical", "label": "傷A"})

    row = client.get(f"{API}/store").json()["images"][0]
    assert row["label"] == "傷A"


def test_the_bank_listing_keys_on_the_stem_and_shows_the_typed_name(client, bank_dir):
    """`label` stays the stem: Develop uses it as a cache key and passes it to
    /bank/images/evaluate. The readable form rides alongside it."""
    from clscore.bank import safe_label

    _seed_store(bank_dir, [("ng.png", 5)])
    ids = _ids(client)
    client.post(f"{API}/labelsets/assign",
                json={"ids": ids, "tier": "critical", "label": "傷A"})
    assert client.post(f"{API}/bank/assemble", json={}).status_code == 200

    row = next(i for i in client.get(f"{API}/bank/images").json()["images"]
               if i["tier"] == "critical")
    assert row["label"] == safe_label("傷A")
    assert row["label_display"] == "傷A"


def test_two_japanese_classes_stay_two_classes_through_the_api(client, bank_dir):
    _seed_store(bank_dir, [("a.png", 5), ("b.png", 5)])
    a, b = _ids(client)
    client.post(f"{API}/labelsets/assign", json={"ids": [a], "tier": "critical", "label": "傷"})
    client.post(f"{API}/labelsets/assign", json={"ids": [b], "tier": "critical", "label": "汚れ"})
    assert client.post(f"{API}/bank/assemble", json={}).status_code == 200

    rows = [i for i in client.get(f"{API}/bank/images").json()["images"]
            if i["tier"] == "critical"]
    assert len({r["label"] for r in rows}) == 2, "two buckets, not one"
    assert {r["label_display"] for r in rows} == {"傷", "汚れ"}
    state = client.get(f"{API}/bank").json()
    assert len(state["critical_by_label"]) == 2


def test_an_ascii_label_is_unchanged_end_to_end(client, bank_dir):
    """The no-migration guarantee, asserted at the API boundary."""
    _seed_store(bank_dir, [("ng.png", 5)])
    client.post(f"{API}/labelsets/assign",
                json={"ids": _ids(client), "tier": "critical", "label": "scratch"})
    assert client.post(f"{API}/bank/assemble", json={}).status_code == 200

    row = next(i for i in client.get(f"{API}/bank/images").json()["images"]
               if i["tier"] == "critical")
    assert row["label"] == "scratch" and row["label_display"] == "scratch"


def test_the_bank_listing_ships_the_store_entry_id(client, bank_dir):
    """The row's only identity. The Teach tab keys its checkboxes on it and
    assigns by it; two images sharing a filename used to collapse into one
    row key, so ticking either box selected both and the suppression action
    refused to move either."""
    _assembled(client, bank_dir, ["img001.png", "img001.png"])
    rows = client.get(f"{API}/bank/images").json()["images"]
    assert len(rows) == 2
    ids = {r["entry_id"] for r in rows}
    assert len(ids) == 2 and "" not in ids
    assert ids == set(_ids(client))
    assert {r["name"] for r in rows} == {"img001.png"}, "one name, two identities"


def test_evaluate_scores_each_duplicate_on_its_own_rows(client, bank_dir):
    """Two photographs, one filename: without ?id= the route took the FIRST
    index entry with a matching name, so both Teach rows showed one image's
    score and the histogram counted a pair as a single image."""
    _assembled(client, bank_dir, ["img001.png", "img001.png"])
    rows = [i for i in client.get(f"{API}/bank/images").json()["images"]
            if i["tier"] == "normal"]
    assert len(rows) == 2 and rows[0]["entry_id"] != rows[1]["entry_id"]

    seen = []
    for r in rows:
        got = client.get(
            f"{API}/bank/images/evaluate",
            params={"tier": "normal", "name": r["name"], "label": "", "id": r["entry_id"]},
        )
        assert got.status_code == 200, got.text
        seen.append(got.json())
    # Same name, same patch count -- what tells them apart is which rows were
    # scored, and the route must have used a different range for each.
    assert all(e["name"] == "img001.png" for e in seen)
    assert seen[0] != seen[1] or rows[0]["patches"] != rows[1]["patches"]


def test_evaluate_falls_back_to_the_name_for_an_unstamped_bank(client, bank_dir):
    _assembled(client, bank_dir, ["solo.png"])
    r = client.get(
        f"{API}/bank/images/evaluate",
        params={"tier": "normal", "name": "solo.png", "label": ""},
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "solo.png"


def test_evaluate_with_an_unknown_id_still_finds_the_image_by_name(client, bank_dir):
    """A client holding a stale id must not lose the row entirely."""
    _assembled(client, bank_dir, ["solo.png"])
    r = client.get(
        f"{API}/bank/images/evaluate",
        params={"tier": "normal", "name": "solo.png", "label": "", "id": "deadbeef"},
    )
    assert r.status_code == 200, r.text
