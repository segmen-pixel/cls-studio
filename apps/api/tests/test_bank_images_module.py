# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The single answer to "where are this bank's images, and how many".

Pure path/JSON tests — no TestClient, no torch. Every case here corresponds to
a defect that shipped because two code paths answered one of these questions
independently and only one of them was revisited after the store migration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core import bank_images as bi

# ---- helpers ---------------------------------------------------------------


def _store(bank: Path, entries: list[dict]) -> None:
    d = bank / bi.STORE_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    (d / bi.STORE_INDEX_FILE).write_text(
        json.dumps({"entries": entries}), encoding="utf-8"
    )


def _blob(bank: Path, name: str, data: bytes = b"\x89PNG") -> str:
    d = bank / bi.STORE_SUBDIR / bi.STORE_IMAGES_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return f"{bi.STORE_SUBDIR}/{bi.STORE_IMAGES_SUBDIR}/{name}"


def _labelset(bank: Path, assignments: dict, ls_id: str = "standard") -> None:
    d = bank / bi.LABELSETS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{ls_id}.json").write_text(
        json.dumps({"id": ls_id, "assignments": assignments}), encoding="utf-8"
    )
    (d / bi.LABELSET_ACTIVE_MARKER).write_text(ls_id, encoding="utf-8")


def _taught(bank: Path, tier: str, name: str) -> Path:
    d = bank / bi.IMAGES_SUBDIR / tier
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"\x89PNG")
    return p


# ---- the name rule ---------------------------------------------------------


def test_the_writers_rule_is_the_readers_rule():
    assert bi.safe_image_name("img001 (1)_豆_1.png") == "img001__1____1.png"
    once = bi.safe_image_name("img001 (1)_豆_1.png")
    assert bi.safe_image_name(once) == once, "mapping must be idempotent"


def test_an_empty_name_still_produces_something_serveable():
    assert bi.safe_image_name("") == "image.png"
    assert bi.safe_image_name("/") == "image.png"


def test_a_traversing_name_raises_but_does_not_escape(tmp_path: Path):
    with pytest.raises(bi.UnsafeImageName):
        bi.tier_image_path(tmp_path, "normal", "..")
    root = (tmp_path / bi.IMAGES_SUBDIR / "normal").resolve()
    for hostile in ("../evil.png", "..\\evil.png", "/etc/passwd", "a/b/c.png"):
        assert bi.tier_image_path(tmp_path, "normal", hostile).parent == root


# ---- resolving one image ---------------------------------------------------


def test_resolve_prefers_the_taught_copy_over_the_store(tmp_path: Path):
    want = _taught(tmp_path, "normal", "a.png")
    ref = _blob(tmp_path, "000000.png")
    _store(tmp_path, [{"id": "000000", "name": "a.png", "image_ref": ref}])
    assert bi.resolve_bank_image(tmp_path, "normal", "a.png") == want


def test_resolve_finds_the_store_copy_when_there_is_no_images_dir(tmp_path: Path):
    ref = _blob(tmp_path, "000000.png")
    _store(tmp_path, [{"id": "000000", "name": "a.png", "image_ref": ref}])
    _labelset(tmp_path, {"000000": {"tier": "normal", "label": ""}})
    got = bi.resolve_bank_image(tmp_path, "normal", "a.png")
    assert got is not None and got.name == "000000.png"


def test_resolve_by_entry_id_ignores_the_label_set_entirely(tmp_path: Path):
    """Re-labelling after an assemble used to 404 every touched image.

    The bank index froze the tier at assemble time; the reader asked the
    ACTIVE label set, which the operator had just changed.
    """
    ref = _blob(tmp_path, "000000.png")
    _store(tmp_path, [{"id": "000000", "name": "a.png", "image_ref": ref}])
    _labelset(tmp_path, {"000000": {"tier": "critical", "label": "kizu"}})
    got = bi.resolve_bank_image(tmp_path, "normal", "a.png", entry_id="000000")
    assert got is not None and got.name == "000000.png"


def test_resolve_serves_an_unassigned_unique_name(tmp_path: Path):
    """Unassign leaves ``assignment is None``, which used to be an outright reject."""
    ref = _blob(tmp_path, "000000.png")
    _store(tmp_path, [{"id": "000000", "name": "a.png", "image_ref": ref}])
    _labelset(tmp_path, {})
    got = bi.resolve_bank_image(tmp_path, "normal", "a.png")
    assert got is not None and got.name == "000000.png"


def test_resolve_still_refuses_an_ambiguous_name_in_the_wrong_tier(tmp_path: Path):
    """Deliberate: the store allows one filename in two tiers.

    Serving the wrong one would put a defect where the viewer expects a good
    part, so the name-only rule fires only when the name is unique.
    """
    r1 = _blob(tmp_path, "000000.png")
    r2 = _blob(tmp_path, "000001.png")
    _store(
        tmp_path,
        [
            {"id": "000000", "name": "same.png", "image_ref": r1},
            {"id": "000001", "name": "same.png", "image_ref": r2},
        ],
    )
    _labelset(
        tmp_path,
        {
            "000000": {"tier": "normal", "label": ""},
            "000001": {"tier": "critical", "label": "kizu"},
        },
    )
    assert bi.resolve_bank_image(tmp_path, "negative", "same.png") is None
    got = bi.resolve_bank_image(tmp_path, "critical", "same.png")
    assert got is not None and got.name == "000001.png"


def test_resolve_returns_none_for_a_traversing_name(tmp_path: Path):
    assert bi.resolve_bank_image(tmp_path, "normal", "../../etc/passwd") is None


def test_synthetic_row_names_are_not_images(tmp_path: Path):
    ref = _blob(tmp_path, "000000.png")
    _store(tmp_path, [{"id": "000000", "name": "__unindexed__0", "image_ref": ref}])
    assert bi.bank_census(tmp_path).images == 0


# ---- image_ref containment -------------------------------------------------


def test_resolve_image_ref_admits_a_migrated_ref_but_refuses_an_escape(tmp_path: Path):
    """A migrated entry legitimately points into _images/; bank.npy never is.

    ``/store/delete`` clamped this to ``store/img/`` only, so a migrated
    entry's source image was never reclaimed.
    """
    migrated = f"{bi.IMAGES_SUBDIR}/normal/a.png"
    assert bi.resolve_image_ref(tmp_path, migrated) is not None
    assert bi.resolve_image_ref(tmp_path, migrated, owned_only=True) is not None
    assert bi.resolve_image_ref(tmp_path, "../../etc/passwd") is None
    assert bi.resolve_image_ref(tmp_path, "bank.npy") is not None
    assert bi.resolve_image_ref(tmp_path, "bank.npy", owned_only=True) is None
    assert bi.resolve_image_ref(tmp_path, "") is None


def test_resolve_image_ref_does_not_require_the_file_to_exist(tmp_path: Path):
    """The deleting caller still wants to drop the entry's cached renditions."""
    ref = f"{bi.STORE_SUBDIR}/{bi.STORE_IMAGES_SUBDIR}/gone.png"
    assert bi.resolve_image_ref(tmp_path, ref, owned_only=True) is not None


# ---- the URL ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lot#3.png", "/bank/images/normal/lot%233.png"),
        ("50%off.png", "/bank/images/normal/50%25off.png"),
        ("a b.png", "/bank/images/normal/a%20b.png"),
        ("q?x.png", "/bank/images/normal/q%3Fx.png"),
        ("豆.png", "/bank/images/normal/%E8%B1%86.png"),
    ],
)
def test_bank_image_url_encodes_the_name(name: str, expected: str):
    assert bi.bank_image_url("normal", name) == expected


def test_bank_image_url_carries_the_entry_id_when_there_is_one():
    assert bi.bank_image_url("normal", "a.png", entry_id="000007") == (
        "/bank/images/normal/a.png?id=000007"
    )


# ---- the thumbnail ---------------------------------------------------------


def test_first_bank_image_never_opens_the_store_index(tmp_path: Path):
    """The perf choice is load-bearing: this runs per project on every rebuild.

    Fails the moment someone "simplifies" the blob listing into an index read.
    """
    bank = tmp_path / "b1"
    _blob(bank, "000000.png")
    (bank / bi.STORE_SUBDIR / bi.STORE_INDEX_FILE).write_bytes(b"{{{ not json")
    found = bi.first_bank_image(tmp_path)
    assert found is not None and found.name == "000000.png"


@pytest.mark.parametrize("ext", sorted(bi.DISPLAYABLE_EXTS))
def test_first_bank_image_accepts_every_displayable_extension(tmp_path: Path, ext: str):
    """.webp was missing, so an all-WebP project had no thumbnail at all."""
    bank = tmp_path / "b1"
    _blob(bank, f"000000{ext}")
    found = bi.first_bank_image(tmp_path)
    assert found is not None and found.suffix == ext


def test_first_bank_image_skips_a_tiff_it_cannot_draw(tmp_path: Path):
    bank = tmp_path / "b1"
    staging = bank / bi.STAGING_SUBDIR
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "scan.tif").write_bytes(b"II*\x00")
    assert bi.first_bank_image(tmp_path) is None
    assert bi.bank_census(bank).images == 1, "counted, just not thumbnailed"


def test_first_bank_image_prefers_the_taught_copy_and_the_normal_tier(tmp_path: Path):
    bank = tmp_path / "b1"
    _taught(bank, "critical", "ng.png")
    _taught(bank, "normal", "ok.png")
    _blob(bank, "000000.png")
    found = bi.first_bank_image(tmp_path)
    assert found is not None and found.name == "ok.png"


def test_first_bank_image_skips_a_tombstoned_bank(tmp_path: Path):
    bank = tmp_path / "b1"
    _blob(bank, "000000.png")
    (bank / bi.DELETED_MARKER).write_text("", encoding="utf-8")
    assert bi.first_bank_image(tmp_path) is None


# ---- the census ------------------------------------------------------------


def test_census_counts_an_unassembled_store(tmp_path: Path):
    """The card painted a thumbnail above "0 images" for exactly this shape."""
    _store(
        tmp_path,
        [{"id": f"{i:06d}", "name": f"a{i}.png", "image_ref": _blob(tmp_path, f"{i:06d}.png")}
         for i in range(5)],
    )
    c = bi.bank_census(tmp_path)
    assert (c.images, c.labeled, c.basis) == (5, 0, "store")


def test_census_counts_the_unassigned_remainder_after_assembly(tmp_path: Path):
    entries = [
        {"id": f"{i:06d}", "name": f"a{i}.png", "image_ref": _blob(tmp_path, f"{i:06d}.png")}
        for i in range(10)
    ]
    _store(tmp_path, entries)
    _labelset(tmp_path, {f"{i:06d}": {"tier": "normal", "label": ""} for i in range(6)})
    c = bi.bank_census(tmp_path)
    assert (c.images, c.labeled) == (10, 6), "the unassigned remainder stays visible"


def test_census_counts_duplicate_names_separately(tmp_path: Path):
    """bank_meta de-dupes the name list, so name-based counters under-report."""
    _store(
        tmp_path,
        [
            {"id": "000000", "name": "img001.png", "image_ref": _blob(tmp_path, "000000.png")},
            {"id": "000001", "name": "img001.png", "image_ref": _blob(tmp_path, "000001.png")},
        ],
    )
    assert bi.bank_census(tmp_path).images == 2


def test_census_ignores_a_dangling_assignment(tmp_path: Path):
    """A delete that unassigned only the first id used to inflate ``labeled``."""
    _store(
        tmp_path,
        [{"id": "000000", "name": "a.png", "image_ref": _blob(tmp_path, "000000.png")}],
    )
    _labelset(
        tmp_path,
        {
            "000000": {"tier": "normal", "label": ""},
            "deadbee": {"tier": "critical", "label": "kizu"},
        },
    )
    c = bi.bank_census(tmp_path)
    assert c.labeled <= c.images and c.labeled == 1


def test_census_falls_back_to_bank_meta_for_an_append_era_bank(tmp_path: Path):
    (tmp_path / bi.BANK_META_FILE).write_text(
        json.dumps(
            {
                "bank_images": ["a.png", "b.png"],
                "critical_images": {"kizu": ["c.png"]},
                "negative_images": {},
            }
        ),
        encoding="utf-8",
    )
    c = bi.bank_census(tmp_path)
    assert (c.images, c.labeled, c.basis) == (3, 3, "bank_meta")


def test_census_does_not_double_count_a_migrated_bank(tmp_path: Path):
    """Store and bank_meta both populated: the store is a superset, count it once."""
    _store(
        tmp_path,
        [{"id": "000000", "name": "a.png", "image_ref": _blob(tmp_path, "000000.png")}],
    )
    (tmp_path / bi.BANK_META_FILE).write_text(
        json.dumps({"bank_images": ["a.png"], "critical_images": {}, "negative_images": {}}),
        encoding="utf-8",
    )
    assert bi.bank_census(tmp_path).images == 1


def test_census_ignores_non_images_in_staging(tmp_path: Path):
    staging = tmp_path / bi.STAGING_SUBDIR
    staging.mkdir(parents=True, exist_ok=True)
    for junk in ("Thumbs.db", ".DS_Store", "x.tmp", bi.STAGING_META_FILE):
        (staging / junk).write_bytes(b"x")
    (staging / "real.png").write_bytes(b"\x89PNG")
    c = bi.bank_census(tmp_path)
    assert (c.images, c.staged) == (1, 1)


def test_census_counts_a_judged_staged_file_as_labelled(tmp_path: Path):
    staging = tmp_path / bi.STAGING_SUBDIR
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "a.png").write_bytes(b"\x89PNG")
    (staging / "b.png").write_bytes(b"\x89PNG")
    (staging / bi.STAGING_META_FILE).write_text(
        json.dumps({"a.png": "normal"}), encoding="utf-8"
    )
    c = bi.bank_census(tmp_path)
    assert (c.images, c.labeled, c.staged) == (2, 1, 2)


def test_an_empty_bank_counts_zero(tmp_path: Path):
    c = bi.bank_census(tmp_path)
    assert (c.images, c.labeled, c.basis) == (0, 0, "empty")


def test_project_census_sums_banks_and_skips_deleted_ones(tmp_path: Path):
    for name in ("b1", "b2"):
        bank = tmp_path / name
        _store(
            tmp_path / name,
            [{"id": "000000", "name": "a.png", "image_ref": _blob(bank, "000000.png")}],
        )
    dead = tmp_path / "b3"
    _store(dead, [{"id": "000000", "name": "a.png", "image_ref": _blob(dead, "000000.png")}])
    (dead / bi.DELETED_MARKER).write_text("", encoding="utf-8")
    assert bi.project_census(tmp_path).images == 2


def test_project_census_of_a_missing_root_is_zero(tmp_path: Path):
    assert bi.project_census(tmp_path / "nope").images == 0


# ---- unassigned ------------------------------------------------------------


def test_unassigned_is_set_wise_and_never_needs_a_clamp():
    """``len(store) - assigned`` under-reports by the stale count and floors at 0."""
    ids = ["a", "b", "c", "d"]
    assignments = {"a": {}, "stale1": {}, "stale2": {}, "stale3": {}}
    assert bi.unassigned_count(ids, assignments) == 3
    assert max(0, len(ids) - len(assignments)) == 0, "what the old arithmetic said"


# ---- the extension sets ----------------------------------------------------


def test_gif_decodes_nowhere_but_still_displays():
    assert ".gif" not in bi.DECODABLE_EXTS
    assert ".gif" in bi.DISPLAYABLE_EXTS
    assert ".gif" in bi.IMAGE_EXTS


def test_tiff_is_an_image_on_disk_but_never_a_thumbnail():
    assert ".tif" in bi.IMAGE_EXTS and ".tif" in bi.DECODABLE_EXTS
    assert ".tif" not in bi.DISPLAYABLE_EXTS


def test_has_image_ext_is_case_insensitive():
    assert bi.has_image_ext("A.PNG")
    assert bi.has_image_ext(Path("a.JpEg"))
    assert not bi.has_image_ext("a.txt")
