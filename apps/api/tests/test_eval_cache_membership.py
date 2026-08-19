# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The eval cache must not outlive the images it describes.

``/bank/evaluation/cached`` promises "the active bank's current contents" and
returned the raw cache, which is only the same thing while the fingerprint
rolls on every change. ``_eval_fingerprint`` is deliberately normal-tier-only,
so removing a CRITICAL or NEGATIVE image leaves it byte-identical: the orphan
survived the delete, the assemble and every restart, and kept feeding the
separation histogram, the AUROC and the Youden auto-threshold.

The two purge hooks that already existed hang off ``/bank/clear/{tier}`` and
``/bank/images/delete`` — neither of which the shipped UI can reach.
"""

from __future__ import annotations

import numpy as np

from app.core.cls_eval_cache import (
    _eval_cache_for,
    _eval_cache_save,
    bank_eval_keys,
    eval_cache_live_entries,
    eval_cache_reconcile,
)
from clscore.bank import Bank, BankMeta

DIM = 16


def _eval(name: str, tier: str, label: str = "") -> dict:
    return {
        "name": name, "tier": tier, "label": label, "patches": 4,
        "score_max": 1.0, "score_p99": 1.0, "score_mean": 1.0,
        "top_scores": [], "top_indices": [],
    }


class _State:
    def __init__(self, bank: Bank, bank_dir):
        self.bank = bank
        self.bank_dir = bank_dir


def _bank() -> Bank:
    rng = np.random.default_rng(0)
    b = Bank(normal=rng.random((8, DIM)).astype(np.float16), meta=BankMeta(dim=DIM))
    b.append("critical", rng.random((4, DIM)).astype(np.float16),
             label="kizu", image_name="ng1.png")
    b.append("critical", rng.random((4, DIM)).astype(np.float16),
             label="kizu", image_name="ng2.png")
    b.meta.normal_image_index = [{"name": "ok.png", "start": 0, "count": 8}]
    b.meta.bank_images = ["ok.png"]
    return b


def test_bank_eval_keys_names_every_tier(tmp_path):
    keys = bank_eval_keys(_bank())
    assert keys == {"normal//ok.png", "critical/kizu/ng1.png", "critical/kizu/ng2.png"}


def test_a_deleted_critical_image_stops_being_returned(tmp_path):
    """The reported symptom: a deleted defect keeps driving the AUROC."""
    bank = _bank()
    state = _State(bank, tmp_path)
    cache = _eval_cache_for(state)
    for nm in ("ng1.png", "ng2.png"):
        cache[f"critical/kizu/{nm}"] = _eval(nm, "critical", "kizu")
    cache["normal//ok.png"] = _eval("ok.png", "normal")
    _eval_cache_save(state, cache)
    assert len(eval_cache_live_entries(state)) == 3

    # The live path: /store/delete then /bank/assemble. Model it by removing
    # the image from the bank, which is what an assemble produces.
    bank.remove_images("critical", "kizu", ["ng2.png"])

    live = eval_cache_live_entries(state)
    assert "critical/kizu/ng2.png" not in live
    assert len(live) == 2, "the survivors are untouched"
    assert "critical/kizu/ng2.png" in _eval_cache_for(state), (
        "the read-side filter is the invariant; the file is tidied separately"
    )


def test_reconcile_drops_the_orphan_from_the_cache_itself(tmp_path):
    bank = _bank()
    state = _State(bank, tmp_path)
    cache = _eval_cache_for(state)
    cache["critical/kizu/ng1.png"] = _eval("ng1.png", "critical", "kizu")
    cache["critical/kizu/gone.png"] = _eval("gone.png", "critical", "kizu")
    _eval_cache_save(state, cache)

    assert eval_cache_reconcile(state) == 1
    assert set(_eval_cache_for(state)) == {"critical/kizu/ng1.png"}
    assert eval_cache_reconcile(state) == 0, "idempotent"


def test_reconcile_keeps_everything_when_nothing_left(tmp_path):
    bank = _bank()
    state = _State(bank, tmp_path)
    cache = _eval_cache_for(state)
    for k, nm, t, lb in (
        ("normal//ok.png", "ok.png", "normal", ""),
        ("critical/kizu/ng1.png", "ng1.png", "critical", "kizu"),
    ):
        cache[k] = _eval(nm, t, lb)
    _eval_cache_save(state, cache)
    assert eval_cache_reconcile(state) == 0
    assert len(_eval_cache_for(state)) == 2


# ---- identity: a filename is not an image -----------------------------------


def test_the_key_carries_the_store_entry_when_the_bank_has_one():
    from app.core.cls_eval_cache import eval_cache_key

    stamped = {"name": "lot1_003.png", "entry_id": "000007", "start": 0, "count": 4}
    legacy = {"name": "lot1_003.png", "start": 0, "count": 4}
    assert eval_cache_key("critical", "kizu", stamped) == "critical/kizu/lot1_003.png#000007"
    assert eval_cache_key("critical", "kizu", legacy) == "critical/kizu/lot1_003.png"


def test_a_retake_under_the_same_filename_is_a_different_key():
    """Delete lot1_003.png, shoot it again, ingest, label, assemble.

    The only freshness guard used to be patches == count, and on a line where
    every frame comes off one camera at one resolution that compares equal
    every time -- so the retake was served the deleted photo's scores, and its
    ranked top_indices picked the NEW image's rows at the OLD image's
    positions for the exemplar block and the live alpha term.
    """
    from app.core.cls_eval_cache import eval_cache_key

    before = {"name": "lot1_003.png", "entry_id": "000007", "start": 0, "count": 4}
    after = {"name": "lot1_003.png", "entry_id": "000042", "start": 0, "count": 4}
    assert eval_cache_key("critical", "kizu", before) != eval_cache_key(
        "critical", "kizu", after
    )


def test_two_entries_sharing_a_filename_get_two_cache_rows():
    from app.core.cls_eval_cache import eval_cache_key

    a = {"name": "img001.png", "entry_id": "000000", "start": 0, "count": 4}
    b = {"name": "img001.png", "entry_id": "000001", "start": 4, "count": 4}
    assert eval_cache_key("normal", "", a) != eval_cache_key("normal", "", b)


def test_a_legacy_bank_keeps_the_old_key_shape(tmp_path):
    """Banks assembled before the stamp carry no id, so nothing invalidates."""
    bank = _bank()
    for e in bank.meta.critical_image_index["kizu"]:
        e.pop("entry_id", None)
    keys = bank_eval_keys(bank)
    assert all("#" not in k for k in keys)
    assert "critical/kizu/ng1.png" in keys


def test_bank_eval_keys_uses_the_stamp_when_it_is_there():
    bank = _bank()
    for e in bank.meta.critical_image_index["kizu"]:
        e["entry_id"] = "id-" + e["name"]
    keys = bank_eval_keys(bank)
    assert "critical/kizu/ng1.png#id-ng1.png" in keys


def test_purge_still_matches_a_stamped_key_by_name(tmp_path):
    """/bank/clear and the per-image delete address images BY NAME, so the
    purge has to see past the id suffix it did not used to have."""
    from app.core.cls_eval_cache import _eval_cache_purge

    bank = _bank()
    for e in bank.meta.critical_image_index["kizu"]:
        e["entry_id"] = "id-" + e["name"]
    state = _State(bank, tmp_path)
    cache = _eval_cache_for(state)
    for nm in ("ng1.png", "ng2.png"):
        cache[f"critical/kizu/{nm}#id-{nm}"] = _eval(nm, "critical", "kizu")
    _eval_cache_save(state, cache)

    _eval_cache_purge(state, "critical", "kizu", ["ng2.png"])

    left = set(_eval_cache_for(state))
    assert left == {"critical/kizu/ng1.png#id-ng1.png"}


def test_purge_matches_a_filename_that_itself_contains_a_hash(tmp_path):
    """The separator is the LAST "#", not the first.

    Index names are the operator's ORIGINAL filenames, kept verbatim -- pinned
    by test_store_routes.py's "lot#3 50%.png". The purge used to re-derive the
    name with ``kn.split("#", 1)[0]``, which turns BOTH "lot#3.png#000007" and
    "lot#3.png" into "lot" and therefore matched neither. The orphan then
    outlived its own image, and on the one path that escapes every later net --
    delete, then re-teach the same filename via /bank/append with no assemble,
    which stamps no entry_id -- the retake is served the DELETED photo's scores
    and its top_indices index the new image's rows at the old one's positions.

    Both key shapes are checked, because a naive rsplit fixes the stamped one
    and breaks the legacy one.
    """
    from app.core.cls_eval_cache import _eval_cache_purge

    hashed, plain = "lot#3 50%.png", "ng1.png"
    for entry_id in ("000007", ""):  # stamped bank, then a pre-stamp one
        bank = _bank()
        state = _State(bank, tmp_path / f"b{entry_id or 'legacy'}")
        state.bank_dir.mkdir(parents=True, exist_ok=True)
        cache = _eval_cache_for(state)
        suffix = f"#{entry_id}" if entry_id else ""
        cache[f"critical/kizu/{hashed}{suffix}"] = _eval(hashed, "critical", "kizu")
        cache[f"critical/kizu/{plain}{suffix}"] = _eval(plain, "critical", "kizu")
        _eval_cache_save(state, cache)

        _eval_cache_purge(state, "critical", "kizu", [hashed])

        left = set(_eval_cache_for(state))
        assert left == {f"critical/kizu/{plain}{suffix}"}, (
            f"entry_id={entry_id!r}: the '#'-bearing name was not purged"
        )
