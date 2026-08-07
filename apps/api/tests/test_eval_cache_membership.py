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
