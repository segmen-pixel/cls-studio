# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Eval-cache lifecycle across bank edits + runtime-config staleness.

Raw eval scores depend only on the normal bank, so labelled-tier edits must
NOT throw the sweep away: adding an NG leaves everything valid, deleting one
purges just its entries. Normal-tier edits roll the fingerprint (full
invalidation). A saved runtime config flags itself stale after any content
change so the Operator can warn instead of silently running an outdated
threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _f32(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _seed_bank(client, project_id: str) -> Path:
    """Two critical images (a.png rows 0..3, b.png rows 4..7) + 8 normal rows."""
    from clscore.bank import Bank

    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200
    bank_dir = Path(r.json()["bank_dir"])
    b = Bank(normal=_f32(8, seed=1))
    b.append("critical", _f32(4, seed=2), label="scratch", image_name="a.png")
    b.append("critical", _f32(4, seed=3), label="scratch", image_name="b.png")
    b.save(bank_dir)
    r = client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert r.status_code == 200
    return bank_dir


def _fake_eval(name: str, patches: int = 4) -> dict:
    return {
        "name": name, "tier": "critical", "label": "scratch", "patches": patches,
        "score_max": 9.0, "score_p99": 8.0, "score_mean": 1.0,
        "top_scores": [9.0, 8.0], "top_indices": [1, 0],
    }


def _inject_cache(entries: dict) -> None:
    from app.core import cls_eval_cache as eval_cache_mod
    from app.core.cls_state import get_state

    state = get_state()
    cache = eval_cache_mod._eval_cache_for(state)
    cache.update(entries)
    eval_cache_mod._eval_cache_save(state, cache)


# ---- fingerprint semantics ---------------------------------------------------


def test_labelled_tier_edits_do_not_roll_eval_fingerprint():
    from app.core import cls_eval_cache as eval_cache_mod
    from clscore.bank import Bank

    b = Bank(normal=_f32(8, seed=1))
    b.meta.normal_image_index.append({"name": "n.png", "start": 0, "count": 8})
    fp = eval_cache_mod._eval_fingerprint(b)
    b.append("critical", _f32(4, seed=2), label="scratch", image_name="a.png")
    assert eval_cache_mod._eval_fingerprint(b) == fp  # NG teach keeps the sweep
    b.append("normal", _f32(2, seed=3), image_name="n2.png")
    assert eval_cache_mod._eval_fingerprint(b) != fp  # OK teach invalidates


def test_content_fingerprint_covers_marks():
    from app.core import cls_eval_cache as eval_cache_mod
    from clscore.bank import Bank

    b = Bank(normal=_f32(8, seed=1))
    b.append("critical", _f32(4, seed=2), label="scratch", image_name="a.png")
    fp = eval_cache_mod._bank_content_fingerprint(b)
    b.set_image_annotation("critical", "scratch", "a.png", [1],
                           rects=[{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}])
    assert eval_cache_mod._bank_content_fingerprint(b) != fp  # marks flip it


# ---- delete purges only the deleted entries ---------------------------------


def test_critical_delete_purges_only_its_eval_entries(client, project_id):
    _seed_bank(client, project_id)
    _inject_cache({
        "critical/scratch/a.png": _fake_eval("a.png"),
        "critical/scratch/b.png": _fake_eval("b.png"),
    })
    r = client.post("/api/v1/bank/images/delete",
                    json={"tier": "critical", "label": "scratch", "names": ["a.png"]})
    assert r.status_code == 200
    names = {e["name"] for e in client.get("/api/v1/bank/evaluation/cached").json()}
    assert names == {"b.png"}  # b's eval survived the delete


def test_clear_tier_purges_the_whole_tier(client, project_id):
    _seed_bank(client, project_id)
    _inject_cache({
        "critical/scratch/a.png": _fake_eval("a.png"),
        "critical/scratch/b.png": _fake_eval("b.png"),
    })
    assert client.post("/api/v1/bank/clear/critical").status_code == 200
    assert client.get("/api/v1/bank/evaluation/cached").json() == []


# ---- runtime-config staleness ------------------------------------------------


def test_runtime_config_reports_stale_after_bank_edit(client, project_id):
    from clscore.bank import Bank

    bank_dir = _seed_bank(client, project_id)
    put = client.put("/api/v1/bank/runtime-config", json={
        "topk": 10, "k": 5, "alpha": 100.0, "beta": 0.0,
        "exemplar_alpha": True, "threshold": 30.0,
    })
    assert put.status_code == 200 and put.json()["bank_fingerprint"]
    assert client.get("/api/v1/bank/runtime-config").json()["stale"] is False

    # Teach one more NG behind the API's back, reload — recipe is now stale.
    b = Bank.load(bank_dir)
    b.append("critical", _f32(4, seed=9), label="scratch", image_name="c.png")
    b.save(bank_dir)
    client.post("/api/v1/bank/select", json={"project_id": project_id})
    assert client.get("/api/v1/bank/runtime-config").json()["stale"] is True

    # Re-saving re-stamps and clears the flag.
    client.put("/api/v1/bank/runtime-config", json={
        "topk": 10, "k": 5, "alpha": 100.0, "beta": 0.0,
        "exemplar_alpha": True, "threshold": 30.0,
    })
    assert client.get("/api/v1/bank/runtime-config").json()["stale"] is False
