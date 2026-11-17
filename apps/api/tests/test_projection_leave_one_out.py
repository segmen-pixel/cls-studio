# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Leave-own-image-out has to exclude the image's OWN rows.

Normal-tier rows live in the bank they are scored against, so each image's
own slice is masked out first — otherwise every patch finds itself at
distance 0 and the whole map reads "perfectly normal".

The mask used to be looked up by FILENAME. The store deliberately allows two
photographs to share one (a zip with two folders that both hold img001.png),
so the two collapsed onto a single range: each duplicate was scored with the
OTHER one's rows excluded and its own left in the bank. A defect image
therefore plotted, and scored, as unusually normal.
"""

from __future__ import annotations

import numpy as np

import clscore.scoring as scoring_mod
from app.core import cls_projection
from clscore.bank import Bank, BankMeta

DIM = 8

DUPLICATES = [
    {"name": "img001.png", "entry_id": "000000", "start": 0, "count": 4},
    {"name": "img001.png", "entry_id": "000001", "start": 4, "count": 4},
]


class _State:
    """The three members compute_projection touches with scores on."""

    def __init__(self, bank: Bank):
        self.bank = bank

    def get_normal_ivf(self):
        return None, 0

    def get_normal_tensor(self):
        return self.bank.normal


def _bank(index: list[dict], rows: int = 8) -> Bank:
    rng = np.random.default_rng(0)
    bank = Bank(normal=rng.random((rows, DIM)).astype(np.float16), meta=BankMeta(dim=DIM))
    bank.meta.normal_image_index = index
    return bank


def _exclusions(monkeypatch, index: list[dict], rows: int = 8) -> list[tuple[int, int]]:
    """Every (exclude_start, exclude_count) the scorer is handed."""
    seen: list[tuple[int, int]] = []

    def fake_score(features, bank, **kw):
        seen.append((int(kw.get("exclude_start", -1)), int(kw.get("exclude_count", 0))))
        return np.zeros(int(features.shape[0]), dtype=np.float32)

    monkeypatch.setattr(scoring_mod, "score_stored_features", fake_score)
    cls_projection.compute_projection(
        _State(_bank(index, rows)),
        mode="normal",
        max_points_per_tier=100,
        with_scores=True,
        guarantee_top=0,
    )
    return sorted(seen)


def test_two_photographs_with_one_filename_each_exclude_their_own_rows(monkeypatch):
    assert _exclusions(monkeypatch, DUPLICATES) == [(0, 4), (4, 4)]


def test_distinct_filenames_are_unaffected(monkeypatch):
    index = [
        {"name": "a.png", "entry_id": "000000", "start": 0, "count": 4},
        {"name": "b.png", "entry_id": "000001", "start": 4, "count": 4},
    ]
    assert _exclusions(monkeypatch, index) == [(0, 4), (4, 4)]


def test_rows_with_no_index_entry_are_scored_without_exclusion(monkeypatch):
    """Legacy banks have no index; they must still be scored, just not masked."""
    index = [{"name": "a.png", "start": 0, "count": 4}]
    assert _exclusions(monkeypatch, index) == [(-1, 0), (0, 4)]


def test_a_legacy_bank_with_no_index_at_all_still_scores(monkeypatch):
    assert _exclusions(monkeypatch, []) == [(-1, 0)]
