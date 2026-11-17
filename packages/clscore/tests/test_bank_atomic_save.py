# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Crash-safe bank persistence: every artifact is written tmp-then-replace.

A process crash mid-``save`` must leave the previous on-disk copy loadable —
a torn ``bank_meta.json`` or tier ``.npy`` makes the whole bank refuse to
load (2026-07-17: a native crash during a bulk teach was one unlucky
scheduling decision away from exactly that).
"""
from __future__ import annotations

import numpy as np
import pytest

from clscore import fsio
from clscore.bank import Bank


def _bank(rows: int = 8, dim: int = 16) -> Bank:
    rng = np.random.default_rng(0)
    b = Bank(normal=rng.standard_normal((rows, dim)).astype(np.float16))
    b.append("critical", rng.standard_normal((4, dim)).astype(np.float16),
             label="scratch", image_name="a.png")
    return b


def test_save_leaves_no_tmp_files(tmp_path):
    b = _bank()
    b.save(tmp_path)
    assert not list(tmp_path.rglob("*.tmp"))
    # Full round-trip still works.
    loaded = Bank.load(tmp_path)
    assert loaded.normal.shape == b.normal.shape
    assert loaded.critical["scratch"].shape == b.critical["scratch"].shape


def test_interrupted_save_preserves_previous_copy(tmp_path, monkeypatch):
    b = _bank()
    b.save(tmp_path)
    before_normal = np.load(tmp_path / "bank.npy")
    before_meta = (tmp_path / "bank_meta.json").read_text(encoding="utf-8")

    # Grow the bank, then crash every replace(): nothing on disk may change.
    b.append("normal", np.zeros((3, 16), dtype=np.float16), image_name="b.png")

    def _boom(src, dst, attempts=5):
        raise OSError("simulated crash between tmp-write and replace")

    # atomic_save_npy / atomic_write_text resolve replace_with_retry through
    # fsio's module namespace at call time, so this one patch covers both.
    monkeypatch.setattr(fsio, "replace_with_retry", _boom)
    with pytest.raises(OSError):
        b.save(tmp_path)

    # The previous copy is untouched and still loads.
    after_normal = np.load(tmp_path / "bank.npy")
    assert after_normal.shape == before_normal.shape
    assert (tmp_path / "bank_meta.json").read_text(encoding="utf-8") == before_meta
    Bank.load(tmp_path)


def test_stale_tmp_leftovers_are_cleaned_on_next_save(tmp_path):
    b = _bank()
    b.save(tmp_path)
    leftover = tmp_path / "critical" / "scratch.npy.tmp"
    leftover.write_bytes(b"torn partial write from a crashed process")
    b.save(tmp_path)
    assert not leftover.exists()
    Bank.load(tmp_path)
