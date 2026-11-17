# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Many threads writing one file: each one's temp file must survive to its replace.

Concurrent writers of a single path are ordinary here. FastAPI serves the sync
endpoints from a threadpool, and opening a freshly created project fires
``/bank/select``, ``/labelsets`` and ``/store`` on mount -- all three of which
lazily create the default label set. With a temp name derived only from the
destination (``standard.json.tmp``) every one of them wrote the *same* temp file
and then called ``replace``: the first consumed it, the rest found their own
source gone (``FileNotFoundError``, WinError 2). v0.2.2 showed that as

    bank select failed: An internal error occurred.

roughly a second after the click, because ``replace_with_retry`` treated the
missing source as a sharing violation and slept through five attempts first.

Which writer loses is a scheduling question, so the burst below is deliberately
wide (8 writers, 3 rounds): measured against the shared-name version it failed
every round at that width, and 1 round in 20 at 4 writers. The invariants that
do *not* depend on winning a race -- a per-write temp name, and no temp file
left behind by a failed write -- are asserted separately below, so a regression
cannot slip through on a lucky interleaving.
"""
from __future__ import annotations

import json
import pathlib
import threading

import numpy as np
import pytest

from clscore import fsio
from clscore.fsio import atomic_save_npy, atomic_write_text

WRITERS = 8
ROUNDS = 3


def _burst(monkeypatch, write, writers: int = WRITERS) -> list[BaseException]:
    """Run ``write(i)`` on ``writers`` threads, all poised to replace at once.

    ``_fsync_file`` is the last thing each writer does inside its ``open``
    block, so a barrier there lines every writer up with its temp file written
    and no replace yet -- the state the collision needs.
    """
    gate = threading.Barrier(writers, timeout=30)
    real_fsync = fsio._fsync_file

    def fsync_then_wait(fh):
        real_fsync(fh)
        gate.wait()

    monkeypatch.setattr(fsio, "_fsync_file", fsync_then_wait)

    errors: list[BaseException] = []

    def wrapped(i: int) -> None:
        try:
            write(i)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a writer never finished"
    return errors


def test_concurrent_text_writers_do_not_consume_each_others_temp_file(tmp_path, monkeypatch):
    for rnd in range(ROUNDS):
        d = tmp_path / f"round{rnd}"
        d.mkdir()
        path = d / "standard.json"

        errors = _burst(monkeypatch, lambda i: atomic_write_text(path, json.dumps({"writer": i})))

        assert errors == [], f"round {rnd}: writers collided: {errors!r}"
        # Last writer wins, but what lands is one writer's whole document.
        assert json.loads(path.read_text(encoding="utf-8"))["writer"] in range(WRITERS)
        assert not list(d.glob("*.tmp")), f"round {rnd}: temp files left behind"


def test_concurrent_npy_writers_do_not_consume_each_others_temp_file(tmp_path, monkeypatch):
    """The serious half: concurrent bank saves are how features get written."""
    for rnd in range(ROUNDS):
        d = tmp_path / f"round{rnd}"
        d.mkdir()
        path = d / "bank.npy"

        errors = _burst(
            monkeypatch, lambda i: atomic_save_npy(path, np.full((4, 8), i, dtype=np.float16))
        )

        assert errors == [], f"round {rnd}: writers collided: {errors!r}"
        arr = np.load(path)
        assert arr.shape == (4, 8)
        assert len(set(arr.ravel().tolist())) == 1, "torn array: rows from two writers"
        assert not list(d.glob("*.tmp")), f"round {rnd}: temp files left behind"


def test_every_write_gets_its_own_temp_name_still_ending_in_tmp(tmp_path, monkeypatch):
    """The invariant behind the fix, asserted without racing anything.

    ``.tmp`` has to stay last: ``Bank._save_tier`` sweeps ``*.tmp`` to clear
    what a crashed writer left, and a unique name that stopped matching that
    glob would accumulate in the bank directory instead.
    """
    used: list[str] = []
    real_fsync = fsio._fsync_file

    def record(fh):
        used.append(str(fh.name))
        real_fsync(fh)

    monkeypatch.setattr(fsio, "_fsync_file", record)
    dst = tmp_path / "bank_meta.json"
    atomic_write_text(dst, "{}")
    atomic_write_text(dst, "{}")
    atomic_save_npy(tmp_path / "bank.npy", np.zeros((2, 2), dtype=np.float16))

    assert len(used) == 3
    assert all(name.endswith(".tmp") for name in used), used
    assert len(set(used)) == 3, f"a temp name was reused: {used}"
    assert str(dst) + ".tmp" not in used, "still the destination-derived name"
    assert all(pathlib.Path(name).parent == tmp_path for name in used), used


def test_a_failed_write_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    """A unique name is never reused, so nothing overwrites a crashed write's temp."""

    def _boom(src, dst, attempts=5):
        raise OSError("simulated crash between temp-write and replace")

    monkeypatch.setattr(fsio, "replace_with_retry", _boom)
    with pytest.raises(OSError):
        atomic_write_text(tmp_path / "bank_meta.json", "{}")
    with pytest.raises(OSError):
        atomic_save_npy(tmp_path / "bank.npy", np.zeros((2, 2), dtype=np.float16))

    assert list(tmp_path.iterdir()) == [], "a failed write littered the directory"


def test_replace_with_retry_does_not_retry_a_missing_source(tmp_path, monkeypatch):
    """The retry is for sharing violations. A vanished source is a different fault.

    Retrying bought five sleeps and then raised the same error, which is why the
    toast for the collision above arrived about a second after the click.
    """
    slept: list[float] = []
    monkeypatch.setattr(fsio.time, "sleep", slept.append)

    with pytest.raises(FileNotFoundError):
        fsio.replace_with_retry(tmp_path / "gone.tmp", tmp_path / "dst.json")

    assert slept == [], "slept on the way out of a missing source"


def test_replace_with_retry_still_retries_a_sharing_violation(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(fsio.time, "sleep", slept.append)
    src = tmp_path / "src.tmp"
    src.write_text("payload", encoding="utf-8")
    dst = tmp_path / "dst.json"

    attempts: list[object] = []
    real_replace = pathlib.Path.replace

    def flaky(self, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError(32, "The process cannot access the file")
        return real_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", flaky)
    fsio.replace_with_retry(src, dst)

    assert len(attempts) == 3
    assert len(slept) == 2
    assert dst.read_text(encoding="utf-8") == "payload"


# ---- crash debris ----------------------------------------------------------
# Unique names are what makes the writes above safe, and they are also why a
# temp is now permanent once its writer dies: the old shared "<name>.tmp" was
# truncated by the next write, so a hard power-off left one stale file per
# destination and it healed itself. _discard_tmp only runs from the except
# clause, so it cannot run on power loss -- the case _fsync_file exists for.
# Bank._save_tier's glob is the only sweep in the tree and it covers the tier
# directories, not the bank root, labelsets/, the store index or store/feat/.


@pytest.fixture
def fresh_process():
    """A process that has never swept anything.

    Requested by name rather than autouse: an autouse fixture touching
    ``fsio._SWEPT_DIRS`` would ERROR every test in this file on a tree where
    the sweep does not exist, which buries the four that are actually meant to
    fail there.
    """
    fsio._SWEPT_DIRS.clear()
    yield
    fsio._SWEPT_DIRS.clear()


def _age(p: pathlib.Path, seconds: float) -> None:
    import os
    old = fsio.time.time() - seconds
    os.utime(p, (old, old))


def test_a_temp_a_crashed_run_left_behind_is_swept(tmp_path, fresh_process):
    """The multi-GB bank.npy.<uniq>.tmp case: nothing else would ever remove it."""
    debris = tmp_path / "bank.npy.4242-1234-0.tmp"
    debris.write_bytes(b"half a bank")
    _age(debris, 4 * 3600)

    atomic_save_npy(tmp_path / "bank.npy", np.zeros(4, dtype=np.float16))

    assert not debris.exists(), "crash debris survived the next write"
    assert (tmp_path / "bank.npy").exists()


def test_a_live_writers_temp_is_left_alone(tmp_path, fresh_process):
    """The guard that makes a *.tmp sweep safe at all: age, not name."""
    live = tmp_path / "bank_meta.json.9999-8888-3.tmp"
    live.write_text("another thread is mid-write", encoding="utf-8")

    atomic_write_text(tmp_path / "bank_meta.json", "{}")

    assert live.exists(), "swept a temp young enough to belong to a live writer"


def test_the_sweep_costs_one_listing_per_directory(tmp_path, monkeypatch, fresh_process):
    """store/feat/ holds one file per image; a scan per write is quadratic.

    Guards the decision, not just the behaviour: a later 'simplification' that
    drops the once-per-directory memo would make importing a 5,000-image
    project pay 5,000 listings of a 5,000-entry directory.
    """
    listings: list[str] = []
    real_glob = pathlib.Path.glob

    def counting_glob(self, pattern):
        if pattern == "*.tmp":
            listings.append(str(self))
        return real_glob(self, pattern)

    monkeypatch.setattr(pathlib.Path, "glob", counting_glob)

    for i in range(12):
        atomic_write_text(tmp_path / f"entry_{i}.json", "{}")

    assert listings == [str(tmp_path)], f"swept {len(listings)} times, want 1"


def test_the_sweep_survives_a_directory_that_is_not_there_yet(tmp_path, fresh_process):
    """The first write into store/feat/ creates it; the sweep must not object."""
    nested = tmp_path / "store" / "feat"
    nested.mkdir(parents=True)
    atomic_save_npy(nested / "000000.npy", np.zeros(2, dtype=np.float16))
    assert (nested / "000000.npy").exists()
