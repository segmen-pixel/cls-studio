# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Crash-safe file writes for bank artifacts.

Every persisted bank artifact (feature ``.npy``, metadata ``.npz``, manifest
JSON) is written via tmp + ``replace`` so a process crash mid-write can never
truncate the previous good copy — a torn ``bank_meta.json`` or tier array
makes the whole bank refuse to load. ``replace`` retries on ``OSError``
because on Windows it fails with a sharing violation while another handle
(browser thumbnail stream, antivirus scan) briefly holds the target.

Each write gets its own temp name. Concurrent writers of one file are normal
here -- FastAPI serves the sync endpoints from a threadpool -- and a temp name
derived only from the destination is shared by every one of them, so the first
``replace`` moves it away and the rest fail with a missing source.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

__all__ = ["replace_with_retry", "atomic_handle", "atomic_save_npy", "atomic_write_text"]


_TMP_SEQ = itertools.count()


def _tmp_path(path: Path) -> Path:
    """A temp sibling of ``path`` that no other writer can be holding.

    Unique per *writer*, not per destination. A shared ``<name>.tmp`` is enough
    for the crash story above but not for two threads writing one file: both
    create it, the first ``replace`` moves it away, and the second finds its own
    source gone (WinError 2). The pid and thread id make the name unique across
    processes and threads, the counter across nested writes on one thread.

    ``.tmp`` stays last so the ``*.tmp`` sweeps that clear what a crashed writer
    left behind (``Bank._save_tier``) keep matching.
    """
    return path.with_name(
        f"{path.name}.{os.getpid()}-{threading.get_ident()}-{next(_TMP_SEQ)}.tmp"
    )


def _discard_tmp(tmp: Path) -> None:
    """Best-effort cleanup of a temp file whose write failed.

    A unique name is never reused, so without this a failed write would leave
    its partial file in the bank directory for good -- the old shared name at
    least got truncated by the next attempt. Errors here are swallowed: the
    write's own exception is the one worth propagating.
    """
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


_SWEPT_DIRS: set[str] = set()
_SWEPT_GUARD = threading.Lock()
_STALE_AFTER_S = 3600.0


def _sweep_stale_tmps(directory: Path) -> None:
    """Drop temps a previous run died holding. Once per directory per process.

    Unique names are what makes concurrent writers safe, and they are also why
    debris is now permanent: the old shared ``<name>.tmp`` was reclaimed by the
    next write's ``open(tmp, "wb")``, so a hard power-off left at most one stale
    file per destination and it healed itself. A unique name is never reused, and
    :func:`_discard_tmp` runs only from the ``except`` below -- it cannot run on
    power loss, which is the exact threat ``_fsync_file`` exists for.

    Nothing else covers most of what this module writes. The only sweep in the
    tree is ``Bank._save_tier``'s, and it globs the tier directories only, so
    ``bank.npy``, ``bank_meta.json``, ``labelsets/``, the store index and
    ``store/feat/`` had none. Without this, a factory PC that loses power during
    a save strands a multi-GB ``bank.npy.<uniq>.tmp`` in the bank root for good:
    the export filters skip it, no loader glob matches it, and nothing reports
    the space. One power cut, one stranded file, forever.

    ONCE PER DIRECTORY PER PROCESS, not per write. The debris is by definition
    from an EARLIER process, so a later scan finds nothing new -- while scanning
    on every write costs a directory listing per ingested image, and
    ``store/feat/`` holds one file per image. A 5,000-image project would pay
    5,000 listings of a 5,000-entry directory to re-ingest.

    AGE-GUARDED, because a live writer in another thread or another process
    holds a temp that matches this glob too. An hour is far longer than any
    single write here and far shorter than the gap between a crash and the next
    run. This is why the sweep may glob ``*.tmp`` rather than only this
    destination's: widening the pattern without the guard is how a sweep starts
    deleting temps out from under a concurrent writer.
    """
    key = str(directory)
    with _SWEPT_GUARD:
        if key in _SWEPT_DIRS:
            return
        _SWEPT_DIRS.add(key)
    cutoff = time.time() - _STALE_AFTER_S
    try:
        candidates = list(directory.glob("*.tmp"))
    except OSError:
        return
    for stale in candidates:
        try:
            if stale.stat().st_mtime < cutoff:
                stale.unlink()
        except OSError:
            pass  # gone already, or held open -- the next run tries again


def _fsync_file(fh) -> None:
    """Flush + fsync before the rename: tmp+replace alone only survives a
    process crash — on OS crash / hard power-off the filesystem may persist
    the rename before the data blocks, leaving a zero-length "atomic" file.
    Factory PCs get hard-powered routinely, so bank artifacts pay the cost."""
    fh.flush()
    os.fsync(fh.fileno())


def replace_with_retry(src: Path, dst: Path, attempts: int = 5) -> None:
    """Replace ``dst`` with ``src``, retrying on Windows sharing violations."""
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except FileNotFoundError:
            # Not a sharing violation: the source (or the destination's parent)
            # is gone, and no amount of waiting brings it back. Retrying only
            # delayed the report, and delay reads as a slow server -- the
            # shared-temp-name collision this used to mask spent ~1s of sleeps
            # here before the UI showed its toast.
            raise
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.1 * (i + 1))


@contextmanager
def atomic_handle(path: Path, mode: str = "wb", *, encoding: str | None = None):
    """Yield a writable handle whose contents land on ``path`` atomically.

    The single place the temp + fsync + replace sequence lives, so no caller
    has to restate the temp-naming rule -- restating it is what left
    ``IncidentMetaArray.save`` sharing one temp name per destination after the
    same bug had already been fixed in the two helpers below.

    Callers get a *handle* rather than passing a finished buffer because
    ``np.save``/``np.savez`` auto-append their extension to string paths that
    don't already carry it, which would turn ``foo.npy.<uniq>.tmp`` into
    ``foo.npy.<uniq>.tmp.npy`` and leave the ``replace()`` with nothing to move.
    """
    path = Path(path)
    _sweep_stale_tmps(path.parent)
    tmp = _tmp_path(path)
    try:
        with open(tmp, mode, encoding=encoding) as fh:
            yield fh
            _fsync_file(fh)
        replace_with_retry(tmp, path)
    except BaseException:
        _discard_tmp(tmp)
        raise


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """``np.save`` via tmp + replace."""
    with atomic_handle(path) as fh:
        np.save(fh, arr)


def atomic_write_text(path: Path, text: str) -> None:
    """UTF-8 text write via tmp + replace."""
    with atomic_handle(path, "w", encoding="utf-8") as fh:
        fh.write(text)
