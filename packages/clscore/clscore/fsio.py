# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Crash-safe file writes for bank artifacts.

Every persisted bank artifact (feature ``.npy``, metadata ``.npz``, manifest
JSON) is written via tmp + ``replace`` so a process crash mid-write can never
truncate the previous good copy — a torn ``bank_meta.json`` or tier array
makes the whole bank refuse to load. ``replace`` retries on ``OSError``
because on Windows it fails with a sharing violation while another handle
(browser thumbnail stream, antivirus scan) briefly holds the target.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

__all__ = ["replace_with_retry", "atomic_save_npy", "atomic_write_text"]


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
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.1 * (i + 1))


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """``np.save`` via tmp + replace.

    The tmp file gets an open handle rather than a string path: numpy
    auto-appends ``.npy`` to string paths that don't end in it, which would
    silently turn ``foo.npy.tmp`` into ``foo.npy.tmp.npy`` and break the
    subsequent ``replace()`` (same trick as ``IncidentMetaArray.save``).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
        _fsync_file(fh)
    replace_with_retry(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    """UTF-8 text write via tmp + replace."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        _fsync_file(fh)
    replace_with_retry(tmp, path)
