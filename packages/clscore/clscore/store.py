# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Immutable, tier-agnostic per-image feature store.

Teaching used to do two things in one call: run DINOv2 over an image
(seconds, GPU) and decide which tier that image belongs to (a judgement
call, free). Fusing them made the *cheap* decision carry the *expensive*
one's cost, so re-deciding — the operation that actually determines what
the model learns — meant re-extracting. Splitting them is the whole point
of this module:

    ingest    image -> DINOv2 -> FeatureStore    heavy, once per image
    label     image -> tier / label              free, as often as you like
    assemble  store + labelset -> Bank           numpy only

The store knows an image's patch features and the geometry they came from,
and nothing about whether the operator considers the image good or bad.
:mod:`clscore.labelset` holds that judgement; :mod:`clscore.assemble` joins
the two back into a :class:`clscore.bank.Bank`.

Disk layout, relative to the store directory::

    store_index.json     manifest: StoreMeta + one StoreEntry per image
    feat/000000.npy      [rows, D] fp16 — that image's patch features
    feat/000001.npy
    ...

One file per image rather than one concatenated array, because the image is
the unit of every operation the store supports (ingest, delete, assemble a
subset). A single array would turn a delete into an O(total) rewrite of tens
of gigabytes, which is exactly the cost this split exists to remove.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bank import BANK_DTYPE
from .feature_extractor import DEFAULT_DINO_DIM, DEFAULT_DINO_NAME
from .fsio import atomic_save_npy, atomic_write_text
from .sw import DINO_PATCH, WINDOW_SIZE, WINDOW_STRIDE

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURES_SUBDIR",
    "STORE_INDEX_FILE",
    "STORE_SUBDIR",
    "FeatureStore",
    "StoreEntry",
    "StoreMeta",
]

STORE_SUBDIR = "store"
STORE_INDEX_FILE = "store_index.json"
FEATURES_SUBDIR = "feat"
IMAGES_SUBDIR = "img"

# Width of the zero-padded entry id. Ids are opaque handles, never derived
# from the filename: two images taught under different tiers can legitimately
# share a name (``_images/normal/a.png`` and ``_images/critical/a.png`` both
# exist today), and a name-derived id would silently collapse them into one.
_ID_WIDTH = 6


@dataclass
class StoreEntry:
    """One ingested image: where its features are and what geometry made them."""

    id: str
    name: str
    rows: int
    # Patches the image's sliding-window grid yields at the store's geometry.
    # Defect marks address this grid, not the stored rows, so it has to be
    # recorded even when the two happen to be equal.
    grid_rows: int = 0
    # Which grid patches survived the per-image coreset cap, in stored-row
    # order. ``None`` means "row i is patch i of the grid" — the identity map
    # every uncapped image uses.
    kept: list[int] | None = None
    # Path to the source image bytes, relative to the BANK directory (not the
    # store): migrated images stay where they are under ``_images/<tier>/``
    # while freshly ingested ones land in ``store/img/``. Empty when the
    # source was not kept.
    image_ref: str = ""
    height: int = 0
    width: int = 0
    ingested_at: int = 0
    # Manual validation group (lot, shooting session). A property of the
    # IMAGE, not of the judgement, so it lives here rather than on the label
    # set: two label sets over the same store describe the same photographs.
    group: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        if d.get("kept") is None:
            d.pop("kept")
        return d


@dataclass
class StoreMeta:
    """How every feature array in this store was produced.

    Assembly stamps these onto the ``BankMeta`` it builds, and ingest refuses
    to mix geometries: features extracted at a different window/stride are not
    comparable to the ones already here, and a bank assembled from a mixture
    would score them against each other as if they were.
    """

    model: str = DEFAULT_DINO_NAME
    dim: int = DEFAULT_DINO_DIM
    layers: list[int] | None = None
    window: int = WINDOW_SIZE
    stride: int = WINDOW_STRIDE
    patch: int = DINO_PATCH
    next_id: int = 0

    def geometry(self) -> dict[str, int]:
        return {"window_size": int(self.window), "stride": int(self.stride), "patch": int(self.patch)}

    def compatible_with(self, other: StoreMeta) -> bool:
        return (
            self.model == other.model
            and int(self.dim) == int(other.dim)
            and int(self.window) == int(other.window)
            and int(self.stride) == int(other.stride)
            and int(self.patch) == int(other.patch)
        )


class FeatureStore:
    """Append-only collection of per-image feature arrays plus its manifest.

    The in-memory object holds only the manifest; feature arrays are read from
    disk on demand (:meth:`features_of`). A production store runs to tens of
    gigabytes, so anything that loads every array at once — assembly included
    — has to stream, and making the arrays lazy by construction is what stops
    a casual ``for e in store`` from paging all of it in.
    """

    def __init__(
        self,
        directory: Path,
        meta: StoreMeta | None = None,
        entries: list[StoreEntry] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.meta = meta or StoreMeta()
        self.entries: list[StoreEntry] = list(entries or [])

    # ---- load / save ------------------------------------------------------

    @classmethod
    def load(cls, directory: Path) -> FeatureStore:
        """Read the manifest. Missing directory / index yields an empty store.

        A store that does not exist yet is not an error: every bank predates
        this layer, and the API creates the store lazily on first ingest.
        """
        directory = Path(directory)
        path = directory / STORE_INDEX_FILE
        if not path.exists():
            return cls(directory)
        data = json.loads(path.read_text(encoding="utf-8"))
        meta_raw = data.get("meta", {})
        meta = StoreMeta(
            **{k: meta_raw[k] for k in meta_raw if k in StoreMeta.__dataclass_fields__}
        )
        entries = [
            StoreEntry(**{k: e[k] for k in e if k in StoreEntry.__dataclass_fields__})
            for e in data.get("entries", [])
        ]
        return cls(directory, meta=meta, entries=entries)

    def save_index(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": asdict(self.meta),
            "entries": [e.to_json() for e in self.entries],
        }
        atomic_write_text(
            self.directory / STORE_INDEX_FILE,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

    # ---- paths ------------------------------------------------------------

    def features_dir(self) -> Path:
        return self.directory / FEATURES_SUBDIR

    def images_dir(self) -> Path:
        return self.directory / IMAGES_SUBDIR

    def feature_path(self, entry_id: str) -> Path:
        return self.features_dir() / f"{entry_id}.npy"

    # ---- lookup -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def by_id(self, entry_id: str) -> StoreEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def ids(self) -> list[str]:
        return [e.id for e in self.entries]

    def total_rows(self) -> int:
        return sum(int(e.rows) for e in self.entries)

    def features_of(self, entry: StoreEntry | str) -> np.ndarray:
        """Load one image's feature array from disk.

        Raises ``FileNotFoundError`` when the manifest names an array that is
        not on disk. That is a real inconsistency, not a recoverable one: a
        silently-empty array would assemble a bank that is missing an image
        the operator can still see listed, and the missing rows would look
        like a labelling mistake rather than a broken store.
        """
        e = self.by_id(entry) if isinstance(entry, str) else entry
        if e is None:
            raise KeyError(f"no store entry: {entry!r}")
        return np.load(self.feature_path(e.id))

    # ---- ingest -----------------------------------------------------------

    def add(
        self,
        features: np.ndarray,
        name: str,
        *,
        grid_rows: int = 0,
        kept: np.ndarray | list[int] | None = None,
        image_ref: str = "",
        height: int = 0,
        width: int = 0,
        ingested_at: int = 0,
        entry_id: str | None = None,
    ) -> StoreEntry:
        """Write one image's features into the store and index them.

        The array is persisted immediately (atomically) rather than buffered:
        ingest is the expensive step, and a crash after a 30-image batch must
        not throw away the 29 that already finished. ``save_index`` still has
        to be called by the caller — batching the manifest write is what keeps
        a 500-image ingest from rewriting the JSON 500 times.
        """
        features = np.asarray(features).astype(BANK_DTYPE, copy=False)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D [rows, dim], got shape {features.shape}")
        if features.size and int(features.shape[1]) != int(self.meta.dim):
            raise ValueError(
                f"feature dim {int(features.shape[1])} does not match store dim "
                f"{int(self.meta.dim)} — the store was built with {self.meta.model}"
            )
        if entry_id is None:
            entry_id = f"{int(self.meta.next_id):0{_ID_WIDTH}d}"
            self.meta.next_id = int(self.meta.next_id) + 1
        kept_list: list[int] | None = None
        if kept is not None:
            kept_list = [int(v) for v in np.asarray(kept).ravel()]
            if len(kept_list) != int(features.shape[0]):
                raise ValueError(
                    f"kept map has {len(kept_list)} entries for {int(features.shape[0])} rows"
                )
        entry = StoreEntry(
            id=entry_id,
            name=name,
            rows=int(features.shape[0]),
            grid_rows=int(grid_rows),
            kept=kept_list,
            image_ref=image_ref,
            height=int(height),
            width=int(width),
            ingested_at=int(ingested_at),
        )
        # atomic_save_npy writes through a sibling .tmp and does not create
        # directories - every other caller mkdirs first (Bank.save does), and
        # a store's very first add is exactly the case where it doesn't exist.
        self.features_dir().mkdir(parents=True, exist_ok=True)
        atomic_save_npy(self.feature_path(entry.id), features)
        self.entries.append(entry)
        return entry

    # ---- removal ----------------------------------------------------------

    def remove(self, ids: list[str] | set[str]) -> int:
        """Drop entries and their feature files. Returns how many were removed.

        Unknown ids are ignored — a delete racing another client's delete
        should be a no-op, not a 500.
        """
        target = set(ids)
        keep: list[StoreEntry] = []
        removed = 0
        for e in self.entries:
            if e.id in target:
                self.feature_path(e.id).unlink(missing_ok=True)
                removed += 1
            else:
                keep.append(e)
        self.entries = keep
        return removed

    # ---- integrity --------------------------------------------------------

    def missing_arrays(self) -> list[str]:
        """Ids whose ``.npy`` is absent — the one check assembly cannot skip."""
        return [e.id for e in self.entries if not self.feature_path(e.id).exists()]

    def __repr__(self) -> str:
        return (
            f"FeatureStore(images={len(self.entries)}, rows={self.total_rows()}, "
            f"dim={int(self.meta.dim)}, dir={self.directory})"
        )
