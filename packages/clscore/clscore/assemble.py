# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Join a feature store and a label set back into a scoreable :class:`Bank`.

This is the cheap third of the split described in :mod:`clscore.store`:
concatenation, a severity array, and — only when the result would exceed the
runtime budget — a coreset. No model, no image decoding, numpy only.

Two invariants make the whole design work, and both are enforced here:

**Per-image contiguity.** Every image's rows stay one contiguous block, and
``BankMeta.*_image_index`` records it. Leave-own-image-out evaluation, the
per-image delete and the defect marks all address rows that way, so a
reduction that interleaved images would break three features at once. When
the normal tier overflows its ceiling we therefore shrink images
*individually* (max-min fair, so small images are never starved to feed a
large one) instead of running one global k-Center over the concatenation.

**Marks address the grid, not the rows.** A stored image may already be a
coreset of its own patch grid, and assembly may reduce it again. Both
reductions compose into a single ``kept`` map from stored row to grid patch,
and the operator's marks are translated through it at the very end — so a
mark survives any number of re-assemblies at different capacities.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .bank import (
    BANK_DTYPE,
    INDEX_ENTRY_ID_KEY,
    SOURCE_IMAGES_SUBDIR,
    Bank,
    BankMeta,
    Tier,
    coreset_reduce_indexed,
)
from .fsio import atomic_write_text
from .incident import DEFAULT_SEVERITY, SEVERITY_HEAVY, IncidentMetaArray
from .labelset import (
    DEFAULT_LABELSET_ID,
    DEFAULT_LABELSET_NAME,
    Assignment,
    LabelSet,
)
from .store import FeatureStore, StoreEntry, StoreMeta

logger = logging.getLogger(__name__)

__all__ = [
    "ASSEMBLY_STATE_FILE",
    "assemble_bank",
    "assembly_fingerprint",
    "read_assembly_state",
    "write_assembly_state",
    "migrate_bank_to_store",
    "per_image_budget",
    "roundtrip_diff",
]


def per_image_budget(rows: list[int], ceiling: int) -> list[int]:
    """Max-min fair split of ``ceiling`` rows across images of size ``rows``.

    Every image gets an equal share; images that want less than their share
    release the remainder to the ones that want more. Straight
    ``ceiling // n`` would truncate a 2,000-row image to 700 while a 300-row
    image left 400 of its allowance unused, which is how a bank ends up
    under-filled *and* lopsided at the same time.
    """
    n = len(rows)
    if n == 0:
        return []
    if ceiling <= 0 or sum(rows) <= ceiling:
        return list(rows)
    budget = [0] * n
    remaining = int(ceiling)
    left = n
    # Smallest first: an image that fits under the current fair share is
    # settled immediately and hands its slack to everything still unsized.
    for i in sorted(range(n), key=lambda j: rows[j]):
        share = max(1, remaining // left) if left else 0
        budget[i] = min(int(rows[i]), share)
        remaining -= budget[i]
        left -= 1
    return budget


def _compose_kept(entry: StoreEntry, sub_idx: np.ndarray | None) -> list[int] | None:
    """Grid-patch index per surviving row, or None when it is the identity.

    ``entry.kept`` maps stored rows to grid patches (from the ingest-time
    cap); ``sub_idx`` selects a subset of those stored rows (from the
    assembly-time cap). Composing them is what keeps a mark pointing at the
    same pixels after both reductions.
    """
    base = entry.kept
    if sub_idx is None:
        return list(base) if base is not None else None
    idx = np.asarray(sub_idx, dtype=np.int64)
    if base is None:
        return [int(v) for v in idx]
    arr = np.asarray(base, dtype=np.int64)
    return [int(v) for v in arr[idx]]


def _rows_for_marks(marks: list[int], kept: list[int] | None, count: int) -> list[int]:
    """Translate grid-patch marks into row positions inside the image's slice."""
    if not marks:
        return []
    if kept is None:
        return sorted({int(m) for m in marks if 0 <= int(m) < count})
    where = {int(g): i for i, g in enumerate(kept)}
    return sorted({where[int(m)] for m in marks if int(m) in where})


class _Carry(NamedTuple):
    """Per-image metadata slices of the previous bank, addressable two ways.

    ``by_id`` is keyed (tier, label, store entry id, rows) and is the real
    identity. ``by_name`` is keyed (tier, label, filename, rows) and holds
    ONLY entries that carry no id, so it can never match a stamped one.
    """

    by_id: dict[tuple[str, str, str, int], IncidentMetaArray]
    by_name: dict[tuple[str, str, str, int], IncidentMetaArray]


def _carry_index(bank: Bank | None) -> _Carry:
    """Per-image metadata slices of ``bank``, for the re-assemble to inherit.

    Re-assembling rebuilds the labelled tiers from scratch, which would reset
    hit counts and consolidation tiers — real operational history that no
    label set records. Carrying the slice over whenever an image lands in the
    same tier/label at the same size preserves it; anything that moved gets
    fresh defaults, which is correct, since the history was about a different
    tier.

    "The same image" used to mean the same FILENAME at the same row count.
    Delete lot1_003.png, shoot the part again and import the retake under the
    same name, and both halves match — the row count always does, since every
    frame comes off one camera at one resolution. The deleted part's hit
    counts, severities and consolidation tier were then donated onto a
    photograph they were never measured on. So the store entry id is the key
    now, and the filename is only consulted for entries that predate it.
    """
    out = _Carry({}, {})
    if bank is None:
        return out
    for tier in ("critical", "negative"):
        index = (
            bank.meta.critical_image_index if tier == "critical"
            else bank.meta.negative_image_index
        )
        metas = bank.critical_meta if tier == "critical" else bank.negative_meta
        for label, entries in index.items():
            m = metas.get(label)
            if m is None:
                continue
            for e in entries:
                s, c = int(e.get("start", -1)), int(e.get("count", 0))
                if s < 0 or c <= 0 or s + c > len(m):
                    continue
                slab = m.take(np.arange(s, s + c, dtype=np.int64))
                entry_id = str(e.get(INDEX_ENTRY_ID_KEY, ""))
                if entry_id:
                    out.by_id[(tier, label, entry_id, c)] = slab
                else:
                    out.by_name[(tier, label, str(e.get("name", "")), c)] = slab
    return out


def assemble_bank(
    store: FeatureStore,
    labelset: LabelSet,
    *,
    normal_ceiling: int = 0,
    device: str = "cpu",
    prev_bank: Bank | None = None,
    description: str = "",
) -> Bank:
    """Build a :class:`Bank` from every assigned image in ``store``.

    ``normal_ceiling`` (0 = unbounded) is the total row budget for the normal
    tier; overflow is resolved per image by :func:`per_image_budget` so the
    row index stays exact. ``prev_bank`` donates per-row incident metadata for
    images that did not change tier — see :func:`_carry_index`.

    Images in the store with no assignment contribute nothing: that is how
    "ingested but not labelled yet" is represented, and it is the state the
    labelling tab starts every image in.
    """
    missing = store.missing_arrays()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} store entries have no feature file "
            f"(first: {missing[0]}) — the store is inconsistent, refusing to assemble"
        )

    smeta: StoreMeta = store.meta
    carry = _carry_index(prev_bank)

    # Group in store order so assembly is deterministic and — for a store
    # migrated from an existing bank — reproduces the original row order
    # exactly. Any other order would be equally valid but would make the
    # migration's byte-for-byte round-trip check impossible.
    groups: dict[tuple[str, str], list[tuple[StoreEntry, Assignment]]] = {}
    label_display: dict[str, str] = {}
    for entry in store.entries:
        a = labelset.assignments.get(entry.id)
        if a is None or a.tier not in ("normal", "critical", "negative"):
            continue
        stem = a.resolved_label()
        groups.setdefault((a.tier, stem), []).append((entry, a))
        # Remember what the operator typed, so the stem can be shown back as a
        # name rather than as a hash-suffixed filename.
        if stem and a.label and stem != a.label:
            label_display[stem] = a.label

    normal_parts: list[np.ndarray] = []
    normal_index: list[dict] = []
    normal_names: list[str] = []
    critical_parts: dict[str, list[np.ndarray]] = {}
    negative_parts: dict[str, list[np.ndarray]] = {}
    critical_meta_parts: dict[str, list[IncidentMetaArray]] = {}
    negative_meta_parts: dict[str, list[IncidentMetaArray]] = {}
    critical_index: dict[str, list[dict]] = {}
    negative_index: dict[str, list[dict]] = {}
    critical_names: dict[str, list[str]] = {}
    negative_names: dict[str, list[str]] = {}

    inspection_count = int(prev_bank.meta.inspection_count) if prev_bank is not None else 0

    for (tier, label), items in groups.items():
        # The ceiling governs the normal tier only: it exists to bound the
        # GPU-resident nominal manifold, and the labelled tiers are already
        # bounded per image at ingest.
        budgets = (
            per_image_budget([int(e.rows) for e, _ in items], normal_ceiling)
            if tier == "normal" else [int(e.rows) for e, _ in items]
        )
        cursor = 0
        for (entry, assignment), budget in zip(items, budgets):
            feats = store.features_of(entry).astype(BANK_DTYPE, copy=False)
            sub_idx: np.ndarray | None = None
            if 0 < budget < int(feats.shape[0]):
                feats, sub_idx = coreset_reduce_indexed(
                    feats, budget / float(feats.shape[0]), device
                )
                feats = feats.astype(BANK_DTYPE, copy=False)
            count = int(feats.shape[0])
            if count == 0:
                continue
            kept = _compose_kept(entry, sub_idx)
            # The store entry id is the only stable link from this row range
            # back to the photograph. Names are not: the store deliberately
            # allows two entries to share a filename (two folders in one zip
            # both holding img001.png), and every name-keyed reader collapses
            # them onto whichever comes first. Stamping the id here is what
            # lets those readers tell the two apart.
            index_entry: dict = {
                "name": entry.name,
                INDEX_ENTRY_ID_KEY: entry.id,
                "start": cursor,
                "count": count,
            }
            if kept is not None:
                index_entry["kept"] = kept
            if tier == "normal":
                normal_parts.append(feats)
                normal_index.append(index_entry)
                if entry.name not in normal_names:
                    normal_names.append(entry.name)
            else:
                if assignment.rects:
                    index_entry["annotations"] = list(assignment.rects)
                feats_d = critical_parts if tier == "critical" else negative_parts
                metas_d = critical_meta_parts if tier == "critical" else negative_meta_parts
                index_d = critical_index if tier == "critical" else negative_index
                names_d = critical_names if tier == "critical" else negative_names
                feats_d.setdefault(label, []).append(feats)
                index_d.setdefault(label, []).append(index_entry)
                names = names_d.setdefault(label, [])
                if entry.name not in names:
                    names.append(entry.name)
                # Per-row metadata: carried over verbatim when this IMAGE was
                # already in this tier at this size, otherwise defaults with
                # the label set's severity, with the marks raised to heavy.
                # by_name holds only entries with no id, so a retake under a
                # dead image's filename inherits nothing.
                donor = carry.by_id.get((tier, label, entry.id, count))
                if donor is None:
                    donor = carry.by_name.get((tier, label, entry.name, count))
                if donor is not None:
                    m = donor
                else:
                    m = IncidentMetaArray.defaults_for(count, registered_at=inspection_count)
                    m.severity[:] = assignment.clamped_severity()
                rows = _rows_for_marks(assignment.marks, kept, count)
                if rows:
                    m.severity[np.asarray(rows, dtype=np.int64)] = SEVERITY_HEAVY
                metas_d.setdefault(label, []).append(m)
            cursor += count

    dim = int(smeta.dim) or 0
    normal = (
        np.concatenate(normal_parts, axis=0)
        if normal_parts else np.zeros((0, dim), dtype=BANK_DTYPE)
    )

    def _join(parts: dict) -> dict[str, np.ndarray]:
        return {
            lab: np.concatenate(chunks, axis=0)
            for lab, chunks in parts.items() if chunks
        }

    def _join_meta(parts: dict) -> dict[str, IncidentMetaArray]:
        out: dict[str, IncidentMetaArray] = {}
        for lab, chunks in parts.items():
            if not chunks:
                continue
            merged = IncidentMetaArray(
                severity=np.concatenate([c.severity for c in chunks]),
                registered_at_inspection=np.concatenate(
                    [c.registered_at_inspection for c in chunks]
                ),
                last_hit_at_inspection=np.concatenate(
                    [c.last_hit_at_inspection for c in chunks]
                ),
                hit_count=np.concatenate([c.hit_count for c in chunks]),
                tier=np.concatenate([c.tier for c in chunks]),
            )
            out[lab] = merged
        return out

    meta = BankMeta(
        model=smeta.model,
        dim=dim,
        layers=list(smeta.layers) if smeta.layers else None,
        window=int(smeta.window),
        stride=int(smeta.stride),
        patch=int(smeta.patch),
        n_patches=int(normal.shape[0]),
        # What the normal tier would have been without the ceiling — the ratio
        # of the two is how much resolution the runtime budget cost.
        n_patches_pre_coreset=sum(
            int(e.rows) for e, _ in groups.get(("normal", ""), [])
        ),
        bank_images=normal_names,
        critical_images=critical_names,
        negative_images=negative_names,
        normal_image_index=normal_index,
        critical_image_index=critical_index,
        negative_image_index=negative_index,
        inspection_count=inspection_count,
        label_display=label_display,
        description=description or (prev_bank.meta.description if prev_bank else ""),
    )
    return Bank(
        normal=normal,
        critical=_join(critical_parts),
        negative=_join(negative_parts),
        meta=meta,
        critical_meta=_join_meta(critical_meta_parts),
        negative_meta=_join_meta(negative_meta_parts),
    )


# ---- migration: existing bank -> store + "standard" label set --------------


def _entry_severity(meta: IncidentMetaArray | None, start: int, count: int) -> tuple[int, list[int]]:
    """Recover ``(base_severity, heavy_row_positions)`` from a metadata slice.

    The label set stores a base severity per image plus the marked rows; the
    bank stores one severity per row. Going backwards, the base is whatever
    the unmarked rows agree on (the mode), and the marks are the heavy rows —
    unless *every* row is heavy, which means the operator set the whole image
    to heavy rather than circling a region.
    """
    if meta is None or len(meta) < start + count or count <= 0:
        return DEFAULT_SEVERITY, []
    sev = np.asarray(meta.severity[start : start + count], dtype=np.int64)
    heavy = np.flatnonzero(sev == SEVERITY_HEAVY)
    if heavy.size == sev.size:
        return SEVERITY_HEAVY, []
    rest = sev[sev != SEVERITY_HEAVY]
    base = int(np.bincount(rest).argmax()) if rest.size else DEFAULT_SEVERITY
    return base, [int(v) for v in heavy]


def migrate_bank_to_store(
    bank: Bank,
    store: FeatureStore,
    *,
    bank_dir: Path | None = None,
    labelset_id: str = DEFAULT_LABELSET_ID,
    labelset_name: str = DEFAULT_LABELSET_NAME,
    read_images: bool = True,
) -> tuple[FeatureStore, LabelSet]:
    """Carve an existing bank into per-image store entries plus one label set.

    No feature is recomputed: every row already in the bank is copied out
    through ``BankMeta.*_image_index``, which records exactly which rows came
    from which image. The resulting store + label set re-assemble into a bank
    with byte-identical tier arrays — :func:`roundtrip_diff` is there to prove
    it before the caller commits to the new layout.

    Rows with no index entry (banks built before the index existed) are kept
    as one synthetic entry per tier/label rather than dropped: they are real
    taught data, and losing them silently would be far worse than carrying an
    image the operator cannot name.
    """
    # Trust the arrays over the manifest: ``BankMeta.dim`` is only refreshed on
    # save, so a bank built in memory (or one whose normal tier is empty while
    # a labelled tier is not) can carry the placeholder default while its rows
    # are a different width. Ingest would then reject every row it is being
    # handed for "not matching the store".
    widths = [int(a.shape[1]) for a in
              [bank.normal, *bank.critical.values(), *bank.negative.values()] if a.size]
    store.meta = StoreMeta(
        model=bank.meta.model,
        dim=widths[0] if widths else int(bank.meta.dim),
        layers=list(bank.meta.layers) if bank.meta.layers else None,
        window=int(bank.meta.window),
        stride=int(bank.meta.stride),
        patch=int(bank.meta.patch),
        next_id=int(store.meta.next_id),
    )
    ls = LabelSet(id=labelset_id, name=labelset_name)

    def _dims(image_ref: str) -> tuple[int, int, int]:
        """(grid_rows, height, width) for a source image, or zeros."""
        if not (read_images and bank_dir is not None and image_ref):
            return 0, 0, 0
        p = Path(bank_dir) / image_ref
        if not p.is_file():
            return 0, 0, 0
        try:
            from .sw import expected_rows

            # Header first. All we need is the pixel size, and decoding a
            # 24 MP TIFF to learn it turns a whole-tree migration from
            # seconds into an hour. cv2 only decodes fully, so it is the
            # fallback for whatever the header reader cannot open.
            h = w = 0
            try:
                from PIL import Image

                with Image.open(p) as im:
                    w, h = int(im.width), int(im.height)
            except Exception:
                h = w = 0
            if not (h and w):
                import cv2

                img = cv2.imread(str(p))
                if img is None:
                    return 0, 0, 0
                h, w = int(img.shape[0]), int(img.shape[1])
            return (
                int(expected_rows(h, w, window_size=int(bank.meta.window),
                                  stride=int(bank.meta.stride), patch=int(bank.meta.patch))),
                h, w,
            )
        except Exception:  # unreadable file, odd geometry
            return 0, 0, 0

    def _carve(
        tier: Tier,
        label: str,
        arr: np.ndarray,
        entries: list[dict],
        meta_arr: IncidentMetaArray | None,
    ) -> None:
        covered = np.zeros(int(arr.shape[0]), dtype=bool) if arr.size else np.zeros(0, dtype=bool)
        for e in entries:
            name = str(e.get("name", ""))
            start, count = int(e.get("start", -1)), int(e.get("count", 0))
            if start < 0 or count <= 0 or start + count > covered.shape[0]:
                logger.warning(
                    "migrate: %s/%s image %r has an out-of-range row index "
                    "(start=%d count=%d) — skipped", tier, label, name, start, count,
                )
                continue
            covered[start : start + count] = True
            kept = e.get("kept")
            image_ref = f"{SOURCE_IMAGES_SUBDIR}/{tier}/{name}" if name else ""
            grid_rows, h, w = _dims(image_ref)
            if not grid_rows:
                # Best effort when the source image is gone: an uncapped image
                # has one row per patch, and a capped one at least reaches its
                # highest kept patch.
                grid_rows = (max(int(v) for v in kept) + 1) if kept else count
            se = store.add(
                arr[start : start + count],
                name=name,
                grid_rows=grid_rows,
                kept=kept,
                image_ref=image_ref,
                height=h,
                width=w,
                ingested_at=int(bank.meta.inspection_count),
            )
            if tier == "normal":
                ls.assignments[se.id] = Assignment(tier="normal")
            else:
                base, heavy_rows = _entry_severity(meta_arr, start, count)
                marks = (
                    [int(kept[r]) for r in heavy_rows] if kept else list(heavy_rows)
                )
                ls.assignments[se.id] = Assignment(
                    tier=tier,
                    label=label,
                    severity=base,
                    marks=sorted(set(marks)),
                    rects=list(e.get("annotations") or []),
                )
        # Anything the index never claimed.
        if arr.size and not covered.all():
            leftover = np.flatnonzero(~covered)
            logger.warning(
                "migrate: %s/%s has %d rows with no image index — kept as one "
                "unindexed entry", tier, label, int(leftover.size),
            )
            se = store.add(
                arr[leftover],
                name=f"__unindexed__{tier}{('_' + label) if label else ''}",
                grid_rows=int(leftover.size),
                kept=None,
                image_ref="",
                ingested_at=int(bank.meta.inspection_count),
            )
            base = DEFAULT_SEVERITY
            ls.assignments[se.id] = Assignment(tier=tier, label=label, severity=base)

    _carve("normal", "", bank.normal, list(bank.meta.normal_image_index), None)
    for label in sorted(bank.critical):
        _carve(
            "critical", label, bank.critical[label],
            list(bank.meta.critical_image_index.get(label, [])),
            bank.critical_meta.get(label),
        )
    for label in sorted(bank.negative):
        _carve(
            "negative", label, bank.negative[label],
            list(bank.meta.negative_image_index.get(label, [])),
            bank.negative_meta.get(label),
        )
    store.save_index()
    return store, ls


def roundtrip_diff(original: Bank, rebuilt: Bank) -> list[str]:
    """Human-readable differences between two banks' scoring-facing contents.

    Used by the migration to refuse the new layout unless it reproduces the
    old bank exactly. Compares the feature arrays and the per-row severity —
    everything the verdict actually depends on. Provenance fields (filename
    logs, descriptions) are deliberately not compared: the migration is
    allowed to normalise those.
    """
    out: list[str] = []
    if original.normal.shape != rebuilt.normal.shape:
        out.append(f"normal shape {original.normal.shape} != {rebuilt.normal.shape}")
    elif original.normal.size and not np.array_equal(original.normal, rebuilt.normal):
        n = int((original.normal != rebuilt.normal).any(axis=1).sum())
        out.append(f"normal rows differ ({n} of {original.normal.shape[0]})")
    for tier in ("critical", "negative"):
        a = original.critical if tier == "critical" else original.negative
        b = rebuilt.critical if tier == "critical" else rebuilt.negative
        am = original.critical_meta if tier == "critical" else original.negative_meta
        bm = rebuilt.critical_meta if tier == "critical" else rebuilt.negative_meta
        if set(a) != set(b):
            out.append(f"{tier} labels {sorted(set(a))} != {sorted(set(b))}")
            continue
        for lab in sorted(a):
            if a[lab].shape != b[lab].shape:
                out.append(f"{tier}/{lab} shape {a[lab].shape} != {b[lab].shape}")
            elif not np.array_equal(a[lab], b[lab]):
                n = int((a[lab] != b[lab]).any(axis=1).sum())
                out.append(f"{tier}/{lab} rows differ ({n} of {a[lab].shape[0]})")
            sa, sb = am.get(lab), bm.get(lab)
            if sa is not None and sb is not None and len(sa) == len(sb):
                if not np.array_equal(sa.severity, sb.severity):
                    n = int((sa.severity != sb.severity).sum())
                    out.append(f"{tier}/{lab} severity differs ({n} rows)")
    return out


# ---- assembly state --------------------------------------------------------
# Which label set a bank on disk was last built from. Lives here rather than in
# the API layer because both writers need it: the HTTP route that assembles,
# and the offline migration script. When only the route wrote it, every bank
# migrated from the command line came up flagged "labels changed since the last
# assemble" — a false alarm inviting a rebuild that would reproduce the bank
# it already had.

ASSEMBLY_STATE_FILE = "assembly_state.json"


def assembly_fingerprint(store: FeatureStore, labelset: LabelSet) -> str:
    """Identity of the bank ``store`` + ``labelset`` would assemble into.

    Covers what actually changes the arrays: which images exist and how many
    rows each contributes, plus every assignment. Deliberately excludes
    timestamps and display names, so renaming a label set does not make a
    perfectly current bank look stale.
    """
    h = hashlib.sha256()
    for e in store.entries:
        h.update(f"{e.id}:{e.rows}\n".encode())
    h.update(b"--\n")
    for eid, a in sorted(labelset.assignments.items()):
        marks = ",".join(str(m) for m in a.marks)
        h.update(
            f"{eid}:{a.tier}:{a.resolved_label()}:{a.clamped_severity()}:{marks}\n".encode()
        )
    return h.hexdigest()[:32]


def read_assembly_state(bank_dir: Path) -> dict:
    """Last-assembled marker, or ``{}`` when absent or unreadable."""
    path = Path(bank_dir) / ASSEMBLY_STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_assembly_state(bank_dir: Path, labelset: LabelSet, fingerprint: str) -> None:
    # atomic_write_text writes through a sibling .tmp and creates no
    # directories; the bank directory exists on every path that reaches here
    # today, which is exactly the kind of assumption that stops being true.
    Path(bank_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        Path(bank_dir) / ASSEMBLY_STATE_FILE,
        json.dumps({"labelset_id": labelset.id, "fingerprint": fingerprint}, indent=2),
    )
