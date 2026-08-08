# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Feature store + label set access for the active bank, and their assembly.

The bank directory gains two subdirectories, both additive — nothing that was
already there moves or changes shape:

    <bank>/store/            immutable per-image features (clscore.store)
    <bank>/labelsets/        the operator's assignments (clscore.labelset)
    <bank>/assembly_state.json   which label set the bank was last built from

Assembly is explicit rather than automatic. Re-labelling is meant to be free
and repeated dozens of times in a sitting, while rebuilding a multi-GB bank
costs seconds to minutes and invalidates the GPU-resident tensors; doing it on
every click would put the expensive operation right back on top of the cheap
one, which is the exact coupling the store exists to break. The UI instead
reads :func:`assembly_status` and shows "assemble to apply".
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from clscore import assemble as _assemble
from clscore.assemble import (
    assemble_bank,
    assembly_fingerprint,
    migrate_bank_to_store,
    roundtrip_diff,
)
from clscore.bank import Bank
from clscore.fsio import replace_with_retry
from clscore.labelset import (
    DEFAULT_LABELSET_ID,
    DEFAULT_LABELSET_NAME,
    LABELSETS_SUBDIR,
    LabelSet,
    list_labelsets,
    read_active_id,
    slug_labelset_id,
    write_active_id,
)
from clscore.store import STORE_SUBDIR, FeatureStore, StoreEntry, StoreMeta

from .cls_eval_cache import eval_cache_reconcile
from .cls_state import ClsStudioState
from .exceptions import ValidationError

logger = logging.getLogger(__name__)

# NOTE: the name of the assembly-state file is NOT restated here. It used to be,
# as a verbatim copy of clscore.assemble.ASSEMBLY_STATE_FILE with zero readers --
# the wrappers below delegate to clscore, which builds the path from its own
# constant. Nothing pinned the two together (unlike RENDER_SUBDIR, pinned by
# apps/api/tests/test_bank_layout_constants.py), so the copy was free to drift
# into a second, silently wrong answer. clscore owns the name.

# Formats browsers render natively are kept as uploaded; everything else (TIFF
# above all) is transcoded so the labelling grid can actually show the image.
# Same rule as ClsStudioState.save_source_image, applied to the store's own
# image directory.
_WEB_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})


# ---- locations -------------------------------------------------------------


def store_dir(state: ClsStudioState) -> Path:
    state.require_active()
    assert state.bank_dir is not None
    return state.bank_dir / STORE_SUBDIR


def labelsets_dir(state: ClsStudioState) -> Path:
    state.require_active()
    assert state.bank_dir is not None
    return state.bank_dir / LABELSETS_SUBDIR


def load_store(state: ClsStudioState) -> FeatureStore:
    """The active bank's store. Absent on disk yields an empty one."""
    st = FeatureStore.load(store_dir(state))
    if not st.entries and int(st.meta.dim) == 0:
        st.meta = StoreMeta(model=state.model_name)
    return st


# ---- label sets ------------------------------------------------------------


def all_labelsets(state: ClsStudioState) -> list[LabelSet]:
    return list_labelsets(labelsets_dir(state))


def active_labelset(state: ClsStudioState, create: bool = True) -> LabelSet:
    """The label set the UI is editing, creating an empty default if needed."""
    d = labelsets_dir(state)
    wanted = read_active_id(d)
    existing = list_labelsets(d)
    if wanted:
        for ls in existing:
            if ls.id == wanted:
                return ls
    if existing:
        write_active_id(d, existing[0].id)
        return existing[0]
    ls = LabelSet(id=DEFAULT_LABELSET_ID, name=DEFAULT_LABELSET_NAME)
    if create:
        ls.save(d)
        write_active_id(d, ls.id)
    return ls


def load_labelset(state: ClsStudioState, labelset_id: str) -> LabelSet:
    path = labelsets_dir(state) / f"{slug_labelset_id(labelset_id)}.json"
    if not path.is_file():
        raise ValidationError(f"no such label set: {labelset_id}")
    return LabelSet.load(path)


def create_labelset(state: ClsStudioState, name: str, copy_from: LabelSet | None = None) -> LabelSet:
    """New label set, optionally seeded from another one.

    Copying is the cheap way to ask "what if these borderline images counted
    as NG?" — fork the current judgement, change a handful of assignments,
    assemble, and compare. Nothing is re-extracted either way.
    """
    d = labelsets_dir(state)
    base = slug_labelset_id(name)
    taken = {ls.id for ls in list_labelsets(d)}
    candidate, i = base, 1
    while candidate in taken:
        i += 1
        candidate = f"{base}-{i}"
    ls = LabelSet(id=candidate, name=name.strip() or candidate)
    if copy_from is not None:
        ls.assignments = {k: v for k, v in copy_from.assignments.items()}
        ls.description = f"copied from {copy_from.name}"
    ls.save(d)
    return ls


def delete_labelset(state: ClsStudioState, labelset_id: str) -> None:
    d = labelsets_dir(state)
    remaining = [ls for ls in list_labelsets(d) if ls.id != labelset_id]
    if len(remaining) == len(list_labelsets(d)):
        raise ValidationError(f"no such label set: {labelset_id}")
    (d / f"{labelset_id}.json").unlink(missing_ok=True)
    # Never leave the active marker pointing at a deleted set: every later
    # read would silently fall through to "first existing", which is a
    # different judgement than the one the operator was editing.
    if read_active_id(d) == labelset_id:
        write_active_id(d, remaining[0].id if remaining else DEFAULT_LABELSET_ID)


# ---- assembly state --------------------------------------------------------


def read_assembly_state(state: ClsStudioState) -> dict:
    state.require_active()
    assert state.bank_dir is not None
    return _assemble.read_assembly_state(state.bank_dir)


def write_assembly_state(state: ClsStudioState, labelset: LabelSet, fingerprint: str) -> None:
    state.require_active()
    assert state.bank_dir is not None
    _assemble.write_assembly_state(state.bank_dir, labelset, fingerprint)


def assembly_status(state: ClsStudioState) -> dict:
    """What the UI needs to decide whether to offer "assemble"."""
    store = load_store(state)
    ls = active_labelset(state)
    fp = assembly_fingerprint(store, ls)
    prev = read_assembly_state(state)
    # Counted set-wise against the store's own ids, not by subtracting two
    # independently-maintained totals. `len(store) - sum(ls.counts())`
    # under-reports by exactly the number of assignments left dangling by a
    # delete, and the max(0, ...) that used to sit here hid the divergence
    # instead of surfacing it: after a multi-image delete the Bank tab read
    # "0 unassigned" and ticked step 2 green over images that had never been
    # labelled, and assembling then produced an empty bank. Stale assignments
    # simply do not count now, so an already-corrupted label set reads right
    # without needing a migration.
    live = {e.id for e in store.entries}
    counts = {"normal": 0, "critical": 0, "negative": 0}
    for entry_id, a in ls.assignments.items():
        if entry_id in live and a.tier in counts:
            counts[a.tier] += 1
    assigned = sum(counts.values())
    return {
        "labelset_id": ls.id,
        "labelset_name": ls.name,
        "store_images": len(store),
        "store_rows": store.total_rows(),
        "assigned": assigned,
        "unassigned": len(live) - assigned,
        "stale_assignments": len(ls.assignments) - assigned,
        "counts": counts,
        "fingerprint": fp,
        # A bank that was never assembled from a label set (every bank today)
        # is not "stale" until there is something to assemble: flagging an
        # untouched project would tell the operator to rebuild a bank the
        # store cannot yet reproduce.
        "stale": bool(store.entries) and prev.get("fingerprint") != fp,
        "assembled_from": prev.get("labelset_id", ""),
        "migrated": bool(store.entries),
    }


# ---- assembly --------------------------------------------------------------


def assemble_active_bank(
    state: ClsStudioState,
    *,
    normal_ceiling: int = 0,
    labelset: LabelSet | None = None,
) -> Bank:
    """Rebuild the active bank from the store and label set, then persist it.

    The in-memory swap happens under ``state.lock`` and the (potentially
    multi-GB) write outside it, matching every other bank mutation: scoring
    only needs the lock to read the current tensors, and blocking it for the
    length of a disk write is what the split lock exists to avoid.
    """
    # Identity snapshot, the same one both teach paths take. The assembly
    # below runs for minutes without the lock, and `state.bank_dir` is
    # process-global and re-resolved at save time — so a /bank/select from
    # another tab during the run made this write land in THAT project, where
    # Bank.save unlinks every tier file it did not produce. check_binding on
    # the route only proves the caller was right when the request arrived.
    bound_bank, bound_dir = state.bank, state.bank_dir
    store = load_store(state)
    ls = labelset if labelset is not None else active_labelset(state)
    if not store.entries:
        raise ValidationError(
            "this bank has no feature store yet — run the migration or ingest images first"
        )
    # ``state.device`` resolves the configured device WITHOUT loading DINOv2 —
    # assembly is numpy plus (at most) a k-Center on the GPU, and pulling a
    # backbone into VRAM to rebuild a bank from features that already exist
    # would reintroduce exactly the cost this split removes.
    bank = assemble_bank(
        store, ls, normal_ceiling=normal_ceiling, device=state.device, prev_bank=state.bank,
    )
    with state.lock:
        if state.bank is not bound_bank or state.bank_dir != bound_dir:
            raise ValidationError(
                "active bank changed during assembly — nothing was written, "
                "re-select the project and run it again"
            )
        state.bank = bank
        state.mark_dirty()
    state.save_bank()
    write_assembly_state(state, ls, assembly_fingerprint(store, ls))
    # An assemble is how an image actually leaves a bank in the shipped UI:
    # delete or re-label in the Bank tab, then assemble. Nothing reconciled
    # the eval cache on that path -- the two purge hooks that exist hang off
    # /bank/clear/{tier} and /bank/images/delete, which no component calls --
    # and the fingerprint could not stand in for it, being normal-tier-only.
    # So a deleted defect kept driving the AUROC and the auto-threshold.
    #
    # Best-effort: a bank that assembled correctly must not fail the request
    # because its cache could not be tidied.
    try:
        eval_cache_reconcile(state)
    except (OSError, ValueError):  # pragma: no cover - cache is disposable
        logger.warning("eval cache reconcile failed after assemble", exc_info=True)
    return bank


def migrate_active_bank(state: ClsStudioState, verify: bool = True) -> dict:
    """Carve the active bank into a store + "standard" label set.

    Additive and verified: the bank's own files are untouched, and unless the
    caller opts out we prove the new layout by re-assembling from the label
    set alone and comparing against what is currently loaded. A failure leaves
    the bank exactly as it was — the store is the only thing that is suspect,
    and it is reported rather than silently kept.
    """
    state.require_active()
    assert state.bank_dir is not None
    sd = store_dir(state)
    if sd.exists() and any(sd.iterdir()):
        raise ValidationError("this bank already has a feature store")
    store, ls = migrate_bank_to_store(
        state.bank, FeatureStore(sd), bank_dir=state.bank_dir, read_images=True,
    )
    problems: list[str] = []
    if verify:
        # prev_bank stays None: the migration is only sound if the LABEL SET
        # alone rebuilds the bank. Donating the original's metadata would mask
        # anything the carve failed to capture.
        problems = roundtrip_diff(state.bank, assemble_bank(store, ls, prev_bank=None))
    if problems:
        import shutil

        shutil.rmtree(sd, ignore_errors=True)
        return {"ok": False, "images": len(store), "rows": store.total_rows(), "problems": problems}
    ls.save(labelsets_dir(state))
    write_active_id(labelsets_dir(state), ls.id)
    write_assembly_state(state, ls, assembly_fingerprint(store, ls))
    return {
        "ok": True,
        "images": len(store),
        "rows": store.total_rows(),
        "problems": [],
        "labelset_id": ls.id,
    }


# ---- ingest ----------------------------------------------------------------


def _store_image_name(entry_id: str, filename: str) -> str:
    suffix = Path(re.sub(r"[^A-Za-z0-9._-]", "_", (filename or "image.png").rsplit("/", 1)[-1])).suffix
    if suffix.lower() not in _WEB_IMAGE_EXTS:
        suffix = ".png"
    return f"{entry_id}{suffix}"


def _write_store_image(store: FeatureStore, entry: StoreEntry, filename: str, data: bytes) -> str:
    """Persist the source image next to its features; returns the bank-relative ref."""
    # The extension _store_image_name picks and the transcode _maybe_png does
    # are driven by the same web-format test, so the two always agree.
    name = _store_image_name(entry.id, filename)
    d = store.images_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return f"{STORE_SUBDIR}/{store.images_dir().name}/{name}"


def _maybe_png(filename: str, data: bytes) -> bytes:
    """Transcode non-web formats so the labelling grid can display them."""
    if Path(filename or "").suffix.lower() in _WEB_IMAGE_EXTS:
        return data
    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return data
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return buf.tobytes() if ok else data
    except Exception:
        return data  # feature extraction already validated it; keep the bytes


# Images per batched forward group, mirroring the teach path: bounds transient
# window memory while still packing full max_batch forwards.
INGEST_GROUP = 24


# ---- display renditions ----------------------------------------------------
# The store keeps the ORIGINAL image, which is what a re-ingest needs and what
# the geometry was measured from. Serving those straight to the UI is what made
# the picker heavy: a 500-image project pulls hundreds of megabytes to fill a
# list of 40px rows, and a 24 MP source decodes for a second before the centre
# preview appears. Renditions are generated once and cached next to the store.

RENDER_SUBDIR = "cache"
# Long-edge pixels. ``thumb`` is for list rows, ``preview`` for the centre
# pane at any window size we can render on; anything larger is only useful to
# a re-ingest, which reads the original anyway.
RENDER_SIZES: dict[str, int] = {"thumb": 192, "preview": 1400}


def rendered_image(src: Path, cache_dir: Path, entry_id: str, size: str) -> Path:
    """Cached downscaled JPEG of ``src``, or ``src`` itself for unknown sizes.

    Falls back to the original on any failure rather than erroring: a picker
    that shows the full-size image is slow, and one that shows nothing is
    broken.
    """
    if size not in RENDER_SIZES:
        return src
    out = Path(cache_dir) / f"{entry_id}_{size}.jpg"
    try:
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
    except OSError:
        pass
    try:
        import cv2

        img = cv2.imread(str(src))
        if img is None:
            return src
        h, w = img.shape[:2]
        target = RENDER_SIZES[size]
        longest = max(h, w)
        if longest > target:
            s = target / float(longest)
            img = cv2.resize(
                img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return src
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(buf.tobytes())
        replace_with_retry(tmp, out)
        return out
    except Exception:  # unreadable, odd colour space, disk full
        return src


def drop_renditions(store: FeatureStore, entry_ids: list[str]) -> None:
    """Remove cached renditions for deleted entries."""
    d = store.directory / RENDER_SUBDIR
    if not d.is_dir():
        return
    for eid in entry_ids:
        for p in d.glob(f"{eid}_*.jpg"):
            p.unlink(missing_ok=True)


def ingest_decoded(
    state: ClsStudioState,
    store: FeatureStore,
    decoded: list[tuple[str, bytes, np.ndarray]],
    *,
    max_patches: int = 0,
) -> list[StoreEntry]:
    """Extract features for already-decoded images and add them to the store.

    Saves the manifest per group rather than once at the end: ingest is the
    expensive half, and a crash 400 images into a 500-image drop must not
    orphan the arrays already written (the same lesson batch teaching learned
    on 2026-07-17).
    """
    from clscore.feature_extractor import iter_images_features_batched
    from clscore.sw import expected_rows

    model, device, _dtype = state.ensure_model()
    geo = store.meta.geometry()
    out: list[StoreEntry] = []
    for i in range(0, len(decoded), INGEST_GROUP):
        grp = decoded[i : i + INGEST_GROUP]
        arrs = [a for (_n, _d, a) in grp]
        produced: list[tuple[np.ndarray, np.ndarray | None]] = []
        for _idx, feats in iter_images_features_batched(
            model, arrs, device, max_batch=state.max_batch()
        ):
            kept: np.ndarray | None = None
            if max_patches > 0 and feats.shape[0] > max_patches:
                from clscore.bank import coreset_reduce_indexed

                feats, kept = coreset_reduce_indexed(
                    feats, max_patches / float(feats.shape[0]), device
                )
            produced.append((feats, kept))
        for (fname, data, arr), (feats, kept) in zip(grp, produced):
            h, w = int(arr.shape[0]), int(arr.shape[1])
            if store.meta.dim == 0 and feats.size:
                store.meta.dim = int(feats.shape[1])
            entry = store.add(
                feats,
                name=(fname or "image.png").rsplit("/", 1)[-1],
                grid_rows=int(expected_rows(h, w, **geo)),
                kept=kept,
                height=h,
                width=w,
                ingested_at=int(state.bank.meta.inspection_count),
            )
            entry.image_ref = _write_store_image(
                store, entry, fname, _maybe_png(fname, data)
            )
            out.append(entry)
        # Under the lock, like every other writer of this manifest. Without it
        # a concurrent /store/delete held its own snapshot and the two wrote
        # the whole index back last-write-wins, leaving entries that name
        # features nobody has and a next_id that hands out ids already used.
        with state.lock:
            store.save_index()
    return out
