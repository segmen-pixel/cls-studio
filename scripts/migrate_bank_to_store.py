# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Carve existing memory banks into a feature store plus one label set.

The bank already records which rows came from which image
(``BankMeta.*_image_index``), so the migration is a copy, not a re-extraction:
no image is decoded and the backbone is never loaded. What it buys is the
ability to re-label without re-teaching -- see :mod:`clscore.store`.

Three modes, in the order you should use them:

    plan       read-only. Reports what each bank would become and, crucially,
               whether its row index covers every row it holds. Touches only
               bank_meta.json and the .npy headers, so it is fast on the whole
               projects tree.

    rehearse   carve a few images per tier into a SCRATCH directory and prove
               the copies are bit-identical to the bank's rows. Never writes
               into the project tree.

    execute    the real thing: writes ``<bank>/store/`` and
               ``<bank>/labelsets/<id>.json`` and then verifies that
               re-assembling from the label set ALONE reproduces the bank.

``execute`` is additive -- it creates two new subdirectories and does not
modify, move or delete anything the bank already has, so the original stays
authoritative until you decide otherwise. It still refuses to start without
free disk for a second copy of the features.

Examples::

    python scripts/migrate_bank_to_store.py plan --projects-dir "%CLS_DATA%/projects"
    python scripts/migrate_bank_to_store.py rehearse --bank-dir <bank> --scratch C:/scratch/mig --sample 3
    python scripts/migrate_bank_to_store.py execute --bank-dir <bank>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the sibling packages are importable when run straight from a checkout.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "packages" / "clscore",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from clscore.assemble import (  # noqa: E402
    assemble_bank,
    assembly_fingerprint,
    migrate_bank_to_store,
    roundtrip_diff,
    write_assembly_state,
)
from clscore.bank import Bank  # noqa: E402
from clscore.labelset import (  # noqa: E402
    DEFAULT_LABELSET_ID,
    DEFAULT_LABELSET_NAME,
    LABELSETS_SUBDIR,
    LabelSet,
    write_active_id,
)
from clscore.store import STORE_SUBDIR, FeatureStore  # noqa: E402


def _npy_shape(path: Path) -> tuple[int, ...]:
    """Shape of a .npy without paging in its data.

    ``mmap_mode`` reads the header and maps the rest; on a 5 GB bank that is
    the difference between a report you can run over the whole projects tree
    and one that needs the machine's RAM.
    """
    return tuple(int(v) for v in np.load(path, mmap_mode="r").shape)


def _fmt(n: int) -> str:
    return f"{n:,}"


# ---- plan -----------------------------------------------------------------


def plan_bank(bank_dir: Path) -> dict:
    """Read-only summary of one bank: sizes, images, and index coverage.

    Index coverage is the number that decides whether the migration is
    lossless for this bank. Rows the index never claims are still real taught
    data; they survive as one synthetic entry, but they cannot be re-labelled
    per image, so it is worth knowing up front how many there are.
    """
    meta_path = bank_dir / Bank.META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    out: dict = {"bank_dir": str(bank_dir), "tiers": {}, "uncovered": 0, "rows": 0, "images": 0}

    def _tier(tier: str, label: str, npy: Path, index: list[dict]) -> None:
        if not npy.exists():
            return
        rows = _npy_shape(npy)[0]
        covered = sum(int(e.get("count", 0)) for e in index)
        annotated = sum(1 for e in index if e.get("annotations"))
        capped = sum(1 for e in index if e.get("kept"))
        out["tiers"][f"{tier}/{label}" if label else tier] = {
            "rows": rows,
            "images": len(index),
            "covered_rows": covered,
            "uncovered_rows": max(0, rows - covered),
            "annotated_images": annotated,
            "capped_images": capped,
        }
        out["rows"] += rows
        out["images"] += len(index)
        out["uncovered"] += max(0, rows - covered)

    _tier("normal", "", bank_dir / Bank.NORMAL_FILE, meta.get("normal_image_index") or [])
    for sub, key in (("critical", "critical_image_index"), ("negative", "negative_image_index")):
        d = bank_dir / sub
        if not d.is_dir():
            continue
        idx = meta.get(key) or {}
        for npy in sorted(d.glob("*.npy")):
            _tier(sub, npy.stem, npy, idx.get(npy.stem) or [])
    out["dim"] = int(meta.get("dim", 0))
    out["model"] = str(meta.get("model", ""))
    out["has_store"] = (bank_dir / STORE_SUBDIR / "store_index.json").exists()
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    banks = _resolve_banks(args)
    if not banks:
        print("no banks found")
        return 1
    grand_rows = grand_uncovered = grand_images = 0
    for b in banks:
        p = plan_bank(b)
        flag = "  [store already present]" if p["has_store"] else ""
        print(f"\n== {b}{flag}")
        print(f"   model={p['model']} dim={p['dim']}")
        for name, t in p["tiers"].items():
            note = ""
            if t["uncovered_rows"]:
                note = f"  <-- {_fmt(t['uncovered_rows'])} rows NOT indexed"
            print(
                f"   {name:<24} rows={_fmt(t['rows']):>12}  images={t['images']:>5}"
                f"  annotated={t['annotated_images']:>4}  capped={t['capped_images']:>4}{note}"
            )
        print(f"   TOTAL rows={_fmt(p['rows'])} images={p['images']} uncovered={_fmt(p['uncovered'])}")
        grand_rows += p["rows"]
        grand_images += p["images"]
        grand_uncovered += p["uncovered"]
    print(
        f"\n{len(banks)} bank(s): rows={_fmt(grand_rows)} images={grand_images} "
        f"uncovered={_fmt(grand_uncovered)}"
    )
    if grand_uncovered:
        print(
            "note: unindexed rows are preserved as one synthetic entry per tier, "
            "but cannot be re-labelled per image."
        )
    return 0


# ---- rehearse / execute ---------------------------------------------------


def _free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _carve(
    bank: Bank,
    bank_dir: Path,
    store_dir: Path,
    *,
    labelset_id: str,
    labelset_name: str,
    read_images: bool,
) -> tuple[FeatureStore, LabelSet, float]:
    t0 = time.perf_counter()
    store = FeatureStore(store_dir)
    store, ls = migrate_bank_to_store(
        bank,
        store,
        bank_dir=bank_dir,
        labelset_id=labelset_id,
        labelset_name=labelset_name,
        read_images=read_images,
    )
    return store, ls, time.perf_counter() - t0


def _verify_slices(bank: Bank, store: FeatureStore, ls: LabelSet) -> list[str]:
    """Compare every stored array against the bank rows it was carved from.

    Weaker than the full round-trip (it does not exercise assembly order or
    severity) but it holds one image in memory at a time, so it is the check
    that works on a bank too large to assemble twice.
    """
    problems: list[str] = []
    index = {
        ("normal", ""): (bank.normal, bank.meta.normal_image_index),
    }
    for lab, arr in bank.critical.items():
        index[("critical", lab)] = (arr, bank.meta.critical_image_index.get(lab, []))
    for lab, arr in bank.negative.items():
        index[("negative", lab)] = (arr, bank.meta.negative_image_index.get(lab, []))
    for entry in store.entries:
        a = ls.assignments.get(entry.id)
        if a is None or entry.name.startswith("__unindexed__"):
            continue
        key = (a.tier, a.resolved_label())
        arr, entries = index.get(key, (None, []))
        src = next((e for e in entries if str(e.get("name", "")) == entry.name), None)
        if arr is None or src is None:
            problems.append(f"{entry.id} {entry.name}: no source rows under {key}")
            continue
        s, c = int(src["start"]), int(src["count"])
        if not np.array_equal(store.features_of(entry), arr[s : s + c]):
            problems.append(f"{entry.id} {entry.name}: stored rows differ from the bank")
    return problems


def cmd_rehearse(args: argparse.Namespace) -> int:
    bank_dir = Path(args.bank_dir)
    scratch = Path(args.scratch)
    if scratch.exists() and any(scratch.iterdir()):
        print(f"scratch dir is not empty: {scratch}")
        return 1
    print(f"loading bank: {bank_dir}")
    bank = Bank.load(bank_dir)
    print(f"  {bank!r}")
    sampled = _sample_bank(bank, args.sample)
    print(f"rehearsing with {args.sample} image(s) per tier")
    store, ls, secs = _carve(
        sampled, bank_dir, scratch / STORE_SUBDIR,
        labelset_id=args.labelset_id, labelset_name=args.labelset_name,
        read_images=args.read_images,
    )
    ls.save(scratch / LABELSETS_SUBDIR)
    print(f"  carved {len(store)} image(s), {_fmt(store.total_rows())} rows in {secs:.1f}s")

    problems = _verify_slices(sampled, store, ls)
    rebuilt = assemble_bank(store, ls, prev_bank=None)
    problems += roundtrip_diff(sampled, rebuilt)
    if problems:
        print("REHEARSAL FAILED:")
        for p in problems[:20]:
            print(f"  - {p}")
        return 2
    print("rehearsal OK: carved rows are bit-identical and re-assemble exactly")
    if args.keep:
        print(f"  artifacts left in {scratch}")
    else:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def _merge_meta(parts: list):
    from clscore.incident import IncidentMetaArray

    return IncidentMetaArray(
        severity=np.concatenate([m.severity for m in parts]),
        registered_at_inspection=np.concatenate([m.registered_at_inspection for m in parts]),
        last_hit_at_inspection=np.concatenate([m.last_hit_at_inspection for m in parts]),
        hit_count=np.concatenate([m.hit_count for m in parts]),
        tier=np.concatenate([m.tier for m in parts]),
    )


def _slice_tier(arr, entries: list[dict], meta_arr, n: int):
    """First ``n`` indexed images of one tier, re-based to start at row 0."""
    parts, new_entries, names, metas = [], [], [], []
    cursor = 0
    for e in entries[:n]:
        s, c = int(e.get("start", -1)), int(e.get("count", 0))
        if s < 0 or c <= 0 or s + c > int(arr.shape[0]):
            continue
        parts.append(arr[s : s + c])
        ne: dict = {"name": e["name"], "start": cursor, "count": c}
        if e.get("kept"):
            ne["kept"] = list(e["kept"])
        if e.get("annotations"):
            ne["annotations"] = list(e["annotations"])
        new_entries.append(ne)
        names.append(e["name"])
        if meta_arr is not None and s + c <= len(meta_arr):
            metas.append(meta_arr.take(np.arange(s, s + c, dtype=np.int64)))
        cursor += c
    joined = np.concatenate(parts, axis=0) if parts else None
    return joined, new_entries, names, (_merge_meta(metas) if metas else None)


def _sample_bank(bank: Bank, n: int) -> Bank:
    """A bank holding only the first ``n`` indexed images of each tier/label.

    Built by slicing rows out through the same index the migration uses, so a
    rehearsal exercises the real geometry -- coreset-capped images, their kept
    maps and their marks -- rather than a synthetic stand-in.
    """
    from clscore.bank import BankMeta

    dim = int(bank.meta.dim) or (int(bank.normal.shape[1]) if bank.normal.size else 1)
    out = Bank(
        normal=np.zeros((0, dim), dtype=bank.normal.dtype),
        meta=BankMeta(
            model=bank.meta.model, dim=dim, layers=bank.meta.layers,
            window=bank.meta.window, stride=bank.meta.stride, patch=bank.meta.patch,
            inspection_count=bank.meta.inspection_count,
        ),
    )
    joined, entries, names, _ = _slice_tier(
        bank.normal, bank.meta.normal_image_index, None, n
    )
    if joined is not None:
        out.normal = joined
        out.meta.normal_image_index = entries
        out.meta.bank_images = names
    for tier in ("critical", "negative"):
        src = bank.critical if tier == "critical" else bank.negative
        src_idx = (bank.meta.critical_image_index if tier == "critical"
                   else bank.meta.negative_image_index)
        src_meta = bank.critical_meta if tier == "critical" else bank.negative_meta
        tgt = out.critical if tier == "critical" else out.negative
        tgt_meta = out.critical_meta if tier == "critical" else out.negative_meta
        tgt_idx = (out.meta.critical_image_index if tier == "critical"
                   else out.meta.negative_image_index)
        tgt_log = out.meta.critical_images if tier == "critical" else out.meta.negative_images
        for lab, arr in src.items():
            joined, entries, names, merged = _slice_tier(
                arr, src_idx.get(lab, []), src_meta.get(lab), n
            )
            if joined is None:
                continue
            tgt[lab] = joined
            tgt_idx[lab] = entries
            tgt_log[lab] = names
            if merged is not None:
                tgt_meta[lab] = merged
    return out


def cmd_execute(args: argparse.Namespace) -> int:
    bank_dir = Path(args.bank_dir)
    store_dir = Path(args.store_dir) if args.store_dir else bank_dir / STORE_SUBDIR
    ls_dir = Path(args.labelsets_dir) if args.labelsets_dir else bank_dir / LABELSETS_SUBDIR
    if store_dir.exists() and any(store_dir.iterdir()) and not args.force:
        print(f"store already exists: {store_dir} (use --force to overwrite)")
        return 1

    plan = plan_bank(bank_dir)
    need = plan["rows"] * max(1, plan["dim"]) * 2  # fp16
    free = _free_bytes(store_dir.parent if store_dir.parent.exists() else bank_dir)
    print(f"bank rows={_fmt(plan['rows'])} dim={plan['dim']}")
    print(f"store needs ~{need / 1024**3:.1f} GB, free {free / 1024**3:.1f} GB")
    if free < need * 1.1:
        print("not enough free disk for a second copy of the features -- refusing")
        return 1

    print(f"loading bank: {bank_dir}")
    bank = Bank.load(bank_dir)
    print(f"  {bank!r}")
    if args.force and store_dir.exists():
        shutil.rmtree(store_dir, ignore_errors=True)
    store, ls, secs = _carve(
        bank, bank_dir, store_dir,
        labelset_id=args.labelset_id, labelset_name=args.labelset_name,
        read_images=args.read_images,
    )
    ls.save(ls_dir)
    write_active_id(ls_dir, ls.id)
    print(f"  carved {len(store)} image(s), {_fmt(store.total_rows())} rows in {secs:.1f}s")
    print(f"  store    -> {store_dir}")
    print(f"  labelset -> {ls_dir / (ls.id + '.json')}")

    if args.verify == "none":
        print("verification skipped (--verify none)")
        return 0
    t0 = time.perf_counter()
    if args.verify == "slices":
        problems = _verify_slices(bank, store, ls)
    else:
        # prev_bank stays None on purpose: the migration is only sound if the
        # LABEL SET alone rebuilds the bank. Passing the original would donate
        # its metadata and mask anything the label set failed to capture.
        problems = roundtrip_diff(bank, assemble_bank(store, ls, prev_bank=None))
    print(f"verify ({args.verify}) took {time.perf_counter() - t0:.1f}s")
    if problems:
        print("VERIFICATION FAILED -- the bank is untouched, the store is suspect:")
        for p in problems[:20]:
            print(f"  - {p}")
        return 2
    # Stamp what the bank on disk was built from. Without it the UI reports
    # every freshly migrated bank as "labels changed since the last
    # assemble" and invites a rebuild that reproduces what is already there.
    write_assembly_state(bank_dir, ls, assembly_fingerprint(store, ls))
    print("verified: re-assembling from the label set reproduces the bank exactly")
    return 0


# ---- cli ------------------------------------------------------------------


def _resolve_banks(args: argparse.Namespace) -> list[Path]:
    if getattr(args, "bank_dir", None):
        return [Path(args.bank_dir)]
    root = Path(args.projects_dir)
    return sorted(
        d for d in root.glob("*/banks/*")
        if d.is_dir() and (d / Bank.META_FILE).exists() and not (d / ".deleted").exists()
    )


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp932 here and raise UnicodeEncodeError on
    # anything outside it -- including bank descriptions and image filenames.
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--labelset-id", default=DEFAULT_LABELSET_ID)
    common.add_argument("--labelset-name", default=DEFAULT_LABELSET_NAME)
    common.add_argument(
        "--no-read-images", dest="read_images", action="store_false",
        help="skip decoding source images (grid geometry becomes a lower bound)",
    )
    common.set_defaults(read_images=True)

    p = sub.add_parser("plan", help="read-only report; writes nothing")
    p.add_argument("--projects-dir")
    p.add_argument("--bank-dir")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("rehearse", parents=[common], help="carve a few images into scratch and verify")
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--sample", type=int, default=3)
    p.add_argument("--keep", action="store_true", help="leave the scratch artifacts behind")
    p.set_defaults(func=cmd_rehearse)

    p = sub.add_parser("execute", parents=[common], help="write the store + label set for real")
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--store-dir")
    p.add_argument("--labelsets-dir")
    p.add_argument("--force", action="store_true", help="replace an existing store")
    p.add_argument("--verify", choices=("full", "slices", "none"), default="full")
    p.set_defaults(func=cmd_execute)

    args = ap.parse_args(argv)
    if args.cmd == "plan" and not (args.projects_dir or args.bank_dir):
        ap.error("plan needs --projects-dir or --bank-dir")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
