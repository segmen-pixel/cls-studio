#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Shrink an existing bank's over-sized images down to the per-image cap.

The per-image patch cap bounds what a *new* teach adds. It does nothing for
rows already on disk, and the labelled tiers were exempt from it for a long
time, so banks in the field can be far larger than the budget suggests — the
ones this was written for reach 12.8 M labelled rows, about 20 GB resident the
moment the bank is activated.

What it does, per image, for any image holding more rows than the cap:

  * keeps every ANNOTATED row. Those are the defect exemplars the alpha term
    scores against; dropping them would quietly change detection, which is the
    one thing a maintenance tool must not do.
  * coreset-reduces the remaining rows to fill what is left of the budget,
    the same k-center selection a teach would have applied.
  * records which patches of the image's grid survived, so the NG marks and
    any later annotation still land on the right rows.

Everything else is preserved: per-row severity / freshness / hit counts travel
with their rows, the per-image row ranges are rebuilt exactly, and images at or
under the cap are not touched at all.

Dry run by default — nothing is written until you pass --apply, and --apply
copies the bank directory first unless you opt out.

Usage:
  python scripts/reduce_bank.py --projects-dir DIR                 # survey
  python scripts/reduce_bank.py --projects-dir DIR --project ID --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Windows consoles default to cp932, which cannot encode the report glyphs —
# the run then dies on its own summary line after doing all the work.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "clscore"))

from clscore.bank import BANK_DTYPE, Bank, coreset_reduce_indexed  # noqa: E402
from clscore.incident import DEFAULT_SEVERITY  # noqa: E402

DEFAULT_CAP = 2048
BYTES_PER_ROW = 768 * 2  # fp16 x DINOv2 dim, for the size estimates


def _plan_image(
    feats: np.ndarray,
    sev: np.ndarray,
    cap: int,
    device: str,
) -> np.ndarray | None:
    """Local row indices to keep for one image, or None to leave it alone."""
    n = int(feats.shape[0])
    if n <= cap:
        return None
    marked = np.flatnonzero(sev != DEFAULT_SEVERITY)
    budget = cap - int(marked.size)
    if budget <= 0:
        # More annotated rows than the cap: keep exactly those and nothing
        # else. Never drop an exemplar to satisfy a budget.
        return marked.astype(np.int64)
    rest = np.setdiff1d(np.arange(n, dtype=np.int64), marked, assume_unique=False)
    if rest.size <= budget:
        return None
    _sub, pick = coreset_reduce_indexed(
        feats[rest].astype(np.float32), budget / float(rest.size), device
    )
    # Marks first: a later truncation to fit a ceiling keeps the exemplars.
    return np.concatenate([marked.astype(np.int64), rest[pick]])


def _reduce_label(
    arr: np.ndarray,
    meta,  # IncidentMetaArray
    entries: list[dict],
    cap: int,
    device: str,
    apply: bool,
) -> tuple[np.ndarray, object, list[dict], int]:
    """Rebuild one label's array/meta/index. Returns (arr, meta, entries, saved)."""
    ordered = sorted(entries, key=lambda e: int(e.get("start", 0)))
    # Refuse to guess: the entries must tile the array exactly, or rows exist
    # that no image claims and any rebuild would silently move them.
    pos = 0
    for e in ordered:
        if int(e.get("start", -1)) != pos:
            raise ValueError("row index does not tile the array — refusing to rewrite")
        pos += int(e.get("count", 0))
    if pos != int(arr.shape[0]):
        raise ValueError(
            f"row index covers {pos} of {arr.shape[0]} rows — refusing to rewrite"
        )

    blocks: list[np.ndarray] = []
    keeps: list[np.ndarray] = []
    new_entries: list[dict] = []
    saved = 0
    start = 0
    for e in ordered:
        s, c = int(e["start"]), int(e["count"])
        feats = arr[s : s + c]
        sev = meta.severity[s : s + c]
        local = _plan_image(feats, sev, cap, device)
        new = dict(e)
        if local is None:
            blocks.append(feats)
            keeps.append(np.arange(s, s + c, dtype=np.int64))
            new["start"], new["count"] = start, c
            start += c
        else:
            blocks.append(feats[local])
            keeps.append(local.astype(np.int64) + s)
            prev = e.get("kept")
            grid = (
                np.asarray(prev, dtype=np.int64)[local] if prev
                else local.astype(np.int64)
            )
            new["start"], new["count"] = start, int(local.size)
            new["kept"] = [int(v) for v in grid]
            saved += c - int(local.size)
            start += int(local.size)
        new_entries.append(new)

    if not apply or saved == 0:
        return arr, meta, entries, saved
    new_arr = np.concatenate(blocks, axis=0).astype(BANK_DTYPE, copy=False)
    # An index array, NOT a boolean mask: within a reduced image the kept rows
    # are emitted marks-first, so they are not in ascending order and a mask
    # would re-sort the metadata away from its features.
    take = np.concatenate(keeps)
    new_meta = meta.take(take)
    if int(new_arr.shape[0]) != len(new_meta):
        raise ValueError("rebuilt array and metadata disagree — refusing to write")
    return new_arr, new_meta, new_entries, saved


def process_bank(
    bank_dir: Path, cap: int, device: str, apply: bool, backup_root: Path | None
) -> dict:
    bank = Bank.load(bank_dir)
    report = {"labels": [], "rows_saved": 0, "errors": []}
    todo = []
    for tier in ("critical", "negative"):
        feats_d = bank.critical if tier == "critical" else bank.negative
        metas_d = bank.critical_meta if tier == "critical" else bank.negative_meta
        index = (
            bank.meta.critical_image_index if tier == "critical"
            else bank.meta.negative_image_index
        )
        for lab, arr in list(feats_d.items()):
            meta = metas_d.get(lab)
            entries = index.get(lab, [])
            if meta is None or not entries or arr.size == 0:
                continue
            if len(meta) != int(arr.shape[0]):
                report["errors"].append(f"{tier}/{lab}: meta length != rows — skipped")
                continue
            try:
                new_arr, new_meta, new_entries, saved = _reduce_label(
                    arr, meta, entries, cap, device, apply
                )
            except ValueError as exc:
                report["errors"].append(f"{tier}/{lab}: {exc}")
                continue
            if saved:
                report["labels"].append(
                    {"tier": tier, "label": lab, "before": int(arr.shape[0]),
                     "after": int(arr.shape[0]) - saved, "saved": saved}
                )
                report["rows_saved"] += saved
                if apply:
                    todo.append((tier, lab, new_arr, new_meta, new_entries, index, feats_d, metas_d))

    if apply and todo:
        if backup_root is not None:
            # OUTSIDE the projects tree, never inside banks/: the app lists
            # every subdirectory of banks/ as a bank (list_banks only skips
            # .deleted tombstones), so an in-place .bak copy would show up in
            # the bank picker and re-load the very rows this run removes.
            dst = backup_root / bank_dir.parent.parent.name / bank_dir.name
            print(f"    backing up -> {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bank_dir, dst)
        for tier, lab, new_arr, new_meta, new_entries, index, feats_d, metas_d in todo:
            feats_d[lab] = new_arr
            metas_d[lab] = new_meta
            index[lab] = new_entries
        bank.save(bank_dir, parts=("critical", "negative"))
        print("    written")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--projects-dir", required=True, type=Path)
    ap.add_argument("--project", default=None, help="project id; default: all")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                    help=f"per-image row cap (default {DEFAULT_CAP})")
    ap.add_argument("--device", default="cpu", help="device for the k-center selection")
    ap.add_argument("--apply", action="store_true", help="write the result (default: dry run)")
    ap.add_argument("--backup-dir", type=Path, default=None,
                    help="where --apply copies each bank before rewriting it "
                         "(default: <projects-dir>/../bank_backups/<timestamp>; "
                         "must be OUTSIDE the projects tree — the app treats "
                         "every directory under banks/ as a bank)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the bank-directory copy that --apply makes first")
    args = ap.parse_args()

    root: Path = args.projects_dir
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2
    backup_root: Path | None = None
    if args.apply and not args.no_backup:
        backup_root = args.backup_dir or (
            root.parent / "bank_backups" / time.strftime("%Y%m%d-%H%M%S")
        )
        if root.resolve() in backup_root.resolve().parents or backup_root.resolve() == root.resolve():
            print(f"backup dir {backup_root} is inside the projects tree — refusing")
            return 2
        print(f"backups -> {backup_root}")
    total_saved = 0
    errors: list[str] = []
    for pdir in sorted(root.iterdir()):
        if args.project and pdir.name != args.project:
            continue
        banks = pdir / "banks"
        if not banks.is_dir():
            continue
        for bdir in sorted(banks.iterdir()):
            if not bdir.is_dir() or bdir.name.startswith("."):
                continue
            print(f"{pdir.name[:8]} / {bdir.name}")
            rep = process_bank(bdir, args.cap, args.device, args.apply, backup_root)
            for row in rep["labels"]:
                print(f"    {row['tier']}/{row['label']}: {row['before']:,} -> "
                      f"{row['after']:,} (-{row['saved']:,})")
            for e in rep["errors"]:
                print(f"    ! {e}")
                errors.append(f"{pdir.name[:8]}/{bdir.name}: {e}")
            if not rep["labels"] and not rep["errors"]:
                print("    nothing over the cap")
            total_saved += rep["rows_saved"]

    print(f"\ntotal rows {'removed' if args.apply else 'that would be removed'}: "
          f"{total_saved:,}  (~{total_saved * BYTES_PER_ROW / 1e6:,.0f} MB)")
    if errors:
        print(f"{len(errors)} label(s) skipped — see the '!' lines above")
    if not args.apply and total_saved:
        print("dry run — re-run with --apply to write (a backup copy is made first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
