# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Where a bank's source images live, what they are called, and how many there are.

Three eras of the code wrote images into three places and every reader picked
its own subset:

* ``<bank>/_images/<tier>/`` — written only by the retired ``/bank/append``.
* ``<bank>/_staging/`` — dropped into the bank, not yet taught.
* ``<bank>/store/img/`` — the current path: import → label → assemble.

Readers that were never revisited after the store migration are the single
biggest source of bugs in this app: a card that paints a thumbnail above
"0 images", a viewer that 404s every re-labelled image, a delete that misses
half of what it should reclaim. This module is the one answer, so a fourth
place — or a fourth extension list — has exactly one site to update.

STDLIB ONLY, deliberately. ``routers/projects.py`` must stay free of
torch-adjacent modules (see its note above ``_IMAGE_EXTS``), and there is no
way to import a clscore submodule without running ``clscore/__init__.py``,
which imports ``clscore.bank`` → ``torch``. The layout names below are
therefore restated rather than imported; ``test_bank_layout_constants.py``
pins every one of them against its owner so the restatement cannot drift.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

# ---- layout ----------------------------------------------------------------
# Each of these is checked against its owning module by
# apps/api/tests/test_bank_layout_constants.py. Do not "simplify" that test
# away: renaming STORE_SUBDIR in clscore with only half the path pinned is
# exactly how the project card lost its thumbnail with a green suite.

BANKS_SUBDIR = "banks"
DELETED_MARKER = ".deleted"
BANK_META_FILE = "bank_meta.json"
IMAGES_SUBDIR = "_images"
STAGING_SUBDIR = "_staging"
STAGING_META_FILE = "staging.json"
STORE_SUBDIR = "store"
STORE_IMAGES_SUBDIR = "img"
STORE_INDEX_FILE = "store_index.json"
STORE_RENDER_SUBDIR = "cache"
LABELSETS_SUBDIR = "labelsets"
LABELSET_ACTIVE_MARKER = ".active"
# Key under which a bank row-range index entry names the store entry it came
# from. Absent on banks assembled before 2026-08, so every reader that uses it
# must keep a name-based fallback.
INDEX_ENTRY_ID_KEY = "entry_id"

TIERS: tuple[str, str, str] = ("normal", "critical", "negative")
"""Thumbnail precedence order. ``normal`` first is load-bearing."""

# assemble mints these for rows it cannot attribute to a source image; they are
# row ranges, not photographs, and must never be counted or thumbnailed.
_SYNTHETIC_NAME_PREFIX = "__unindexed__"


# ---- what is an image file? -------------------------------------------------
# Three sets, because there are genuinely three questions. There used to be
# five disagreeing lists.

DECODABLE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
"""What ``cv2.imdecode`` turns into an ndarray — the INPUT allowlist.

GIF is excluded on purpose: ``imdecode`` returns None for it, so admitting it
would put every decorative logo an archive carries into ``failed``.
"""

DISPLAYABLE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
"""What a browser renders as uploaded — the SERVE set.

Everything outside this is transcoded to PNG on write, and is never chosen as a
card thumbnail: a staged ``.tif`` is a real image, but a browser cannot draw it
and a broken ``<img>`` is worse than the placeholder.
"""

IMAGE_EXTS = DECODABLE_EXTS | DISPLAYABLE_EXTS
"""Everything that can legitimately BE a bank source image on disk.

Counting uses this. ``_staging/`` writes files verbatim with no transcode, so a
``.tif`` really does sit in a bank and really is one of its images.
"""


def has_image_ext(name: str | Path, exts: frozenset[str] = IMAGE_EXTS) -> bool:
    """True when this filename's suffix is in ``exts`` (case-insensitively)."""
    return Path(name).suffix.lower() in exts


# ---- what is it called on disk? ---------------------------------------------

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


class UnsafeImageName(ValueError):
    """The mapped name still escapes its tier directory — a traversal attempt."""


def safe_image_name(filename: str) -> str:
    """The one rule that decides what a taught image is called on disk.

    A MAPPING, never a validator. Both halves used to decide this for
    themselves: the writer replaced every character outside the set with "_",
    and the reader REJECTED any name containing one. That agrees only for a
    name the writer produced — and the name every caller actually holds is the
    ORIGINAL, recorded in ``bank_meta.bank_images`` and in the store index. So
    ``img001 (1)_豆_1.png`` sat on disk as ``img001__1____1.png`` while every
    request for it was refused as an "invalid image name".
    """
    base = (filename or "image.png").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return _UNSAFE_CHARS.sub("_", base)[:128] or "image.png"


# ---- where are the three roots? ---------------------------------------------


@dataclass(frozen=True)
class BankImageRoots:
    """The three places one bank's source images can be."""

    bank_dir: Path
    images: Path  # <bank>/_images      (append era; per-tier below)
    staging: Path  # <bank>/_staging     (dropped, untaught)
    store_images: Path  # <bank>/store/img    (current era)

    def tier(self, tier: str) -> Path:
        """``<bank>/_images/<tier>`` — the append era's per-tier directory."""
        return self.images / tier


def image_roots(bank_dir: Path) -> BankImageRoots:
    """The three roots for ``bank_dir``. Pure path arithmetic; nothing is stat'd."""
    bank_dir = Path(bank_dir)
    return BankImageRoots(
        bank_dir=bank_dir,
        images=bank_dir / IMAGES_SUBDIR,
        staging=bank_dir / STAGING_SUBDIR,
        store_images=bank_dir / STORE_SUBDIR / STORE_IMAGES_SUBDIR,
    )


def tier_image_path(bank_dir: Path, tier: str, name: str) -> Path:
    """``<bank>/_images/<tier>/<safe name>`` for a caller-held ORIGINAL filename.

    Raises :class:`UnsafeImageName` when the result escapes the tier directory.
    The mapping keeps "." and "-", so ".." still reaches the guard, which is
    the actual protection and is unchanged.
    """
    root = image_roots(bank_dir).tier(tier)
    p = (root / safe_image_name(name)).resolve()
    if not p.is_relative_to(root.resolve()):
        raise UnsafeImageName(f"invalid image path: {name!r}")
    return p


def staging_file_path(bank_dir: Path, name: str) -> Path | None:
    """``<bank>/_staging/<name>`` for a name taken from the staging LISTING.

    Deliberately NOT mapped through :func:`safe_image_name`: the staging
    listing is derived from the directory itself, so the on-disk name IS the
    name. Returns None when the result escapes the staging directory.
    """
    root = image_roots(bank_dir).staging
    p = (root / name).resolve()
    return p if p.is_relative_to(root.resolve()) else None


def resolve_image_ref(
    bank_dir: Path, image_ref: str, *, owned_only: bool = False
) -> Path | None:
    """The file a store entry's ``image_ref`` names, or None if it escapes.

    ``image_ref`` is read off disk, so containment is checked rather than
    trusted. The default boundary is the BANK directory, because a migrated
    entry legitimately points into ``_images/<tier>/``. ``owned_only=True``
    narrows it to the two roots this app writes — ``store/img/`` and
    ``_images/`` — and is what a caller must use before *unlinking*, so a
    hand-edited index can never name ``bank.npy``.

    Existence is NOT checked: the serving caller 404s on a missing file, and
    the deleting caller still wants to drop the entry's cached renditions.
    """
    if not image_ref:
        return None
    bank_dir = Path(bank_dir)
    p = (bank_dir / image_ref).resolve()
    if owned_only:
        roots = image_roots(bank_dir)
        allowed = (roots.store_images.resolve(), roots.images.resolve())
        return p if any(p.is_relative_to(a) for a in allowed) else None
    return p if p.is_relative_to(bank_dir.resolve()) else None


# ---- reading the two indices ------------------------------------------------
# Cached on (mtime_ns, size): _store_source used to re-parse the whole store
# index AND the label set on every single thumbnail request, so a 635-image
# project paid 635 x 160 KB to render one grid.

_JSON_CACHE: dict[str, tuple[tuple[int, int], object]] = {}
_JSON_LOCK = threading.Lock()


def _read_json(path: Path) -> object | None:
    try:
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    stamp = (st.st_mtime_ns, st.st_size)
    with _JSON_LOCK:
        hit = _JSON_CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    with _JSON_LOCK:
        _JSON_CACHE[key] = (stamp, data)
        if len(_JSON_CACHE) > 64:  # a handful of banks, not a leak
            _JSON_CACHE.pop(next(iter(_JSON_CACHE)))
    return data


def store_entries(bank_dir: Path) -> list[dict]:
    """Entries from ``store_index.json``, synthetic row-only names filtered out."""
    data = _read_json(Path(bank_dir) / STORE_SUBDIR / STORE_INDEX_FILE)
    if not isinstance(data, dict):
        return []
    out = []
    for e in data.get("entries", []) or []:
        if not isinstance(e, dict):
            continue
        if str(e.get("name", "")).startswith(_SYNTHETIC_NAME_PREFIX):
            continue
        out.append(e)
    return out


def active_assignments(bank_dir: Path) -> dict[str, dict]:
    """The active label set's ``assignments``, or ``{}`` when there is none."""
    d = Path(bank_dir) / LABELSETS_SUBDIR
    try:
        wanted = (d / LABELSET_ACTIVE_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        wanted = ""
    candidates = [d / f"{wanted}.json"] if wanted else []
    if not candidates:
        try:
            candidates = sorted(p for p in d.iterdir() if p.suffix == ".json")
        except OSError:
            return {}
    for path in candidates:
        data = _read_json(path)
        if isinstance(data, dict) and isinstance(data.get("assignments"), dict):
            return {k: v for k, v in data["assignments"].items() if isinstance(v, dict)}
    return {}


# ---- where is the image for (bank, tier, name)? -----------------------------


def resolve_bank_image(
    bank_dir: Path, tier: str, name: str, *, entry_id: str = ""
) -> Path | None:
    """The source image for one row-range of the assembled bank, whichever era wrote it.

    ``_images/<tier>/`` first (the append era kept its own copy), then the
    store's. In the store the entry is found by, in order:

    1. ``entry_id`` — exact, when the bank index carries the stamp;
    2. name + the ACTIVE label set's tier — the historical rule, kept so that
       the same filename in two tiers never serves the wrong one;
    3. name, when EXACTLY ONE store entry bears it — unambiguous by
       construction, and the only rule that survives a re-label or an unassign
       on a bank assembled before the stamp existed.

    Returns None on a miss and on a traversing name: a name that escapes its
    tier directory is not an image *of this bank*, and 404 is the right answer
    for the route. Callers who want the exception use :func:`tier_image_path`.
    """
    bank_dir = Path(bank_dir)
    try:
        taught = tier_image_path(bank_dir, tier, name)
    except UnsafeImageName:
        return None
    if taught.is_file():
        return taught

    entries = store_entries(bank_dir)
    if not entries:
        return None

    if entry_id:
        for e in entries:
            if str(e.get("id", "")) == entry_id:
                return resolve_image_ref(bank_dir, str(e.get("image_ref", "")))

    named = [e for e in entries if str(e.get("name", "")) == name and e.get("image_ref")]
    if not named:
        return None

    assignments = active_assignments(bank_dir)
    for e in named:
        a = assignments.get(str(e.get("id", "")))
        if a is not None and a.get("tier") == tier:
            return resolve_image_ref(bank_dir, str(e.get("image_ref", "")))

    if len(named) == 1:
        return resolve_image_ref(bank_dir, str(named[0].get("image_ref", "")))
    return None


def bank_image_url(tier: str, name: str, *, entry_id: str = "") -> str:
    """``/bank/images/<tier>/<name>`` with the name percent-encoded.

    Store-era names are the operator's ORIGINAL filenames, so ``lot#3.png``
    truncates at the fragment if this is built by concatenation. The client's
    other URL builders already encode; this is the one that never did.
    """
    url = f"/bank/images/{quote(tier, safe='')}/{quote(name, safe='')}"
    return f"{url}?id={quote(entry_id, safe='')}" if entry_id else url


# ---- what does this bank hold, and how much of it? --------------------------


def first_bank_image(banks_root: Path) -> Path | None:
    """First DISPLAYABLE image across a project's banks.

    ``_images/`` (in tier order), then ``_staging/``, then ``store/img/``.

    Directory listings only — no JSON is opened. This runs once per project on
    every summary rebuild, and parsing a 160 KB store index twenty times over
    to learn one filename is not worth it.
    """
    banks_root = Path(banks_root)
    if not banks_root.is_dir():
        return None
    for bank_dir in sorted(p for p in banks_root.iterdir() if p.is_dir()):
        if (bank_dir / DELETED_MARKER).exists():
            continue  # partially-deleted bank — mirrors list_banks' skip
        roots = image_roots(bank_dir)
        for d in (*(roots.tier(t) for t in TIERS), roots.staging, roots.store_images):
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file() and has_image_ext(f, DISPLAYABLE_EXTS):
                    return f
    return None


@dataclass(frozen=True)
class BankImageCensus:
    """How many source images a bank (or a project) holds."""

    images: int = 0
    labeled: int = 0
    staged: int = 0
    basis: str = "empty"  # "store" | "bank_meta" | "empty" | "mixed"

    def __add__(self, other: BankImageCensus) -> BankImageCensus:
        bases = {self.basis, other.basis} - {"empty"}
        return BankImageCensus(
            images=self.images + other.images,
            labeled=self.labeled + other.labeled,
            staged=self.staged + other.staged,
            basis=bases.pop() if len(bases) == 1 else ("mixed" if bases else "empty"),
        )


def _staged_count(staging_dir: Path) -> tuple[int, int]:
    """``(files, judged)`` in ``_staging/`` — images only, ``staging.json`` aside."""
    if not staging_dir.is_dir():
        return 0, 0
    meta = _read_json(staging_dir / STAGING_META_FILE)
    if not isinstance(meta, dict):
        meta = {}
    files = judged = 0
    try:
        listing = list(staging_dir.iterdir())
    except OSError:
        return 0, 0
    for f in listing:
        # A stray Thumbs.db or .DS_Store used to inflate the card: the count
        # excluded only staging.json and *.tmp, while the thumbnail lookup
        # twenty lines away filtered by extension.
        if f.is_file() and has_image_ext(f, IMAGE_EXTS):
            files += 1
            if meta.get(f.name):
                judged += 1
    return files, judged


def bank_census(bank_dir: Path) -> BankImageCensus:
    """How many source images this bank holds, and how many carry a judgement.

    The store wins when it has entries: it is a SUPERSET of the assembled bank
    (assembly only ever selects assigned entries), so counting both would
    double, and counting only ``bank_meta`` makes an ingested-but-unassembled
    project read "0 images" underneath its own thumbnail.

    ``labeled`` for the store era is the INTERSECTION of the active label set's
    assignments with the entries that still exist, so an assignment left
    dangling by a delete cannot inflate the card.
    """
    bank_dir = Path(bank_dir)
    staged, staged_judged = _staged_count(image_roots(bank_dir).staging)

    entries = store_entries(bank_dir)
    if entries:
        assignments = active_assignments(bank_dir)
        labeled = sum(1 for e in entries if str(e.get("id", "")) in assignments)
        return BankImageCensus(
            images=len(entries) + staged,
            labeled=labeled + staged_judged,
            staged=staged,
            basis="store",
        )

    meta = _read_json(bank_dir / BANK_META_FILE)
    if isinstance(meta, dict):
        # Taught images always carry a tier, so they all count as labelled.
        taught = (
            len(meta.get("bank_images", []) or [])
            + sum(len(v or []) for v in (meta.get("critical_images") or {}).values())
            + sum(len(v or []) for v in (meta.get("negative_images") or {}).values())
        )
        if taught or staged:
            return BankImageCensus(
                images=taught + staged,
                labeled=taught + staged_judged,
                staged=staged,
                basis="bank_meta",
            )

    if staged:
        return BankImageCensus(
            images=staged, labeled=staged_judged, staged=staged, basis="bank_meta"
        )
    return BankImageCensus()


def project_census(banks_root: Path) -> BankImageCensus:
    """:func:`bank_census` summed over a project's banks, skipping deleted ones."""
    banks_root = Path(banks_root)
    total = BankImageCensus()
    if not banks_root.is_dir():
        return total
    for bank_dir in banks_root.iterdir():
        if not bank_dir.is_dir() or (bank_dir / DELETED_MARKER).exists():
            continue
        total = total + bank_census(bank_dir)
    return total


def unassigned_count(entry_ids: list[str] | set[str], assignments: dict) -> int:
    """Store entries with no assignment, counted set-wise.

    Subtracting two independently-maintained totals (``len(store) - assigned``)
    under-reports by exactly the number of assignments left dangling by a
    delete, and needs a ``max(0, ...)`` to stop going negative. The clamp hid
    the divergence instead of surfacing it, so the Bank tab ticked "labelled"
    green over images that were never labelled.
    """
    return sum(1 for i in set(entry_ids) if i not in assignments)
