# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Grouping taught images so validation can leave out more than one at a time.

The separation check scores every taught image against the bank with its own
rows excluded — leave-one-image-out, which is k-fold at k=N and therefore the
strongest setting there is *per image*. It is still optimistic, for a reason
k has nothing to do with: a line photographed continuously produces near
duplicates, so excluding one frame leaves its twins in the bank and the query
finds itself at one remove. The number that comes out answers "can it
recognise this part again", not "can it recognise a part it has never seen".

Grouping fixes the unit of exclusion. Put every frame of a lot (or a shooting
day) in one group, exclude the whole group, and the check answers the question
the line actually asks.

Three ways to derive the group, because naming conventions differ per site:

    datetime  pull a date out of the filename (``OK_170104_094937_...``)
    prefix    the first N ``sep``-separated fields (lot numbers, fixtures)
    manual    an explicit assignment stored per image

``none`` keeps the old leave-one-image-out behaviour and is the default —
changing what a saved AUROC means without being asked would be worse than
leaving it optimistic.
"""

from __future__ import annotations

import re

__all__ = [
    "GROUP_MODES",
    "derive_groups",
    "exclusion_ranges",
    "group_of",
    "group_summary",
]

GROUP_MODES = ("none", "datetime", "prefix", "manual")

# Tried in order, most specific first, and every candidate is validated as a
# real calendar date before it is accepted. That check is not a nicety: the
# filenames on these lines carry serials shaped exactly like dates —
# ``OK_0002-08-02-0008.jpg`` reads as 0002-08-02 — and accepting them produces
# one group per image, which is leave-one-out again under another name.
#
# Each entry captures (year, month, day). ``ymd4`` carries a century and is
# range-checked against it; ``ymd2`` is a two-digit year, which no filename
# disambiguates, so it is kept as-is rather than padded with a guess.
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 2017-10-16_12-46-58 / ok_Image__2017-05-16__09-33-06
    (re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})(?!\d)"), "ymd4"),
    # 20260804_094937
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[_-]\d{6}(?!\d)"), "ymd4"),
    # OK_170104_094937_...   YYMMDD_HHMMSS
    (re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})[_-]\d{6}(?!\d)"), "ymd2"),
    # OK_a02_02_170302164558368.png — YYMMDDHHMMSS with optional milliseconds,
    # run together with no separator at all.
    (re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})\d{6,11}(?!\d)"), "ymd2"),
    # A bare 8-digit date.
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"), "ymd4"),
)


def _plausible(year: int, month: int, day: int, has_century: bool) -> bool:
    """Cheap calendar sanity check — enough to reject serial numbers.

    Day 31 is allowed in every month on purpose: rejecting 2026-02-31 would
    need a real calendar for no gain, since a filename that carries an
    impossible day is already not the date it claims to be and the group it
    forms is harmless either way.
    """
    if has_century and not (1900 <= year <= 2099):
        return False
    return 1 <= month <= 12 and 1 <= day <= 31


def _stem(name: str) -> str:
    base = str(name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def _datetime_group(name: str) -> str:
    for pattern, kind in _DATE_PATTERNS:
        # Every match, not just the first: a leading serial that happens to fit
        # the shape must not stop a real date later in the same filename.
        for m in pattern.finditer(name):
            y, mo, d = m.group(1), m.group(2), m.group(3)
            if _plausible(int(y), int(mo), int(d), kind == "ymd4"):
                return f"{y}{mo}{d}"
    return ""


def _prefix_group(name: str, sep: str, fields: int) -> str:
    stem = _stem(name)
    if not sep:
        return stem
    parts = stem.split(sep)
    n = max(1, int(fields))
    # Asking for more fields than the name has means the whole stem is the
    # group, which makes every image its own group — the same thing as no
    # grouping, and better than silently grouping unrelated files together.
    return sep.join(parts[:n])


def group_of(
    name: str,
    mode: str,
    *,
    sep: str = "_",
    fields: int = 1,
    manual: dict[str, str] | None = None,
) -> str:
    """Group key for one image, or ``""`` when it has none.

    An empty key is not an error: it means the image is validated on its own,
    exactly as before. Filenames that carry no date should not be silently
    lumped into one giant "no date" group — that would exclude half the bank
    and make the check pessimistic in a way nobody asked for.
    """
    if mode == "datetime":
        return _datetime_group(name)
    if mode == "prefix":
        return _prefix_group(name, sep, fields)
    if mode == "manual":
        return str((manual or {}).get(name, "")).strip()
    return ""


def derive_groups(
    names: list[str],
    mode: str,
    *,
    sep: str = "_",
    fields: int = 1,
    manual: dict[str, str] | None = None,
) -> dict[str, str]:
    """``{image name: group key}``; keys are ``""`` for ungrouped images."""
    if mode not in GROUP_MODES:
        raise ValueError(f"unknown group mode: {mode!r}")
    if mode == "none":
        return {}
    return {
        n: group_of(n, mode, sep=sep, fields=fields, manual=manual) for n in names
    }


def group_summary(groups: dict[str, str]) -> dict[str, list[str]]:
    """``{group key: [names]}`` for the UI's "what will this split into" preview.

    Ungrouped images are collected under ``""`` so the caller can show how many
    the rule failed to place — the number that says whether the convention was
    guessed right.
    """
    out: dict[str, list[str]] = {}
    for name, key in sorted(groups.items()):
        out.setdefault(key, []).append(name)
    return out


def exclusion_ranges(
    index: list[dict],
    groups: dict[str, str],
    name: str,
) -> list[tuple[int, int]]:
    """Bank row ranges to mask when validating ``name``.

    Its own rows always, plus every other image sharing a non-empty group key.
    ``index`` is a ``BankMeta.*_image_index`` list, so each entry already
    carries the contiguous ``(start, count)`` the assembly guarantees.
    """
    key = groups.get(name, "")
    out: list[tuple[int, int]] = []
    for e in index:
        n = str(e.get("name", ""))
        if n != name and not (key and groups.get(n, "") == key):
            continue
        start, count = int(e.get("start", -1)), int(e.get("count", 0))
        if start >= 0 and count > 0:
            out.append((start, count))
    return sorted(out)
