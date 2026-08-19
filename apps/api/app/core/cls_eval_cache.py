# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Image-evaluation cache and bank-content fingerprints.

An image's eval is a pure function of the bank contents, so results are
cached per (bank dir, membership fingerprint): in memory for the process
lifetime and mirrored to ``<bank>/eval_cache.json`` so they survive restarts
and page reloads. Any append/delete/move changes the fingerprint, which
silently invalidates the whole cache. Extracted verbatim from
``routers/bank.py`` during the post-audit split; behaviour is unchanged.
"""

from __future__ import annotations

import hashlib
import json

from .bank_images import INDEX_ENTRY_ID_KEY
from .cls_state import ClsStudioState
from .paths import write_bytes_atomic
from .runtime_compression import read_compression_settings

_EVAL_CACHE: dict[str, dict[str, dict]] = {}


def _index_basis(index: list[dict]) -> list[dict]:
    # Membership only — annotations (defect marks) also live on the index
    # entries but don't affect the raw scores, so including them here would
    # needlessly throw away a whole sweep on every mark edit.
    return [
        {"name": e.get("name", ""), "start": e.get("start", -1), "count": e.get("count", 0)}
        for e in index
    ]


def _eval_fingerprint(bank) -> str:
    # Raw eval scores are top-k distances to the NORMAL bank only, so only
    # normal-tier membership can invalidate them. Adding / deleting critical
    # or negative images keeps every other image's eval valid — those edits
    # are handled entry-wise (new image: just evaluate it; deletion:
    # _eval_cache_purge) instead of throwing the whole sweep away.
    basis = {
        "normal_rows": int(bank.normal.shape[0]),
        "normal": _index_basis(bank.meta.normal_image_index),
        # Raw scores depend on how the normal tensor is compressed (int8
        # round-trip, IVF candidate routing), so a settings flip must
        # invalidate every cached eval the same way a teach does.
        "compression": read_compression_settings(),
    }
    return hashlib.sha1(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()


def _legacy_eval_fingerprint(bank) -> str:
    # The pre-normal-only basis. Accepted read-only in _eval_cache_for so a
    # cache written by the previous build survives the upgrade; the next
    # save re-stamps it with the new fingerprint. Delete once every active
    # bank has been re-saved.
    basis = {
        "normal_rows": int(bank.normal.shape[0]),
        "normal": _index_basis(bank.meta.normal_image_index),
        "critical": {k: _index_basis(v) for k, v in bank.meta.critical_image_index.items()},
        "negative": {k: _index_basis(v) for k, v in bank.meta.negative_image_index.items()},
    }
    return hashlib.sha1(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()


def _bank_content_fingerprint(bank) -> str:
    """Full content fingerprint for runtime-config staleness.

    Unlike the eval fingerprint this covers everything that feeds the
    suggested threshold: normal membership, labelled-tier membership AND
    the defect marks (raw index entries carry ``annotations``), so any
    teach / delete / mark edit flags a previously saved verdict recipe as
    needing a re-check.
    """
    basis = {
        "normal_rows": int(bank.normal.shape[0]),
        "normal": bank.meta.normal_image_index,
        "critical": bank.meta.critical_image_index,
        "negative": bank.meta.negative_image_index,
        # A verdict threshold picked under one compression config does not
        # transfer to another — flag the saved recipe stale on a flip.
        "compression": read_compression_settings(),
    }
    return hashlib.sha1(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()


def _eval_cache_for(state: ClsStudioState) -> dict[str, dict]:
    cache_key = f"{state.bank_dir}|{_eval_fingerprint(state.bank)}"
    hit = _EVAL_CACHE.get(cache_key)
    if hit is not None:
        return hit
    data: dict[str, dict] = {}
    if state.bank_dir is not None:
        path = state.bank_dir / "eval_cache.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                cfg = read_compression_settings()
                accepted = {_eval_fingerprint(state.bank)}
                if not (cfg["int8"] or cfg["ivf"]):
                    # Legacy caches predate compression and were computed on
                    # the raw fp16 full scan — only valid while both
                    # transforms are off.
                    accepted.add(_legacy_eval_fingerprint(state.bank))
                if raw.get("fingerprint") in accepted:
                    data = raw.get("results", {})
            except (OSError, ValueError):
                data = {}
    _EVAL_CACHE.clear()  # one active bank at a time — drop stale entries
    _EVAL_CACHE[cache_key] = data
    return data


def eval_cache_key(tier: str, label: str, entry: dict) -> str:
    """The cache key for one row-range index entry.

    ``"<tier>/<label>/<name>"``, plus ``"#<entry_id>"`` when the bank index
    carries the stamp. The filename alone is not an identity: the store lets
    two entries share one, and deleting a photo and re-ingesting a different
    one under the same name is ordinary line work (a retake of lot1_003.png).
    The only freshness guard used to be ``patches == count``, and on a line
    where every frame comes off the same camera at the same resolution that
    compares equal every time -- so the retake was served the deleted photo's
    scores, and its ranked ``top_indices`` selected the NEW image's rows at
    the OLD image's positions for the exemplar block and the alpha term.

    Banks assembled before the stamp existed carry no id and keep the old key,
    so their behaviour is unchanged. On a stamped bank the shape changes once
    and the sweep is recomputed; the cache is disposable by design.
    """
    base = f"{tier}/{label}/{entry.get('name', '')}"
    entry_id = str(entry.get(INDEX_ENTRY_ID_KEY, ""))
    return f"{base}#{entry_id}" if entry_id else base


def bank_eval_keys(bank) -> set[str]:
    """Every ``"<tier>/<label>/<name>"`` the assembled bank actually contains.

    The membership answer the cache has no way to derive for itself: its keys
    are strings, and nothing in them says whether the image is still in the
    bank.
    """
    keys = {
        eval_cache_key("normal", "", e)
        for e in bank.meta.normal_image_index
        if e.get("name")
    }
    for tier, index in (
        ("critical", bank.meta.critical_image_index),
        ("negative", bank.meta.negative_image_index),
    ):
        for label, entries in index.items():
            keys.update(eval_cache_key(tier, label, e) for e in entries if e.get("name"))
    return keys


def eval_cache_live_entries(state: ClsStudioState) -> dict[str, dict]:
    """The cached evals whose image is still in the bank.

    ``/bank/evaluation/cached`` promises "the active bank's CURRENT contents"
    and used to return the raw cache instead. The two only agreed while the
    fingerprint rolled on every change — and ``_eval_fingerprint`` is
    deliberately normal-tier-only, so removing a critical or negative image
    leaves it identical and the orphan survives every restart. The separation
    histogram, the AUROC and the Youden auto-threshold are all computed over
    what this returns, so an image the operator deleted kept voting.
    """
    live = bank_eval_keys(state.bank)
    # list() first: another request may mutate the shared cache mid-loop.
    return {k: v for k, v in list(_eval_cache_for(state).items()) if k in live}


def eval_cache_reconcile(state: ClsStudioState) -> int:
    """Drop cached evals for images the bank no longer holds. Returns the count.

    Garbage collection for the invariant above. Called after an assemble,
    which is the live way a labelled image leaves a bank: /store/delete then
    /bank/assemble. The two purge hooks that already existed hang off
    /bank/clear/{tier} and /bank/images/delete, neither of which the shipped
    UI can reach.
    """
    cache = _eval_cache_for(state)
    live = bank_eval_keys(state.bank)
    orphans = [k for k in list(cache.keys()) if k not in live]
    for k in orphans:
        cache.pop(k, None)
    if orphans:
        _eval_cache_save(state, cache)
    return len(orphans)


def _eval_cache_purge(state: ClsStudioState, tier: str, label: str | None, names: list[str]) -> None:
    """Drop cached evals for images removed from a labelled tier.

    Normal-tier deletions change the normal bank itself, which rolls the
    eval fingerprint and invalidates everything — nothing to do here. For
    critical / negative deletions the other images' raw scores stay valid,
    so only the deleted images' entries go (``label=None`` matches any
    label, mirroring the delete route's semantics).
    """
    if tier == "normal" or not names:
        return
    cache = _eval_cache_for(state)
    tgt = set(names)
    removed = False
    for k in list(cache.keys()):
        parts = k.split("/", 2)
        if len(parts) != 3:
            continue
        kt, kl, kn = parts
        if kt != tier or (label is not None and kl != label):
            continue
        # ``eval_cache_key`` appends "#<entry_id>" as the LAST separator, and
        # the names it embeds are the operator's ORIGINAL filenames, kept
        # verbatim (bank_images.safe_image_name is the on-disk mapping only).
        # So BOTH forms have to be tried, and neither one alone is enough:
        #
        #   stamped   "lot#3.png#000007" -> rsplit gives "lot#3.png"   ✓
        #   unstamped "lot#3.png"        -> rsplit gives "lot"         ✗, but
        #                                   the key IS the name        ✓
        #
        # This used to be ``kn.split("#", 1)[0]`` -- the FIRST "#" -- which
        # turned both of the above into "lot" and silently matched nothing, so
        # every cached eval whose filename contained a "#" survived its own
        # image's deletion. "#" in a filename has already had to be fixed twice
        # for URLs (bank_images.bank_image_url, routers/images.py).
        if kn in tgt or kn.rsplit("#", 1)[0] in tgt:
            cache.pop(k, None)
            removed = True
    if removed:
        _eval_cache_save(state, cache)


def _eval_cache_save(
    state: ClsStudioState,
    results: dict[str, dict],
    fingerprint: str | None = None,
) -> None:
    if state.bank_dir is None:
        return
    try:
        fp_now = _eval_fingerprint(state.bank)
        if fingerprint is not None and fingerprint != fp_now:
            # The caller's results were computed against a bank that has
            # since changed (a teach raced the evaluation) — persisting them
            # under the CURRENT fingerprint would present stale numbers as
            # fresh. Drop the write; the next sweep recomputes cleanly.
            return
        payload = {"fingerprint": fp_now, "results": results}
        # Atomic: a crash mid-write must leave the previous mirror intact, not
        # a torn JSON that silently voids the whole sweep on next load.
        write_bytes_atomic(
            state.bank_dir / "eval_cache.json", json.dumps(payload).encode("utf-8")
        )
    except OSError:
        pass  # best-effort mirror; the in-memory cache still works
