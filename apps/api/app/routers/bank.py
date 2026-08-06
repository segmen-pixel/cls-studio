# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Memory-bank management for the active cls-studio project.

A project holds one or more named banks under ``<project_dir>/banks/``.
Selecting a project (and optionally a bank) loads it into the process-wide
:class:`ClsStudioState`; the remaining routes read or mutate that bank. Only
``append`` needs the DINOv2 backbone (loaded lazily); every other route is
pure bank bookkeeping and works without torch.hub.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from clscore.bank import DEFAULT_LABEL, coreset_reduce_indexed
from clscore.compress import IVF_INDEX_FILE
from clscore.feature_extractor import (
    extract_image_features_for_bank,
    iter_images_features_batched,
)
from clscore.store import STORE_SUBDIR

from ..core.cls_edge_export import EDGE_EXPORT_SUFFIX, build_edge_package
from ..core.cls_eval_cache import (
    _bank_content_fingerprint,
    _eval_cache_for,
    _eval_cache_purge,
    _eval_cache_save,
    _eval_fingerprint,
)
from ..core.cls_exemplar import _exemplar_rows
from ..core.cls_projection import compute_projection
from ..core.cls_schemas import (
    AnnotateResult,
    AnnotationRect,
    AppendResult,
    BankCapacityInfo,
    BankCapacitySet,
    BankInfo,
    BankList,
    BankState,
    ImageCmin,
    LabelsResponse,
    ProjectionResponse,
    RuntimeConfig,
    SelectResult,
    StoredImageEval,
    Tier,
)
from ..core.cls_state import ClsStudioState, bank_state_of, get_state, slug_bank_id
from ..core.config import MAX_ARCHIVE_BYTES

# Bank mutations must invalidate the cached /projects/summary payload the same
# way annotate mutations do (touch_project): the project card's thumbnail /
# has_bank_thumbnail flag comes from that summary, and without the touch a
# teach followed by a quick return to the project list keeps serving the
# pre-teach cache — the card shows "no images" even though the bank has them.
from ..core.db_utils import touch_project
from ..core.paths import write_bytes_atomic
from ..core.runtime_compression import read_compression_settings
from ..core.security import check_upload_batch, read_upload

router = APIRouter(tags=["bank"])


def check_binding(state: ClsStudioState, binding: str | None) -> None:
    """Reject a mutation aimed at a different bank than the caller intended.

    Every mutating / scoring route operates on the process-global active
    bank, which any LAN client can re-bind via ``/bank/select`` at any
    moment. Clients therefore send ``X-Bank-Binding: <project_id>/<bank_id>``
    (from their last SelectResult); a mismatch means another client re-bound
    the bank between that select and this request — without this check the
    write lands in the wrong project's bank and still returns 200 (same
    family as the 2026-07-17 deleted-project incident, minus the delete).
    Requests without the header (older clients, curl) are accepted as-is.
    """
    if not binding:
        return
    have = f"{state.active_project_id}/{state.active_bank_id}"
    if binding.strip() != have:
        raise HTTPException(
            status_code=409,
            detail=(
                f"active bank changed: request bound to {binding.strip()}, "
                f"server is on {have} — re-select and retry"
            ),
        )

# Per-image patch cap for the normal tier. Each taught OK image contributes at
# most this many coreset-selected patches, bounding bank growth (~5x smaller
# than the raw ~11k patches a 1MP image yields) without touching earlier
# images' rows. Override via env; 0 disables the cap entirely.
_DEFAULT_MAX_PATCHES_PER_IMAGE = 2048


def _max_patches_per_image() -> int:
    raw = os.environ.get("CLS_MAX_PATCHES_PER_IMAGE")
    if raw is None:
        return _DEFAULT_MAX_PATCHES_PER_IMAGE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MAX_PATCHES_PER_IMAGE


def _banks(state: ClsStudioState) -> list[BankInfo]:
    if state.active_project_id is None:
        return []
    return [BankInfo(**b) for b in state.list_banks(state.active_project_id)]


def _select_result(state: ClsStudioState) -> SelectResult:
    return SelectResult(
        project_id=state.active_project_id or "",
        bank_id=state.active_bank_id or "",
        bank_dir=str(state.bank_dir),
        device=state.device,
        bank=bank_state_of(state.bank),
        banks=_banks(state),
    )


class SelectRequest(BaseModel):
    project_id: str
    bank_id: str | None = None


@router.post("/bank/select", response_model=SelectResult)
def select_project(body: SelectRequest) -> SelectResult:
    """Activate ``project_id`` (and a bank) and load it.

    The project must exist in the DB, not just on disk: a stale client
    (sessionStorage, another LAN browser) re-selecting a deleted project
    would otherwise re-create its bank tree as a DB-orphan ghost that the
    startup purge later removes together with anything taught into it.
    """
    from sqlmodel import Session

    from ..db import get_engine
    from ..models import Project

    with Session(get_engine()) as session:
        if session.get(Project, body.project_id) is None:
            raise HTTPException(
                status_code=404, detail=f"project not found: {body.project_id}"
            )
    state = get_state()
    with state.lock:
        state.activate(body.project_id, body.bank_id)
        return _select_result(state)


@router.get("/banks", response_model=BankList)
def list_banks() -> BankList:
    """Every memory bank in the active project."""
    state = get_state()
    state.require_active_project()
    return BankList(banks=_banks(state), active_bank_id=state.active_bank_id)


class CreateBankRequest(BaseModel):
    name: str


@router.post("/banks/create", response_model=SelectResult)
def create_bank(body: CreateBankRequest) -> SelectResult:
    """Create a new empty bank in the active project and select it."""
    state = get_state()
    with state.lock:
        state.create_bank(body.name)
        return _select_result(state)


class SelectBankRequest(BaseModel):
    bank_id: str


@router.post("/banks/select", response_model=SelectResult)
def select_bank(body: SelectBankRequest) -> SelectResult:
    """Select a different bank within the active project."""
    state = get_state()
    state.require_active_project()
    with state.lock:
        assert state.active_project_id is not None
        state.activate(state.active_project_id, body.bank_id)
        return _select_result(state)


class DeleteBankRequest(BaseModel):
    bank_id: str


@router.post("/banks/delete", response_model=SelectResult)
def delete_bank(body: DeleteBankRequest) -> SelectResult:
    """Delete a bank from the active project and re-select a remaining one.

    Irreversible — removes the whole bank directory (feature arrays, defect
    marks, verdict recipe, thumbnails). Deleting the active bank falls through
    to the project's next bank; deleting the last bank lands on a fresh empty
    ``default`` so the selector always has an entry. Returns the same shape as
    bank selection, with the newly-active bank loaded.
    """
    state = get_state()
    state.require_active_project()
    with state.lock:
        state.delete_bank(body.bank_id)
        result = _select_result(state)
    if state.active_project_id:
        touch_project(state.active_project_id)
    return result


@router.get("/bank", response_model=BankState)
def get_bank() -> BankState:
    """Current tier sizes and per-label breakdown for the active bank."""
    state = get_state()
    state.require_active()
    return bank_state_of(state.bank)


@router.get("/bank/labels", response_model=LabelsResponse)
def get_labels() -> LabelsResponse:
    state = get_state()
    state.require_active()
    return LabelsResponse(
        critical=sorted(state.bank.label_sizes("critical")),
        negative=sorted(state.bank.label_sizes("negative")),
    )


@router.get("/bank/projection", response_model=ProjectionResponse)
def get_projection(
    mode: str = "auto",
    max_points_per_tier: int = 800,
    alpha: float = 1.0,
    seed: int = 42,
    with_scores: bool = False,
    guarantee_top: int = 10,
    granularity: str = "patch",
) -> ProjectionResponse:
    """2D projection of the active bank's patches for the visualisation panel.

    Modes:
        * ``normal`` — PCA fit on the normal bank only. Axes reflect
          appearance variation among OK patches; always available as long
          as ``normal`` has >=2 patches.
        * ``anomaly`` — contrastive PCA (Cov(NG) - alpha*Cov(normal)).
          Requires both tiers to have >=2 patches; falls back to
          ``normal`` mode otherwise.
        * ``auto`` (default) — picks ``anomaly`` if both tiers are
          non-empty, else ``normal``.

    Patch counts routinely reach hundreds of thousands, so each tier is
    downsampled to ``max_points_per_tier`` uniformly at random (seeded
    for reproducibility) before projection. ``total`` and ``sampled`` in
    the response report the ratio so the UI can flag when the plot is a
    sample rather than the full bank.

    A uniform sample almost never contains the interesting rows: a bank of
    ~2M patches holds only a few hundred defective ones, so their expected
    count in a 1,600-point draw is below one. ``guarantee_top`` therefore
    force-includes each image's ``guarantee_top`` highest-scoring rows
    (from the separation-check eval cache; images never evaluated are
    skipped) before random fill. Set 0 to disable.

    ``granularity="image"`` collapses each taught image to one point: the
    mean of its cached top-``guarantee_top`` patch features (mean of all
    rows when no eval is cached), scored with the same top-k mean. This is
    the view that corresponds 1:1 with the separation-check histogram.
    """
    state = get_state()
    state.require_active()
    return compute_projection(
        state,
        mode=mode,
        max_points_per_tier=max_points_per_tier,
        alpha=alpha,
        seed=seed,
        with_scores=with_scores,
        guarantee_top=guarantee_top,
        granularity=granularity,
    )


@router.get("/bank/evaluation/cached", response_model=list[StoredImageEval])
def cached_evaluations() -> list[StoredImageEval]:
    """Already-computed image evals for the active bank's current contents.

    Lets the UI restore a finished separation check after a page reload
    without re-scoring anything. Returns [] when the bank changed since.
    """
    state = get_state()
    state.require_active()
    # Snapshot the shared eval cache before iterating: another request may
    # mutate it mid-loop (single global state), which would otherwise raise
    # "dictionary changed size during iteration". list() is atomic under the
    # GIL, mirroring the list(cache.keys()) idiom used elsewhere.
    return [StoredImageEval(**v) for v in list(_eval_cache_for(state).values())]


@router.get("/bank/images/evaluate", response_model=StoredImageEval)
def evaluate_bank_image(
    tier: Tier,
    name: str,
    label: str = "",
    group_mode: str = "none",
    group_sep: str = "_",
    group_fields: int = 1,
) -> StoredImageEval:
    """Image-level anomaly stats for one taught image, from its stored rows.

    Scores the patch rows this image contributed to the bank against the
    normal bank (top-k mean distance; leave-own-image-out for normal-tier
    images) — no model forward, so a client can sweep the whole bank image
    by image and build an OK/NG score-separation histogram cheaply.
    """
    from clscore.scoring import score_stored_features

    state = get_state()
    state.require_active()
    bank = state.bank
    if int(bank.normal.shape[0]) < 2:
        raise HTTPException(status_code=400, detail="normal bank is empty — teach OK images first")
    # Fingerprint of the bank these scores are ABOUT to be computed against.
    # A teach can mutate the bank while we score below (this route holds no
    # lock); stamping the mirror with a save-time fingerprint would then
    # persist pre-teach numbers as if they were fresh (silent corruption of
    # the separation check, exemplars and live alpha). _eval_cache_save
    # drops the write when the fingerprint no longer matches.
    fp_at_score = _eval_fingerprint(bank)

    if tier == "normal":
        index = bank.meta.normal_image_index
        feats_src = bank.normal
    elif tier == "critical":
        index = bank.meta.critical_image_index.get(label, [])
        feats_src = bank.critical.get(label)
    else:
        index = bank.meta.negative_image_index.get(label, [])
        feats_src = bank.negative.get(label)

    entry = next((e for e in index if str(e.get("name", "")) == name), None)
    if entry is None or feats_src is None or not feats_src.size:
        raise HTTPException(status_code=404, detail="image has no indexed rows in this bank")
    start, count = int(entry.get("start", -1)), int(entry.get("count", 0))
    if start < 0 or count <= 0 or start + count > int(feats_src.shape[0]):
        raise HTTPException(status_code=409, detail="row index out of range — legacy bank entry")

    # Leave-own-GROUP-out. A lot photographed in sequence leaves near
    # duplicates of the query in the bank when only its own rows are masked,
    # so the whole group goes out together. Normal-tier only: the labelled
    # tiers are not in the bank being searched, so nothing of theirs to mask.
    excl_ranges: list[tuple[int, int]] = []
    grouped = tier == "normal" and group_mode != "none"
    if grouped:
        from clscore.grouping import GROUP_MODES, derive_groups, exclusion_ranges

        from ..core.cls_store import load_store
        if group_mode not in GROUP_MODES:
            raise HTTPException(status_code=422, detail=f"unknown group mode: {group_mode}")
        manual = (
            {e.name: e.group for e in load_store(state).entries if e.group}
            if group_mode == "manual" else {}
        )
        groups = derive_groups(
            [str(e.get("name", "")) for e in index],
            group_mode, sep=group_sep, fields=group_fields, manual=manual,
        )
        excl_ranges = exclusion_ranges(index, groups, name)

    cache = _eval_cache_for(state)
    cache_key = f"{tier}/{label}/{name}"
    cached = cache.get(cache_key)
    # Entries written before top_indices existed are treated as misses so a
    # re-run upgrades them in place instead of pinning the old shape forever.
    # The patches==count guard defends the delete-then-reteach-same-name
    # case: labelled-tier edits no longer roll the whole-cache fingerprint,
    # so a stale same-name entry must at least match the stored row count.
    # A grouped run is deliberately NOT served from (or written to) the
    # cache: it is keyed by image and is also read by the exemplar and
    # projection paths, so a second score for the same image under a
    # different validation rule would make those disagree with the histogram.
    if (
        not grouped
        and cached is not None
        and cached.get("top_indices")
        and int(cached.get("patches", -1)) == count
    ):
        return StoredImageEval(**cached)

    ivf, ivf_nprobe = state.get_normal_ivf()
    if ivf is not None and ivf.has_storage:
        bank_t, n_rows_bank = None, ivf.n_rows
    else:
        bank_t = state.get_normal_tensor()
        n_rows_bank = int(bank_t.shape[0])
    scores = score_stored_features(
        np.ascontiguousarray(feats_src[start : start + count], dtype=np.float32),
        bank_t,
        exclude_start=start if tier == "normal" and not grouped else -1,
        exclude_count=count if tier == "normal" and not grouped else 0,
        exclude_ranges=excl_ranges or None,
        cdist_chunk=state.cdist_chunk_for(n_rows_bank, ivf_active=ivf is not None),
        ivf=ivf,
        ivf_nprobe=ivf_nprobe or 8,
    )
    order = np.argsort(scores)[::-1][: min(256, int(scores.size))]
    result = StoredImageEval(
        name=name,
        tier=tier,
        label=label,
        patches=int(count),
        score_max=float(scores.max()),
        score_p99=float(np.percentile(scores, 99)),
        score_mean=float(scores.mean()),
        top_scores=[float(scores[i]) for i in order],
        top_indices=[int(i) for i in order],
    )
    if not grouped:
        cache[cache_key] = result.model_dump()
        _eval_cache_save(state, cache, fingerprint=fp_at_score)
    return result


# ---- runtime package: verdict config + one-file bank export/import ---------
# "Export like a trained model": the bank directory already holds everything a
# runtime needs (feature arrays, exemplar-severity sidecars, eval cache); the
# runtime config adds the verdict recipe (metric / k / alpha / threshold), so
# one zip of that directory is a complete, loadable inspection package.

RUNTIME_CONFIG_FILE = "runtime_config.json"
BANK_EXPORT_SUFFIX = ".clsbank.zip"


@router.get("/bank/runtime-config", response_model=RuntimeConfig | None)
def get_runtime_config() -> RuntimeConfig | None:
    """The active bank's saved verdict recipe, or null if never saved."""
    state = get_state()
    state.require_active()
    assert state.bank_dir is not None
    path = state.bank_dir / RUNTIME_CONFIG_FILE
    if not path.exists():
        return None
    try:
        cfg = RuntimeConfig(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    # Stale = the bank's contents (teaches / deletes / marks) changed since
    # this recipe was saved — the Operator shows a re-check warning. Configs
    # from before fingerprint stamping count as stale too.
    cfg.stale = cfg.bank_fingerprint != _bank_content_fingerprint(state.bank)
    return cfg


@router.put("/bank/runtime-config", response_model=RuntimeConfig)
def put_runtime_config(
    body: RuntimeConfig,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> RuntimeConfig:
    """Persist the verdict recipe into the bank directory (rides along in exports)."""
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    assert state.bank_dir is not None
    body.saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body.bank_fingerprint = _bank_content_fingerprint(state.bank)
    body.stale = False
    write_bytes_atomic(
        state.bank_dir / RUNTIME_CONFIG_FILE, body.model_dump_json(indent=2).encode("utf-8")
    )
    return body


# ---- runtime memory-bank budget (small / medium / large) ------------------
# The normal tensor is GPU-resident for scoring, so its total patch count
# (fp16 x DINOv2 dim => ~1.5 KB/patch) sets the resident VRAM floor. These
# ceilings cap that at teach time — append-only, so lowering a tier only
# bounds FUTURE growth; an already-oversized bank keeps its rows until an
# explicit reduce. Override any tier via env for site-specific VRAM budgets.
_DEFAULT_CAPACITY = "medium"
_DEFAULT_CAPACITY_CEILINGS: dict[str, int] = {
    "small": 350_000,     # ~0.5 GB resident (fp16, 768-dim)
    "medium": 1_400_000,  # ~2 GB
    "large": 4_000_000,   # ~6 GB
}
_CAPACITY_ENV = {"small": "CLS_CAPACITY_SMALL", "medium": "CLS_CAPACITY_MEDIUM", "large": "CLS_CAPACITY_LARGE"}


def _capacity_ceilings() -> dict[str, int]:
    out = dict(_DEFAULT_CAPACITY_CEILINGS)
    for tier, env in _CAPACITY_ENV.items():
        raw = os.environ.get(env)
        if raw is not None:
            try:
                out[tier] = max(0, int(raw))
            except ValueError:
                pass
    return out


def _bank_capacity(state: ClsStudioState) -> str:
    """The active bank's saved capacity tier, defaulting to medium."""
    if state.bank_dir is None:
        return _DEFAULT_CAPACITY
    path = state.bank_dir / RUNTIME_CONFIG_FILE
    if not path.exists():
        return _DEFAULT_CAPACITY
    try:
        cap = json.loads(path.read_text(encoding="utf-8")).get("bank_capacity")
    except (OSError, ValueError):
        return _DEFAULT_CAPACITY
    return cap if cap in _DEFAULT_CAPACITY_CEILINGS else _DEFAULT_CAPACITY


def _capacity_ceiling(state: ClsStudioState) -> int:
    """Total normal-patch ceiling for the active bank (0 = unbounded)."""
    return _capacity_ceilings().get(_bank_capacity(state), 0)


def _capacity_info(state: ClsStudioState) -> BankCapacityInfo:
    cap = _bank_capacity(state)
    ceiling = _capacity_ceilings().get(cap, 0)
    normal = int(state.bank.normal.shape[0]) if state.bank.normal.size else 0
    dim = int(state.bank.meta.dim) or 768
    pct = min(100.0, round(normal / ceiling * 100, 1)) if ceiling else 0.0
    est_vram_mb = round(normal * dim * 2 / (1024 ** 2))  # fp16 => 2 bytes/elem
    # The labelled tiers are resident alongside the normal one but are neither
    # per-image capped nor counted against the ceiling, so a gauge built only
    # from `normal` under-reports what the bank actually costs — on this dev
    # box one project carries 1.25M critical patches (~1.9 GB) while the bar
    # sits near zero. Reported separately rather than folded into pct: the
    # ceiling really does only govern the normal tier, and a bar that filled
    # up from rows no budget controls would be its own kind of lie.
    labeled = int(state.bank.tier_size("critical")) + int(state.bank.tier_size("negative"))
    est_vram_total_mb = round((normal + labeled) * dim * 2 / (1024 ** 2))
    return BankCapacityInfo(
        capacity=cap, ceiling=ceiling, normal=normal, pct=pct, est_vram_mb=est_vram_mb,
        labeled=labeled, est_vram_total_mb=est_vram_total_mb,
    )


@router.get("/bank/capacity", response_model=BankCapacityInfo)
def get_bank_capacity() -> BankCapacityInfo:
    """The active bank's size budget and how full the normal tier is."""
    state = get_state()
    state.require_active()
    return _capacity_info(state)


@router.put("/bank/capacity", response_model=BankCapacityInfo)
def put_bank_capacity(body: BankCapacitySet) -> BankCapacityInfo:
    """Set the active bank's size budget (small/medium/large).

    Read-modify-writes only ``bank_capacity`` in runtime_config.json — the
    verdict recipe and its staleness stamp are left untouched, since the
    budget is an operational knob, not part of the tuned recipe. Takes effect
    on the next teach; existing rows are never evicted.
    """
    state = get_state()
    state.require_active()
    assert state.bank_dir is not None
    path = state.bank_dir / RUNTIME_CONFIG_FILE
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
    else:
        raw = {}
    raw["bank_capacity"] = body.capacity
    # Round-trip through the model so a fresh file is schema-valid, but do NOT
    # restamp saved_at/bank_fingerprint/stale here.
    cfg = RuntimeConfig(**raw)
    write_bytes_atomic(path, cfg.model_dump_json(indent=2).encode("utf-8"))
    return _capacity_info(state)


@router.get("/bank/export")
def export_bank(include_images: bool = True, include_store: bool = True) -> FileResponse:
    """Download the active bank as one zip package.

    Contains the feature arrays, per-row severity sidecars (defect marks),
    the eval cache (feeds auto-exemplars), the runtime config if saved, and
    (unless ``include_images=false``) the taught source images. STORED, not
    deflated — float32 features barely compress and the archive can run to
    several GB, so skipping compression keeps export time disk-bound.

    ``include_store`` carries the feature store, which is what lets the
    receiving machine re-label without re-extracting — the whole point of
    the split. It roughly doubles the archive, because the store holds the
    same features the tier arrays were assembled from, so a deployment that
    only needs to *score* can turn it off. Label sets ride along either way:
    they are a few KB and dropping them would lose the judgement itself.
    """
    state = get_state()
    state.require_active()
    assert state.bank_dir is not None
    bank_dir = state.bank_dir
    if not (bank_dir / "bank.npy").exists():
        raise HTTPException(status_code=400, detail="bank is empty — nothing to export")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    try:
        # io_lock too: teach saves the feature arrays under io_lock only, so
        # zipping under state.lock alone could package a post-teach bank.npy
        # next to a pre-teach .meta.npz — an archive that 422s on import.
        with state.lock, state.io_lock, zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(bank_dir.rglob("*")):
                if p.is_dir():
                    continue
                rel = p.relative_to(bank_dir)
                if not include_images and rel.parts and rel.parts[0] == "_images":
                    continue
                if not include_store and rel.parts and rel.parts[0] == STORE_SUBDIR:
                    continue
                # Display renditions are derived and regenerate on demand;
                # shipping them would inflate the package for nothing.
                if rel.parts[:2] == (STORE_SUBDIR, "cache"):
                    continue
                if p.suffix == ".tmp":
                    continue  # in-flight atomic-write leftovers, never data
                zf.write(p, rel.as_posix())
    except BaseException:
        os.unlink(tmp.name)
        raise
    fname = f"{state.active_bank_id or 'bank'}{BANK_EXPORT_SUFFIX}"
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename=fname,
        background=BackgroundTask(os.unlink, tmp.name),
    )


@router.get("/bank/export/edge")
def export_bank_for_edge() -> FileResponse:
    """Download the active bank as an on-device inspection package.

    Unlike ``/bank/export`` this is not a bank you can re-import here: it is
    what a phone or embedded box needs to score images by itself. The normal
    rows ship already quantised to int8, the IVF centroids ride along so the
    device narrows to the same candidates this server would, and the manifest
    names the encoder the features came from. No source images, no fp16
    arrays - typically under half the size of a full export.
    """
    state = get_state()
    state.require_active()
    assert state.bank_dir is not None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    try:
        # Both locks, for the same reason the full export takes both: teach
        # writes the feature arrays under io_lock, so state.lock alone could
        # package a half-updated bank.
        with state.lock, state.io_lock:
            comp = read_compression_settings()
            build_edge_package(
                bank=state.bank,
                bank_dir=state.bank_dir,
                bank_id=state.active_bank_id or "bank",
                dest=Path(tmp.name),
                runtime_config_file=RUNTIME_CONFIG_FILE,
                exemplars=_exemplar_rows(state.bank, _eval_cache_for(state)),
                ivf_index_file=IVF_INDEX_FILE if comp.get("ivf") else None,
                nprobe=comp.get("ivf_nprobe"),
            )
    except ValueError as exc:
        os.unlink(tmp.name)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BaseException:
        os.unlink(tmp.name)
        raise
    fname = f"{state.active_bank_id or 'bank'}{EDGE_EXPORT_SUFFIX}"
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename=fname,
        background=BackgroundTask(os.unlink, tmp.name),
    )


@router.post("/banks/import", response_model=SelectResult)
async def import_bank(
    archive: UploadFile = File(...),
    name: str = Form(""),
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> SelectResult:
    """Load an exported bank package as a new bank of the active project.

    The archive is validated (bank files at the root, no absolute /
    parent-escaping paths), extracted into a fresh bank directory, and
    activated — ``Bank.load``'s integrity checks run as part of activation,
    so a corrupt package is rejected and the directory removed rather than
    left half-imported. Returns the same shape as bank selection, with the
    imported bank active and ready to inspect.
    """
    state = get_state()
    state.require_active_project()
    assert state.active_project_id is not None
    project_id = state.active_project_id
    # Import creates a NEW bank, so only the project half of the binding is
    # checked — the client's bound bank id is allowed to differ.
    if binding and binding.split("/", 1)[0] != project_id:
        raise HTTPException(
            status_code=409,
            detail="active project changed — re-select and retry",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    bank_dir = None
    try:
        total = 0
        with open(tmp.name, "wb") as out:
            while chunk := await archive.read(8 * 1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file too large (max {MAX_ARCHIVE_BYTES // (1024*1024)} MB)",
                    )
                out.write(chunk)

        def _extract_and_activate() -> None:
            nonlocal bank_dir
            with zipfile.ZipFile(tmp.name) as zf:
                names = zf.namelist()
                if "bank.npy" not in names or "bank_meta.json" not in names:
                    raise HTTPException(
                        status_code=422,
                        detail="not a bank package — bank.npy / bank_meta.json missing at archive root",
                    )
                # Zip-bomb guard, as a RATIO rather than an absolute size: the
                # cap is now generous enough that "4x the cap" would no longer
                # catch anything. Our own packages are written STORED, so a
                # legitimate one expands ~1x; 8x still leaves room for an
                # archive someone recompressed on the way.
                declared_total = sum(info.file_size for info in zf.infolist())
                if declared_total > max(8 * total, 64 * 1024 * 1024):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"archive expands to {declared_total // (1024*1024)} MB "
                            f"from {total // (1024*1024)} MB — refusing to extract"
                        ),
                    )
                for n in names:
                    pp = PurePosixPath(n)
                    if pp.is_absolute() or ".." in pp.parts or (pp.parts and ":" in pp.parts[0]):
                        raise HTTPException(status_code=422, detail=f"unsafe path in archive: {n}")
                base = (name or archive.filename or "imported").rsplit("/", 1)[-1]
                base = re.sub(r"(\.clsbank)?\.zip$", "", base, flags=re.IGNORECASE) or "imported"
                with state.lock:
                    # Re-verify under the lock: a project delete racing this
                    # multi-GB upload must not have the extraction resurrect
                    # the removed tree (the startup purge would destroy it).
                    state.require_active_project()
                    root = state._banks_root(project_id)
                    candidate, i = slug_bank_id(base), 1
                    while (root / candidate).is_dir():
                        i += 1
                        candidate = f"{slug_bank_id(base)}-{i}"
                    bank_dir = root / candidate
                    bank_dir.mkdir(parents=True, exist_ok=True)
                    zf.extractall(bank_dir)
                    state.activate(project_id, candidate)
                    # Display name follows the package, not the source bank's
                    # old description (usually just "Default").
                    state.bank.meta.description = base
                    (bank_dir / "bank_meta.json").write_text(
                        state.bank.meta.to_json(), encoding="utf-8"
                    )

        await run_in_threadpool(_extract_and_activate)
    except BaseException:
        if bank_dir is not None:
            shutil.rmtree(bank_dir, ignore_errors=True)
        raise
    finally:
        os.unlink(tmp.name)
    if state.active_project_id:
        touch_project(state.active_project_id)
    return _select_result(state)


class AnnotateRequest(BaseModel):
    tier: Tier
    label: str = ""
    name: str
    rects: list[AnnotationRect] = []


@router.post("/bank/images/annotate", response_model=AnnotateResult)
def annotate_bank_image(
    body: AnnotateRequest,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> AnnotateResult:
    """Mark an NG image's defect regions; the covered bank rows become exemplars.

    The rectangles (normalized to the source image) are mapped through the
    SW geometry to the exact rows this image contributed, and those rows'
    severity is set to heavy — the separation check's alpha term and the
    weighted scoring path both key off it. Sending an empty ``rects`` list
    clears the annotation. Only the tier metadata sidecars are rewritten,
    not the multi-GB feature arrays.
    """
    from clscore.sw import expected_rows, rows_for_rects

    state = get_state()
    state.require_active()
    check_binding(state, binding)
    bank = state.bank
    if body.tier != "critical":
        raise HTTPException(status_code=422, detail="defect marks only apply to the critical tier")

    path = state.image_path(body.tier, body.name)
    img = cv2.imread(str(path)) if path.is_file() else None
    if img is None:
        raise HTTPException(status_code=404, detail="source image not found — cannot resolve geometry")
    h, w = img.shape[:2]
    geo = {"window_size": int(bank.meta.window), "stride": int(bank.meta.stride), "patch": int(bank.meta.patch)}

    # ``label`` is the resolved on-disk label from the images listing — used
    # verbatim (safe_label would strip "_default" down to a nonexistent key).
    index = bank.meta.critical_image_index.get(body.label or DEFAULT_LABEL, [])
    entry = next((e for e in index if str(e.get("name", "")) == body.name), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="image has no indexed rows in this bank")
    grid_rows = expected_rows(h, w, **geo)
    kept = entry.get("kept")
    if kept:
        # A coreset-reduced image legitimately holds fewer rows than its patch
        # grid. What must still line up is the map: one entry per stored row,
        # every one of them naming a real patch of this geometry.
        if len(kept) != int(entry.get("count", 0)) or max(int(v) for v in kept) >= grid_rows:
            raise HTTPException(
                status_code=409,
                detail="stored kept-row map does not match the image's SW geometry",
            )
    elif int(entry.get("count", 0)) != grid_rows:
        raise HTTPException(
            status_code=409,
            detail="stored row count does not match the image's SW geometry — legacy bank entry",
        )

    rows = rows_for_rects(h, w, [(r.x, r.y, r.w, r.h) for r in body.rects], **geo)
    with state.lock:
        try:
            n = bank.set_image_annotation(
                body.tier, body.label, body.name, rows,
                rects=[r.model_dump() for r in body.rects],
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        assert state.bank_dir is not None
        bank.save_meta_only(state.bank_dir, body.tier)
    return AnnotateResult(
        tier=body.tier, label=body.label, name=body.name,
        rows_marked=n, rects=body.rects,
    )


@router.get("/bank/evaluation/cmin", response_model=list[ImageCmin])
def evaluation_cmin(auto_top: int = 10) -> list[ImageCmin]:
    """Min distance to the defect-exemplar rows, per cached eval's top patches.

    The exemplar set is built from the critical tier: an annotated image
    contributes its heavy-severity rows, an unannotated one falls back to
    its cached top-``auto_top`` rows (skipped if it was never evaluated).
    For every cached eval we then return, aligned with its ``top_indices``,
    the min distance from each of those patches to the exemplar set — the
    image's own exemplar rows excluded, mirroring the leave-own-image-out
    raw scores. The client composes ``score + alpha / (eps + cmin)`` and
    sweeps alpha for free. Recomputed per call (one small cdist); nothing
    is cached, so mark edits take effect immediately.
    """
    import torch

    state = get_state()
    state.require_active()
    bank = state.bank
    # Snapshot the shared eval cache: it may be mutated by a concurrent
    # request while we iterate below (single global state), which raised
    # "dictionary changed size during iteration". dict() is atomic under
    # the GIL; _exemplar_rows only reads via .get so the copy is enough.
    cache = dict(_eval_cache_for(state))
    if not cache:
        return []

    # Device anchor without materialising the normal tensor (kept off the
    # GPU entirely when IVF resident storage is active). All row gathering
    # below goes through the numpy arrays so the full critical / negative
    # tiers never get materialised on the GPU either.
    _, device, _dtype = state.ensure_model()

    # ---- exemplar set: annotated rows, else cached auto-top rows ----------
    ex_parts: list[np.ndarray] = []
    ex_src: list[str] = []
    for label, rows in _exemplar_rows(bank, cache, auto_top).items():
        arr = bank.critical.get(label)
        if arr is None:
            continue
        idx = np.asarray([r for r, _ in rows], dtype=np.int64)
        ex_parts.append(np.ascontiguousarray(arr[idx], dtype=np.float32))
        ex_src.extend(k for _, k in rows)
    if not ex_parts:
        return []
    exemplars = torch.from_numpy(np.concatenate(ex_parts, axis=0)).to(device)
    src_arr = np.array(ex_src)

    # ---- per cached eval: min distance of its top rows to the exemplars ---
    out: list[ImageCmin] = []
    for key, ent in cache.items():
        tis = ent.get("top_indices") or []
        tier, label, name = str(ent.get("tier", "")), str(ent.get("label", "")), str(ent.get("name", ""))
        if not tis or tier not in ("normal", "critical", "negative"):
            continue
        if tier == "normal":
            feats, index = bank.normal, bank.meta.normal_image_index
        elif tier == "critical":
            feats = bank.critical.get(label)
            index = bank.meta.critical_image_index.get(label, [])
        else:
            feats = bank.negative.get(label)
            index = bank.meta.negative_image_index.get(label, [])
        entry = next((e for e in index if str(e.get("name", "")) == name), None)
        if entry is None or feats is None:
            continue
        start, count = int(entry.get("start", -1)), int(entry.get("count", 0))
        rows = [start + int(li) for li in tis if 0 <= int(li) < count]
        if start < 0 or len(rows) != len(tis) or start + count > int(feats.shape[0]):
            continue
        q = torch.from_numpy(
            np.ascontiguousarray(feats[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
        ).to(device)
        d = torch.cdist(q, exemplars)
        own = src_arr == key
        if own.all():
            continue  # nothing left after leave-own-image-out
        if own.any():
            d[:, torch.from_numpy(own).to(d.device)] = float("inf")
        cm = d.min(dim=1).values.cpu().numpy()
        out.append(ImageCmin(name=name, tier=tier, label=label, top_cmin=[float(v) for v in cm]))
    return out


@router.post("/bank/save", response_model=BankState)
def save_bank(
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> BankState:
    state = get_state()
    # Was missing entirely: with no (or a deleted) active project this
    # route reached state.save_bank() with bank_dir=None and 500'd —
    # and a save racing a project delete could resurrect the removed tree.
    state.require_active()
    check_binding(state, binding)
    with state.lock:
        state.save_bank()
        return bank_state_of(state.bank)


@router.post("/bank/reload", response_model=BankState)
def reload_bank() -> BankState:
    """Discard in-memory edits and reload the bank from disk."""
    state = get_state()
    state.require_active()
    with state.lock:
        assert state.active_project_id is not None
        state.activate(state.active_project_id, state.active_bank_id)
        return bank_state_of(state.bank)


@router.post("/bank/clear/{tier}", response_model=BankState)
def clear_tier(
    tier: Tier,
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> BankState:
    state = get_state()
    state.require_active()
    check_binding(state, binding)
    if tier == "normal":
        # Bank.clear_tier raises a bare ValueError for the protected normal
        # tier, which surfaced as a 500 — map it to a proper client error.
        raise HTTPException(status_code=422, detail="normal tier is protected")
    with state.lock:
        # Collect names before clearing so the eval-cache purge can drop
        # exactly this tier's entries (labelled tiers don't roll the
        # fingerprint anymore).
        if tier == "critical":
            names = [n for ns in state.bank.meta.critical_images.values() for n in ns]
        elif tier == "negative":
            names = [n for ns in state.bank.meta.negative_images.values() for n in ns]
        else:
            names = []
        state.bank.clear_tier(tier)
        state.mark_dirty()
        state.save_bank()
        # Drop the tier's saved source images too: leaving them behind kept
        # has_bank_thumbnail alive on the project card, shipped orphans in
        # every export, and forced _1 suffixes on re-teach of the same names.
        img_dir = state.images_dir(tier)
        if img_dir.is_dir():
            for f in img_dir.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                except OSError:
                    pass  # a browser may still stream one — orphan, not fatal
        _eval_cache_purge(state, tier, None, names)
        if state.active_project_id:
            touch_project(state.active_project_id)
        return bank_state_of(state.bank)


@router.post("/bank/append/{tier}", response_model=AppendResult)
async def append_image(
    tier: Tier,
    image: UploadFile = File(...),
    label: str = Form(""),
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> AppendResult:
    """Extract one image's patch tokens and add them to ``tier`` (HITL teach).

    The raw image is also saved under ``<bank>/_images/<tier>/`` so the UI can
    show a thumbnail; its saved filename is what the bank records, so the two
    always line up. This is the only bank route that loads DINOv2.
    """
    state = get_state()
    state.require_active()
    check_binding(state, binding)

    data = await read_upload(image)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(status_code=422, detail="could not decode uploaded image")

    def _teach() -> tuple[str, int]:
        # Identity snapshot: extraction below runs for seconds without the
        # lock, and a concurrent /bank/select (another client) can swap the
        # active bank meanwhile — the append must then abort, not land the
        # rows in whichever bank happens to be active.
        bound_bank, bound_dir = state.bank, state.bank_dir
        # Runtime bank budget: reject up front when the normal tier is already
        # at its ceiling, so we never save an orphan source image with no rows.
        ceiling = _capacity_ceiling(state) if tier == "normal" else 0
        if ceiling and int(state.bank.normal.shape[0]) >= ceiling:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"normal bank at '{_bank_capacity(state)}' capacity "
                    f"({ceiling:,} patches) — raise the limit or reduce the bank"
                ),
            )
        saved_name = state.save_source_image(tier, image.filename or "image.png", data)
        model, device, _dtype = state.ensure_model()
        features = extract_image_features_for_bank(model, arr, device, max_batch=state.max_batch())
        # Per-image patch cap: keeps a huge library from growing unbounded in
        # RAM/disk/VRAM. 0 disables. Applies to the labelled tiers too now —
        # they were exempt because "their exact rows back the alpha exemplars /
        # NG marks", and that is still true, so a reduced image records which
        # grid patches it kept and the marks map through it (2026-07-31).
        cap = _max_patches_per_image()
        with state.lock:
            if state.bank is not bound_bank or state.bank_dir != bound_dir:
                raise HTTPException(
                    status_code=409,
                    detail="active bank changed during teach — nothing was added, re-select and retry",
                )
            before = int(state.bank.normal.shape[0]) if tier == "normal" else 0
            # Tighten the per-image cap so this image cannot push the normal
            # bank past its runtime ceiling; append coreset-reduces to eff_cap.
            eff_cap = cap
            if ceiling:
                headroom = ceiling - before
                if headroom <= 0:  # filled by a concurrent teach since the pre-check
                    return (label or ""), 0
                eff_cap = headroom if cap == 0 else min(cap, headroom)
            resolved = state.bank.append(
                tier, features, label=label or None, image_name=saved_name,
                max_patches=eff_cap, device=device,
            )
            # The cap may have reduced the row count; report what actually
            # landed, which for a labelled tier is now also capped.
            n_added = (
                (int(state.bank.normal.shape[0]) - before) if tier == "normal"
                else min(int(features.shape[0]), eff_cap) if eff_cap > 0
                else int(features.shape[0])
            )
            state.mark_dirty()
        # Disk write OUTSIDE state.lock (io_lock-serialised): a multi-GB
        # bank.npy write must not stall concurrent scoring. parts=(tier,)
        # skips rewriting the tiers this teach didn't touch.
        if state.bank is not bound_bank:
            raise HTTPException(
                status_code=409,
                detail="active bank changed during teach — the appended rows were discarded, re-select and retry",
            )
        state.save_bank(parts=(tier,))
        return resolved, n_added

    resolved, n_patches = await run_in_threadpool(_teach)
    if state.active_project_id:
        touch_project(state.active_project_id)
    return AppendResult(
        tier=tier,
        label=resolved,
        appended_patches=n_patches,
        bank=bank_state_of(state.bank),
    )


# Images per batched forward group. Bounds the transient window memory
# (~24 imgs x ~6 windows x 518^2 fp16 ≈ 0.5 GB) while still packing full
# max_batch forwards so the GPU stays busy across the whole teach.
_BATCH_TEACH_GROUP = 24


def _teach_decoded(
    state: ClsStudioState,
    tier: Tier,
    decoded: list[tuple[str, bytes, np.ndarray]],
    label: str,
) -> tuple[str, int, list[str]]:
    """Teach core shared by ``append_batch`` and the staged teach.

    Batched forward per group, per-image coreset cap, capacity ceiling and
    the identity snapshot (a concurrent re-select must abort, not scatter
    groups across two banks). Saves PER GROUP, not once at the end: a crash
    mid-batch then loses at most one group's rows (2026-07-17: a native
    crash before the single end-of-batch save orphaned all 24 images).

    Returns ``(resolved_label, total_added_rows, taught_input_names)`` —
    an input name absent from the returned list did NOT reach the bank
    (capacity stop, aborted group) and must stay wherever it came from.
    """
    bound_bank, bound_dir = state.bank, state.bank_dir
    model, device, _dtype = state.ensure_model()
    cap = _max_patches_per_image()
    ceiling = _capacity_ceiling(state) if tier == "normal" else 0
    total_added = 0
    resolved = ""
    taught: list[str] = []
    for i in range(0, len(decoded), _BATCH_TEACH_GROUP):
        # Runtime bank budget: stop before extracting/saving a whole group
        # once the normal tier is already full (avoids orphan thumbnails).
        if ceiling and int(state.bank.normal.shape[0]) >= ceiling:
            break
        grp = decoded[i : i + _BATCH_TEACH_GROUP]
        arrs = [a for (_n, _d, a) in grp]
        # Heavy work OFF the lock: one batched forward for the group, then
        # per-image coreset cap, then persist thumbnails.
        # Stream, don't materialise. The group's full features are ~673 MB per
        # 24 MP image and the coreset cap immediately reduces each one to a few
        # MB, so collecting them all first was tens of GB of host memory held
        # for no reason — measured at +6.2 GB for a single 24 MP teach on the
        # dev box (2026-07-31). Consuming per image keeps one image's features
        # live at a time; the forwards and their batching are unchanged.
        # `kept` travels with each image: for a labelled tier it is what lets
        # the NG marks, which address the patch grid, still find their rows
        # after the reduction. None means the image kept every patch.
        capped: list[tuple[np.ndarray, np.ndarray | None]] = []
        for _i, feats in iter_images_features_batched(model, arrs, device, max_batch=state.max_batch()):
            kept: np.ndarray | None = None
            if cap > 0 and feats.shape[0] > cap:
                feats, kept = coreset_reduce_indexed(feats, cap / float(feats.shape[0]), device)
            capped.append((feats, kept))
        saved_names = [state.save_source_image(tier, fname, data) for (fname, data, _a) in grp]
        # Only the in-memory concatenation needs the lock — kept short so
        # concurrent scoring isn't blocked. Cap already applied above.
        with state.lock:
            if state.bank is not bound_bank or state.bank_dir != bound_dir:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "active bank changed during batch teach — "
                        f"{total_added} rows from earlier groups were saved, "
                        "the rest was aborted; re-select and re-teach the remainder"
                    ),
                )
            for (orig_name, _data, _a), (feats, kept), saved in zip(grp, capped, saved_names):
                before = int(state.bank.normal.shape[0]) if tier == "normal" else 0
                if ceiling:
                    headroom = ceiling - before
                    if headroom <= 0:
                        break  # bank hit its budget — skip the rest of this group
                    if feats.shape[0] > headroom:
                        # Truncate the per-image coreset subset to fit: its
                        # greedy farthest-point order keeps the prefix diverse.
                        # The kept-index map has to be cut in lockstep or it
                        # stops describing the rows that landed.
                        feats = feats[:headroom]
                        if kept is not None:
                            kept = kept[:headroom]
                resolved = state.bank.append(
                    tier, feats, label=label or None, image_name=saved,
                    max_patches=0, device=device, kept_idx=kept,
                )
                total_added += (int(state.bank.normal.shape[0]) - before) if tier == "normal" else int(feats.shape[0])
                taught.append(orig_name)
            state.mark_dirty()
        # Under io_lock, so concurrent scoring is never blocked by it.
        state.save_bank(parts=(tier,))
    return resolved, total_added, taught


@router.post("/bank/append_batch/{tier}", response_model=AppendResult)
async def append_batch(
    tier: Tier,
    images: list[UploadFile] = File(...),
    label: str = Form(""),
    binding: str | None = Header(None, alias="X-Bank-Binding"),
) -> AppendResult:
    """Teach many images (same tier/label) in one request.

    Per-image teaching runs a tiny ~6-window forward then stalls on CPU/disk,
    so the GPU never fills. This collects every image's sliding windows and
    runs them through the backbone in full ``max_batch`` forwards, then saves
    the bank once. Feature-identical to teaching each image via
    ``/bank/append/{tier}`` — only the forward is shared.
    """
    state = get_state()
    state.require_active()
    check_binding(state, binding)

    # Per-file read_upload only bounds ONE part; without the aggregate check a
    # single request can hold N x 200 MB of bytes + decoded arrays in RAM. The
    # cap budgets the uploaded bytes only — the arrays decoded from them below
    # scale with the batch but are not counted (see app.core.config).
    check_upload_batch(len(images), 0)
    decoded: list[tuple[str, bytes, np.ndarray]] = []
    total_bytes = 0
    for up in images:
        data = await read_upload(up)
        total_bytes += len(data)
        check_upload_batch(len(images), total_bytes)
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            decoded.append((up.filename or "image.png", data, arr))
    if not decoded:
        raise HTTPException(status_code=422, detail="no decodable images in batch")

    def _teach_batch() -> tuple[str, int, list[str]]:
        return _teach_decoded(state, tier, decoded, label)

    resolved, n_patches, _taught = await run_in_threadpool(_teach_batch)
    if state.active_project_id:
        touch_project(state.active_project_id)
    return AppendResult(
        tier=tier,
        label=resolved,
        appended_patches=n_patches,
        bank=bank_state_of(state.bank),
    )
