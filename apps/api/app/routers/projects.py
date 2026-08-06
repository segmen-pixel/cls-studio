# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from starlette.background import BackgroundTask

_logger = logging.getLogger(__name__)

from pydantic import BaseModel as _BaseModel

from ..models import ModelRecord, Project, TrainingRun
from ..schemas import ProjectCreate, ProjectRead, ProjectUpdate


class _ReorderPayload(_BaseModel):
    order: list[str]  # list of project IDs in desired order
from ..core.annotate_index import load_annotate_index
from ..core.config import MAX_ARCHIVE_BYTES
from ..core.db_utils import default_classes_payload, log_action
from ..core.paths import (
    LAYOUT_VERSION,
    annotate_images_dir,
    annotate_masks_dir,
    classes_path,
    ensure_project_dirs,
    new_project_id,
    project_dir,
    write_json,
    write_project_json,
)
from ..core.state import RUN_FLAGS
from ..db import get_engine

router = APIRouter()


@router.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate) -> ProjectRead:
    """Create a new project.

    Allocates a new id, lays down the on-disk project directory with a
    default ``classes.json`` and ``project.json``, then records the project
    in the database. If any step fails, the partial on-disk directory is
    removed so the next startup orphan-adopt does not resurrect a stub.

    Raises:
        Exception: If the on-disk layout or database insert fails. The
            partial project directory is cleaned up before the exception
            propagates.
    """
    project_id = new_project_id()
    now = datetime.now(timezone.utc)
    project = Project(
        id=project_id, name=payload.name, description=payload.description,
        memo=payload.memo, tags=json.dumps(payload.tags or [], ensure_ascii=False),
        created_at=now, updated_at=now,
    )
    # Lay down the on-disk structure first so the DB never references an
    # incomplete project. If anything fails, tear down the partial dir so the
    # startup orphan-adopt doesn't resurrect a stub on the next boot.
    try:
        ensure_project_dirs(project_id)
        classes_path(project_id).write_text(
            json.dumps(default_classes_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        serialized = ProjectRead.model_validate(project).model_dump(mode="json")
        serialized["schema_version"] = LAYOUT_VERSION
        write_json(project_dir(project_id) / "project.json", serialized)
    except Exception:
        shutil.rmtree(project_dir(project_id), ignore_errors=True)
        raise
    engine = get_engine()
    try:
        with Session(engine) as session:
            session.add(project)
            log_action(session, "project_create", "project", project_id)
            session.commit()
            session.refresh(project)
            _invalidate_projects_summary_cache()
            return ProjectRead.model_validate(project)
    except Exception:
        shutil.rmtree(project_dir(project_id), ignore_errors=True)
        raise


PROJECT_EXPORT_SUFFIX = ".clsproj.zip"


@router.get("/projects/{project_id}/export")
def export_project(project_id: str) -> FileResponse:
    """Download a whole project as one zip package.

    Everything under the project directory: ``project.json``,
    ``classes.json``, EVERY bank (not just the active one), the source
    images, the masks and the inspection log. ``/bank/export`` only ever
    packaged one bank of one project, so until now there was no way to move a
    project between machines or to back one up whole.

    STORED, not deflated — the bank feature arrays barely compress and the
    archive runs to GB, so skipping compression keeps the export disk-bound.
    """
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        name = project.name
    root = project_dir(project_id)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="project directory is missing")

    # The bank writer holds io_lock, not state.lock, so both are needed or the
    # archive can pair a post-teach bank.npy with a pre-teach .meta.npz — the
    # same tearing /bank/export guards against. Imported lazily to keep this
    # router free of the torch-adjacent state module at import time.
    from ..core.cls_state import get_state

    state = get_state()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    try:
        with state.lock, state.io_lock, zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(root.rglob("*")):
                if p.is_dir() or p.suffix == ".tmp":
                    continue  # .tmp are in-flight atomic-write leftovers
                zf.write(p, p.relative_to(root).as_posix())
    except BaseException:
        os.unlink(tmp.name)
        raise
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:64] or project_id[:8]
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename=f"{safe}{PROJECT_EXPORT_SUFFIX}",
        background=BackgroundTask(os.unlink, tmp.name),
    )


@router.post("/projects/import", response_model=ProjectRead)
async def import_project(
    archive: UploadFile = File(...),
    name: str = Form(""),
) -> ProjectRead:
    """Load an exported project package as a NEW project.

    Always a new project with a fresh id — never an overwrite of an existing
    one. The archive's ``project.json`` supplies the name (unless one is given)
    and the original creation time; everything else is laid down verbatim,
    including every bank it carried.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    new_id = new_project_id()
    root = project_dir(new_id)
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

        with zipfile.ZipFile(tmp.name) as zf:
            names = zf.namelist()
            if "project.json" not in names:
                raise HTTPException(
                    status_code=422,
                    detail="not a project package — project.json missing at archive root",
                )
            # Ratio-based zip-bomb guard, same rule as the bank import.
            declared = sum(i.file_size for i in zf.infolist())
            if declared > max(8 * total, 64 * 1024 * 1024):
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"archive expands to {declared // (1024*1024)} MB "
                        f"from {total // (1024*1024)} MB — refusing to extract"
                    ),
                )
            for n in names:
                pp = PurePosixPath(n)
                if pp.is_absolute() or ".." in pp.parts or (pp.parts and ":" in pp.parts[0]):
                    raise HTTPException(status_code=422, detail=f"unsafe path in archive: {n}")
            try:
                meta = json.loads(zf.read("project.json").decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise HTTPException(status_code=422, detail="project.json is not valid JSON") from exc
            root.mkdir(parents=True, exist_ok=False)
            zf.extractall(root)

        now = datetime.now(timezone.utc)
        created = now
        raw_created = meta.get("created_at")
        if isinstance(raw_created, str):
            try:
                created = datetime.fromisoformat(raw_created)
            except ValueError:
                created = now
        project = Project(
            id=new_id,
            name=(name or str(meta.get("name") or "imported"))[:200],
            description=str(meta.get("description") or ""),
            memo=meta.get("memo"),
            tags=json.dumps(meta.get("tags") or [], ensure_ascii=False),
            created_at=created,
            updated_at=now,
        )
        # The id inside the package is the exporting machine's; rewrite it or
        # the on-disk copy disagrees with the row that now owns it.
        serialized = ProjectRead.model_validate(project).model_dump(mode="json")
        serialized["schema_version"] = LAYOUT_VERSION
        write_json(root / "project.json", serialized)
        ensure_project_dirs(new_id)  # fill in any directory the package lacked

        engine = get_engine()
        with Session(engine) as session:
            session.add(project)
            log_action(session, "project_import", "project", new_id)
            session.commit()
            session.refresh(project)
            _invalidate_projects_summary_cache()
            return ProjectRead.model_validate(project)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@router.get("/projects", response_model=list[ProjectRead])
def list_projects() -> list[ProjectRead]:
    """List all projects.

    Returns every project row from the database without any image, mask,
    or annotation index counts. Use ``GET /projects/summary`` when the
    counts are needed.
    """
    engine = get_engine()
    with Session(engine) as session:
        results = session.exec(select(Project)).all()
    return [ProjectRead.model_validate(p) for p in results]


_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})

# Memory-bank layout (mirrors core.cls_state, which is not imported
# here to keep the projects router free of torch-adjacent modules).
_BANKS_SUBDIR = "banks"
_BANK_IMAGE_TIERS = ("normal", "critical", "negative")


def _first_bank_image(project_id: str):
    """First image taught into (or staged for) the project's memory banks.

    Projects whose images went straight into a bank (teach via the Develop
    tab) have an empty annotate dataset, but the teach flow saves each
    source file under ``banks/<id>/_images/<tier>/`` and drops land in
    ``banks/<id>/_staging/`` — use those so the project card shows a
    thumbnail from the very first drop. Pure file lookup, no activation.
    """
    root = project_dir(project_id) / _BANKS_SUBDIR
    if not root.is_dir():
        return None
    for bank_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if (bank_dir / ".deleted").exists():
            continue  # partially-deleted bank — mirrors list_banks' skip
        for tier in _BANK_IMAGE_TIERS:
            tier_dir = bank_dir / "_images" / tier
            if not tier_dir.is_dir():
                continue
            for f in sorted(tier_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                    return f
        staging_dir = bank_dir / "_staging"
        if staging_dir.is_dir():
            for f in sorted(staging_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                    return f
    return None


def _bank_counts(project_id: str) -> tuple[int, int]:
    """``(images, labeled)`` across the project's banks: taught + staged.

    cls-studio projects keep their images in memory banks (and the
    dropped-but-untaught staging area), never in the annotate dataset the
    legacy summary counted — without this the card claims "0 images"
    forever no matter how much the operator taught (2026-07-18 report).
    Taught images always carry a tier, so they all count as labeled;
    staged ones count once the operator assigned OK/NG (staging.json).
    """
    root = project_dir(project_id) / _BANKS_SUBDIR
    images = labeled = 0
    if not root.is_dir():
        return 0, 0
    for bank_dir in root.iterdir():
        if not bank_dir.is_dir() or (bank_dir / ".deleted").exists():
            continue
        meta_p = bank_dir / "bank_meta.json"
        if meta_p.exists():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                taught = (
                    len(m.get("bank_images", []))
                    + sum(len(v) for v in m.get("critical_images", {}).values())
                    + sum(len(v) for v in m.get("negative_images", {}).values())
                )
                images += taught
                labeled += taught
            except (OSError, ValueError):
                pass
        staging_dir = bank_dir / "_staging"
        if staging_dir.is_dir():
            try:
                st_meta = json.loads((staging_dir / "staging.json").read_text(encoding="utf-8"))
                if not isinstance(st_meta, dict):
                    st_meta = {}
            except (OSError, ValueError):
                st_meta = {}
            for f in staging_dir.iterdir():
                if f.is_file() and f.name != "staging.json" and not f.name.endswith(".tmp"):
                    images += 1
                    if st_meta.get(f.name):
                        labeled += 1
    return images, labeled


def _quick_file_count(project_id: str) -> tuple[int, int, str | None]:
    """Count images/masks by file existence only — no PIL open, no numpy."""
    imgs_dir = annotate_images_dir(project_id)
    masks_dir = annotate_masks_dir(project_id)
    first_filename: str | None = None
    image_count = 0
    if imgs_dir.exists():
        for p in sorted(imgs_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                image_count += 1
                if first_filename is None:
                    first_filename = p.name
    mask_stems: set[str] = set()
    if masks_dir.exists():
        for p in masks_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".png":
                mask_stems.add(p.stem)
    mask_count = len(mask_stems)
    return image_count, mask_count, first_filename


# Short-lived in-memory cache for the projects summary. Scanning every
# project's annotate index (or falling back to a directory walk) on each
# call is expensive once there are 100+ projects. The project list page
# typically re-renders several times in quick succession, so a 30 s TTL
# cache keeps the first call expensive but makes the follow-ups instant.
# The cache itself lives in core.summary_cache so that ANY mutation path —
# including uploads/deletes in other routers — invalidates it via
# core.db_utils.touch_project().
from ..core.summary_cache import (
    get_cached_summary as _get_cached_summary,
)
from ..core.summary_cache import (
    invalidate_projects_summary_cache as _invalidate_projects_summary_cache,
)
from ..core.summary_cache import (
    set_cached_summary as _set_cached_summary,
)


@router.get("/projects/summary")
def list_projects_summary() -> list[dict[str, Any]]:
    """List projects with image_count, mask_count and first_filename.

    Reads the cached annotate index for each project to avoid a full
    rescan of every mask file (``sync=False``). Falls back to a quick
    file count when ``index.json`` does not yet exist for a project.
    Results are cached in-process for ``_PROJECTS_SUMMARY_TTL_SEC`` to
    cheapen repeated calls from the project list page; every mutating
    endpoint here invalidates that cache.
    """
    cached = _get_cached_summary()
    if cached is not None:
        return cached

    engine = get_engine()
    with Session(engine) as session:
        results = session.exec(select(Project)).all()
        projects = [ProjectRead.model_validate(p).model_dump(mode="json") for p in results]

    summaries = []
    for p in projects:
        image_count = 0
        mask_count = 0
        first_filename = None
        try:
            idx = load_annotate_index(p["id"], sync=False)
            items = idx.get("items", [])
            if items:
                # Index exists and has items — use it directly.
                image_count = len(items)
                mask_count = sum(1 for it in items if (it.get("annotation") or {}).get("hasForeground") or (it.get("annotation") or {}).get("markedClean"))
                first_filename = items[0].get("filename")
            else:
                # No index yet (never opened in Annotate) — quick file count.
                image_count, mask_count, first_filename = _quick_file_count(p["id"])
        except Exception:
            # Last resort: quick file count so cards never lie about having data.
            try:
                image_count, mask_count, first_filename = _quick_file_count(p["id"])
            except Exception:
                pass
        if image_count == 0:
            # cls-studio projects: the images live in memory banks / staging,
            # not the annotate dataset — surface those counts on the card.
            try:
                b_images, b_labeled = _bank_counts(p["id"])
            except OSError:
                b_images, b_labeled = 0, 0
            if b_images:
                image_count, mask_count = b_images, b_labeled
        has_bank_thumbnail = False
        if first_filename is None:
            try:
                has_bank_thumbnail = _first_bank_image(p["id"]) is not None
            except OSError:
                pass
        summaries.append({
            **p,
            "image_count": image_count,
            "mask_count": mask_count,
            "first_filename": first_filename,
            "has_bank_thumbnail": has_bank_thumbnail,
        })
    _set_cached_summary(summaries)
    return summaries


@router.get("/projects/{project_id}/bank-thumbnail")
def project_bank_thumbnail(project_id: str):
    """Serve the first bank-taught image as a project-card thumbnail.

    Fallback for projects with an empty annotate dataset (see
    ``_first_bank_image``); works for inactive projects.
    """
    from fastapi.responses import FileResponse

    f = _first_bank_image(project_id)
    if f is None:
        raise HTTPException(status_code=404, detail="no bank images in this project")
    return FileResponse(f)


# NOTE: must be registered BEFORE the /projects/{project_id} routes below.
# Starlette matches routes in registration order, so if the parameterized
# PUT /projects/{project_id} came first it would swallow PUT /projects/reorder
# (treating "reorder" as a project id and returning 404).
@router.put("/projects/reorder")
def reorder_projects(payload: _ReorderPayload) -> dict[str, str]:
    """Persist the project card display order.

    Accepts an ordered list of project ids and writes the index of each
    id into the project's ``sort_order`` column. Unknown ids in the
    payload are silently skipped so a stale UI cannot 500 the call.
    """
    engine = get_engine()
    with Session(engine) as session:
        for idx, pid in enumerate(payload.order):
            project = session.get(Project, pid)
            if project is not None:
                project.sort_order = idx
                session.add(project)
        session.commit()
    _invalidate_projects_summary_cache()
    return {"status": "ok"}


@router.get("/projects/{project_id}", response_model=ProjectRead)
def get_project(project_id: str) -> ProjectRead:
    """Return a single project by id.

    Raises:
        HTTPException: 404 if no project exists with ``project_id``.
    """
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return ProjectRead.model_validate(project)


@router.put("/projects/{project_id}", response_model=ProjectRead)
def update_project(project_id: str, payload: ProjectUpdate) -> ProjectRead:
    """Update name, description, memo, or tags of an existing project.

    Only fields present (non-None) in the payload are applied; the rest
    are left untouched. ``updated_at`` is refreshed and the on-disk
    ``project.json`` snapshot is rewritten so it stays in sync with the
    database row.

    Raises:
        HTTPException: 404 if no project exists with ``project_id``.
    """
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        if payload.name is not None:
            project.name = payload.name
        if payload.description is not None:
            project.description = payload.description
        if payload.memo is not None:
            project.memo = payload.memo
        if payload.tags is not None:
            project.tags = json.dumps(payload.tags, ensure_ascii=False)
        project.updated_at = datetime.now(timezone.utc)
        session.add(project)
        log_action(session, "project_update", "project", project_id)
        session.commit()
        session.refresh(project)
        result = ProjectRead.model_validate(project)
        write_project_json(project)
    _invalidate_projects_summary_cache()
    return result


@router.delete("/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, str]:
    """Delete a project together with its training runs and model records.

    Stops any in-flight training for the project, deletes the database
    rows for runs and model records, archives the best run, then removes
    the on-disk project directory. If a locked file prevents removal
    (e.g. an antivirus or held handle on Windows), a ``.deleted``
    tombstone is dropped so the startup orphan-adopt does not resurrect
    the project on the next boot.

    Raises:
        HTTPException: 404 if no project exists with ``project_id``.
    """
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        # Stop any running training for this project
        runs = session.exec(
            select(TrainingRun).where(TrainingRun.project_id == project_id)
        ).all()
        for run in runs:
            stop_event = RUN_FLAGS.pop(run.run_id, None)
            if stop_event is not None:
                stop_event.set()
            session.delete(run)
        # Delete related ModelRecords
        models = session.exec(
            select(ModelRecord).where(ModelRecord.project_id == project_id)
        ).all()
        for model in models:
            session.delete(model)
        session.delete(project)
        log_action(session, "project_delete", "project", project_id)
        session.commit()
    # Release the in-memory bank BEFORE removing files: the process-wide
    # active bank otherwise keeps pointing into the removed tree, later
    # teaches silently rebuild it on disk, and the next startup orphan purge
    # destroys everything taught in the meantime (2026-07-17 incident).
    # Imported lazily: this router deliberately avoids torch-adjacent
    # modules at import time (see _first_bank_image), and by the time a
    # delete arrives the bank router has long since loaded them anyway.
    from ..core.cls_state import get_state
    get_state().deactivate_project(project_id)
    # Archive best run before deleting project files
    from ..core.paths import archive_best_run
    try:
        archive_best_run(project_id)
    except Exception as e:
        _logger.warning("Failed to archive best run for project %s: %s", project_id, e)
    path = project_dir(project_id)
    if path.exists():
        # ignore_errors=True so a locked file (Windows AV / held handle) never
        # surfaces as 500 after the DB row is already gone. If the dir survives,
        # drop a .deleted tombstone so the startup orphan-adopt won't resurrect it.
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            try:
                (path / ".deleted").write_text("", encoding="utf-8")
                _logger.warning(
                    "Partial delete for project %s: dir remains, tombstone placed",
                    project_id[:8],
                )
            except Exception as e:
                _logger.warning(
                    "Failed to place tombstone for partially-deleted project %s: %s",
                    project_id[:8], e,
                )
    _invalidate_projects_summary_cache()
    return {"status": "ok"}
