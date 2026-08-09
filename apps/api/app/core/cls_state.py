# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Process-wide cls-studio runtime state, scoped to the active project + bank.

A project can hold several named memory banks under
``PROJECTS_DIR/<project_id>/banks/<bank_id>/``. Selecting a project (and,
optionally, one of its banks) loads that bank into memory; every
bank/score/images route then operates on it. Each bank also keeps the raw
source images an operator taught it under ``<bank>/_images/<tier>/`` so the
UI can show per-tier thumbnails. The DINOv2 backbone is shared across
everything and loaded lazily on the first feature-extracting call.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from clscore.bank import Bank
from clscore.feature_extractor import DEFAULT_DINO_DIM, DEFAULT_DINO_NAME

from . import bank_images
from .bank_images import IMAGES_SUBDIR  # noqa: F401 -- re-export for bank.py + tests
from .cls_schemas import BankState
from .exceptions import (
    ActiveProjectDeletedError,
    BankCorruptError,
    BankNotFoundError,
    NoActiveProjectError,
    PathTraversalError,
    ProjectNotFoundError,
)
from .paths import project_dir, write_bytes_atomic

if TYPE_CHECKING:  # torch is only needed once a model is loaded
    import torch

logger = logging.getLogger(__name__)

BANKS_SUBDIR = "banks"
# File (not a directory, so list_banks ignores it) remembering the project's
# last-active bank id; see ClsStudioState.activate.
LAST_BANK_MARKER = ".last_active"
# IMAGES_SUBDIR is re-exported from bank_images (which owns the bank layout)
# so this module and routers/projects.py cannot drift apart on the spelling.
CAPTURES_SUBDIR = "captures"
DEFAULT_BANK_ID = "default"
DEFAULT_BANK_NAME = "Default"

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _empty_bank() -> Bank:
    """A fresh, dimension-only bank for a project that has none on disk yet."""
    return Bank(normal=np.empty((0, DEFAULT_DINO_DIM), dtype=np.float32))


def slug_bank_id(name: str) -> str:
    """Filesystem-safe bank id from a display name (``Line A`` -> ``line-a``)."""
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return s[:64] or DEFAULT_BANK_ID


@dataclass
class ClsStudioState:
    """Mutable per-process state for the active project + bank."""

    bank: Bank = field(default_factory=_empty_bank)
    bank_dir: Path | None = None
    active_project_id: str | None = None
    active_bank_id: str | None = None
    model_name: str = DEFAULT_DINO_NAME
    backend: str = "torch"
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Serialises bank-to-disk writes independently of ``lock`` so a large
    # save (bank.npy can be GBs) never blocks scoring, which only needs
    # ``lock`` to read the in-memory bank / GPU tensors.
    io_lock: threading.Lock = field(default_factory=threading.Lock)

    # Populated lazily by ensure_model().
    _model: torch.nn.Module | None = None
    _device: str | None = None
    _dtype: Any = None
    # Largest window batch the forward fits on this device (dry-run probed
    # once at model load). None until probed; max_batch() falls back to 32.
    _max_batch: int | None = None

    # Invalidated on any bank mutation (see mark_dirty). Keys are filled
    # lazily per tier so a runtime that never touches the full critical /
    # negative tiers (the exemplar-alpha path) never pays their VRAM.
    _tensor_cache: Any = None

    # ---- GPU lease (idle VRAM release) -------------------------------------
    # A COUNTER, not a mutex: the reaper needs "is any request doing GPU work
    # right now", and ``lock`` answers neither direction of that question. It
    # is held for minutes with zero GPU work by the export routes, and not
    # held at all by teach / ingest / assemble / evaluate / cmin / projection.
    # ``_gpu_lock`` is deliberately a SEPARATE lock: ``lock`` is non-reentrant
    # and /score already holds it across ensure_model + get_normal_tensor, so
    # reusing it here would deadlock scoring on the first request.
    _gpu_lock: threading.Lock = field(default_factory=threading.Lock)
    _gpu_inflight: int = 0
    # Seeded to "now" rather than 0.0: monotonic() is uptime-based, so a 0.0
    # seed makes ``now - last_active > idle_seconds`` true on the reaper's
    # very first tick on any machine that has been up longer than the timeout.
    _gpu_last_active: float = field(default_factory=time.monotonic)
    # When the last release ran. The reaper skips a tick whenever this is at
    # or after ``_gpu_last_active`` -- otherwise an idle server would run a
    # synchronize + empty_cache every poll interval, forever, for nothing.
    _gpu_released_at: float = 0.0

    def mark_dirty(self) -> None:
        self._tensor_cache = None

    @contextlib.contextmanager
    def gpu_lease(self):
        """Mark a span of real GPU work so the idle reaper cannot cut into it.

        Must wrap the *thread-side* function for routes that hand work to a
        threadpool: a client disconnect can abandon the awaiting coroutine
        while the worker thread keeps running, and a lease released by the
        coroutine would under-count.
        """
        with self._gpu_lock:
            self._gpu_inflight += 1
            self._gpu_last_active = time.monotonic()
        try:
            yield
        finally:
            with self._gpu_lock:
                self._gpu_inflight -= 1
                self._gpu_last_active = time.monotonic()

    def leased(self, fn):
        """Wrap a closure so it holds a lease for its whole run.

        For ``run_in_threadpool(state.leased(_work))``: the wrapper is what
        executes on the worker thread, so the lease spans the real work even
        when the awaiting coroutine is abandoned by a client disconnect.
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.gpu_lease():
                return fn(*args, **kwargs)

        return wrapper

    def _tensor_slot(self) -> dict:
        if self._tensor_cache is None:
            self._tensor_cache = {}
        return self._tensor_cache

    def get_normal_tensor(self) -> Any:
        """The normal bank on device — the only tensor every scoring path needs.

        With the int8 compression setting on (the default), the rows are
        int8-quantised and dequantised before upload — the same transform the
        compression sweep verdict-validated. The on-disk bank stays fp16, so
        turning the setting off restores full precision on the next rebuild
        (settings changes call ``mark_dirty``).
        """
        slot = self._tensor_slot()
        if "normal" not in slot:
            # Bind device/dtype from the RETURN value, never by re-reading the
            # fields: the idle reaper can null them between the two reads, and
            # ``.to(None, dtype=None)`` does not raise -- it silently yields a
            # CPU tensor, which only surfaces much later inside torch.cdist as
            # "expected all tensors on the same device".
            _, dev, dt = self.ensure_model()
            import torch

            arr = self.bank.normal
            from .runtime_compression import read_compression_settings

            if read_compression_settings()["int8"] and arr.size:
                from clscore.compress import quantize_int8_roundtrip

                arr = quantize_int8_roundtrip(arr)
            slot["normal"] = torch.from_numpy(arr).to(dev, dtype=dt)
        return slot["normal"]

    def get_normal_ivf(self) -> tuple[Any, int]:
        """``(IvfIndex | None, nprobe)`` for the normal bank under current settings.

        None when routing is disabled or the bank is too small to route.
        The index is cached with the other bank tensors (``mark_dirty``
        drops it) and mirrored to ``<bank>/ivf_index.npz``: on a cache miss
        the mirror is revived when its membership basis is a prefix of the
        current one (append-only growth since it was written — the taught
        delta rides on nearest-centroid assignment) and rebuilt from
        scratch otherwise (deletes, growth past ``REBUILD_GROWTH``, or an
        int8-setting flip, which changes the geometry the clusters were
        fitted to).
        """
        from .runtime_compression import read_compression_settings

        cfg = read_compression_settings()
        if not cfg["ivf"]:
            return None, 0
        from clscore.compress import (
            IVF_INDEX_FILE,
            MIN_IVF_ROWS,
            IvfIndex,
            normal_index_basis,
        )

        n_rows = int(self.bank.normal.shape[0])
        if n_rows < MIN_IVF_ROWS:
            return None, 0
        slot = self._tensor_slot()
        if "ivf" in slot:
            return slot["ivf"], int(cfg["ivf_nprobe"])
        # Bind once from the return value; see get_normal_tensor for why the
        # fields must not be re-read after this point.
        _, dev, dt = self.ensure_model()
        import torch

        from clscore.compress import quantize_int8_roundtrip

        # The index carries its own cluster-sorted storage (int8 when the
        # setting is on), so the full normal tensor is only materialised
        # TRANSIENTLY here for k-means / delta assignment — deliberately not
        # via get_normal_tensor, whose cache would keep it resident and
        # forfeit the VRAM saving.
        arr = self.bank.normal
        # Computed on demand, NOT unconditionally: on the clean mirror-hit path
        # below the quantised copy is never read, and building it eagerly cost
        # two full int8 round trips over the whole bank (GBs of host allocation
        # and seconds of single-threaded numpy) on every single cache miss.
        _geom: list[Any] = []

        def geom() -> Any:
            if not _geom:
                _geom.append(
                    quantize_int8_roundtrip(arr) if cfg["int8"] and arr.size else arr
                )
            return _geom[0]

        basis = normal_index_basis(self.bank.meta.normal_image_index)
        idx = None
        path = self.bank_dir / IVF_INDEX_FILE if self.bank_dir is not None else None
        if path is not None and path.exists():
            cand = IvfIndex.load(path, dev, dt)
            if (
                cand is not None
                and cand.int8 == bool(cfg["int8"])
                and not cand.needs_rebuild(n_rows)
            ):
                covered = int(cand.row_cluster.shape[0])
                m = len(cand.index_basis)
                if covered == n_rows and cand.index_basis == basis:
                    idx = cand
                elif (
                    covered < n_rows
                    and m <= len(basis)
                    and cand.index_basis == basis[:m]
                    and covered == (
                        basis[m - 1]["start"] + basis[m - 1]["count"] if m else 0
                    )
                ):
                    # Append-only growth since the mirror was written: assign
                    # the taught delta to its nearest centroids and move on.
                    delta = torch.from_numpy(
                        np.ascontiguousarray(geom()[covered:])
                    ).to(dev, dtype=dt)
                    cand.extend(delta, index_basis=basis)
                    idx = cand
                    self._persist_ivf(idx)
        if idx is None:
            import logging
            import time

            t0 = time.perf_counter()
            bank_t = torch.from_numpy(np.ascontiguousarray(geom())).to(
                dev, dtype=dt
            )
            idx = IvfIndex.build(
                bank_t, int8=bool(cfg["int8"]), index_basis=basis
            )
            del bank_t
            logging.getLogger(__name__).info(
                "IVF index built: %d rows -> %d clusters in %.1fs",
                n_rows, int(idx.centroids.shape[0]), time.perf_counter() - t0,
            )
            self._persist_ivf(idx)
        if not idx.has_storage:
            idx.set_storage(arr)
        slot["ivf"] = idx
        return idx, int(cfg["ivf_nprobe"])

    def _persist_ivf(self, idx: Any) -> None:
        """Best-effort mirror of the IVF index into the bank directory.

        Guarded like ``save_bank``: never write into a deleted project's
        tree (the write would resurrect it for the startup purge to destroy)
        and never mkdir — a missing bank directory means the bank is gone.
        Failure is harmless: the index is derived data and rebuilds.
        """
        from clscore.compress import IVF_INDEX_FILE

        if self.bank_dir is None:
            return
        pid = self.active_project_id
        pdir = project_dir(pid) if pid else None
        if pdir is None or not pdir.is_dir() or (pdir / ".deleted").exists():
            return
        try:
            with self.io_lock:
                if not self.bank_dir.is_dir():
                    return
                idx.save(self.bank_dir / IVF_INDEX_FILE)
        except OSError as exc:
            import logging

            logging.getLogger(__name__).warning("IVF index mirror failed: %s", exc)

    def get_tier_tensors(self, tier: str) -> Any:
        """``{label: tensor}`` for one labelled tier, moved to device on first use.

        The full critical tier of a whole-image-taught bank is as large as
        the normal bank; only the legacy full-tier alpha/beta paths need it
        on device, so it must not ride along for free.
        """
        slot = self._tensor_slot()
        if tier not in slot:
            # Bind once; see get_normal_tensor.
            _, dev, dt = self.ensure_model()
            import torch

            src = self.bank.critical if tier == "critical" else self.bank.negative
            slot[tier] = {
                lab: torch.from_numpy(arr).to(dev, dtype=dt)
                for lab, arr in src.items()
                if arr is not None and arr.size
            }
        return slot[tier]

    def get_bank_tensors(self) -> Any:
        """Cached ``(normal, critical_by_label, negative_by_label)`` on device.

        Materialises every tier — callers that don't need the labelled tiers
        should use ``get_normal_tensor`` / ``get_tier_tensors`` instead.
        """
        return (
            self.get_normal_tensor(),
            self.get_tier_tensors("critical"),
            self.get_tier_tensors("negative"),
        )

    # ---- bank discovery ----------------------------------------------------

    def _banks_root(self, project_id: str) -> Path:
        return project_dir(project_id) / BANKS_SUBDIR

    def list_banks(self, project_id: str) -> list[dict]:
        """Cheap summary of every bank in ``project_id`` (reads meta JSON only)."""
        root = self._banks_root(project_id)
        out: list[dict] = []
        if not root.is_dir():
            return out
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            # A bank whose rmtree partially failed on Windows (locked
            # thumbnail handle) leaves the directory behind with a ``.deleted``
            # tombstone; skip it so the selector never resurrects a bank the
            # operator already deleted. Mirrors delete_project's tombstone.
            if (d / ".deleted").exists():
                continue
            out.append(self._bank_summary(d))
        return out

    def _bank_summary(self, bank_dir: Path) -> dict:
        bank_id = bank_dir.name
        name = bank_id
        images = {"normal": 0, "critical": 0, "negative": 0}
        meta_path = bank_dir / Bank.META_FILE
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta.get("description") or bank_id
                images["normal"] = len(meta.get("bank_images", []))
                images["critical"] = sum(len(v) for v in meta.get("critical_images", {}).values())
                images["negative"] = sum(len(v) for v in meta.get("negative_images", {}).values())
            except (OSError, ValueError):
                pass
        return {"id": bank_id, "name": name, "images": images}

    # ---- project / bank activation -----------------------------------------

    def activate(self, project_id: str, bank_id: str | None = None) -> None:
        """Bind this state to ``project_id`` and one of its banks.

        ``bank_id`` defaults to the project's last-active bank (remembered in
        a marker file), then the first existing bank, then ``default`` (which
        is created empty on first save). The marker matters because tab
        activations re-select the project without a bank id — without it,
        every tab switch silently snapped the active bank back to the first
        one, so verdict configs and exports went to the wrong bank.

        The project directory must already exist: activating a deleted (or
        never-created) project would silently mkdir a ghost tree that the
        startup orphan purge later destroys together with anything taught
        into it (2026-07-17 data-loss incident).
        """
        pdir = project_dir(project_id)
        if not pdir.is_dir() or (pdir / ".deleted").exists():
            # The tombstone counts as deleted: a Windows partial rmtree leaves
            # the dir behind, and re-selecting it would let hours of teaching
            # accumulate in a tree the next startup purge is contracted to
            # destroy.
            raise ProjectNotFoundError(
                f"project not found on disk: {project_id}",
                context={"project_id": project_id},
            )
        existing = [b["id"] for b in self.list_banks(project_id)]
        marker = self._banks_root(project_id) / LAST_BANK_MARKER
        if bank_id is None:
            remembered: str | None = None
            try:
                remembered = marker.read_text(encoding="utf-8").strip() or None
            except OSError:
                pass
            if remembered is not None and remembered in existing:
                bank_id = remembered
            else:
                bank_id = existing[0] if existing else DEFAULT_BANK_ID
        bank_dir = self._banks_root(project_id) / bank_id
        if (bank_dir / ".deleted").exists():
            # An explicitly-requested bank id can name a partially-deleted
            # bank that list_banks already hides; loading it would overlay an
            # empty bank on the remnants and the next save would wipe the
            # surviving tier files.
            raise BankNotFoundError(f"bank not found: {bank_id}")
        if (bank_dir / Bank.NORMAL_FILE).exists():
            try:
                bank = Bank.load(bank_dir)
            except (OSError, ValueError) as exc:  # corrupt on disk
                # Full path + cause stay server-side only; the client just
                # learns which bank id is corrupt.
                logger.error(
                    "corrupt bank at %s: %s: %s", bank_dir, type(exc).__name__, exc,
                    exc_info=True,
                )
                raise BankCorruptError(
                    f"bank data is corrupt: {bank_id}",
                    detail=f"corrupt bank at {bank_dir}: {type(exc).__name__}: {exc}",
                    context={"project_id": project_id, "bank_id": bank_id},
                ) from exc
        else:
            bank = _empty_bank()
            # A project with no banks lands on an empty "default"; persist it so
            # the bank selector always has at least one entry to show.
            if not existing:
                bank.meta.description = DEFAULT_BANK_NAME
                bank_dir.mkdir(parents=True, exist_ok=True)
                bank.save(bank_dir)
        self.bank = bank
        self.bank_dir = bank_dir
        self.active_project_id = project_id
        self.active_bank_id = bank_id
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic(marker, bank_id.encode("utf-8"))
        except OSError:
            pass  # remembering the choice is best-effort
        self.mark_dirty()

    def create_bank(self, name: str) -> str:
        """Create (and select) a new empty bank in the active project."""
        self.require_active_project()
        assert self.active_project_id is not None
        bank_id = slug_bank_id(name)
        root = self._banks_root(self.active_project_id)
        # De-dup ids so two "Line A"s don't collide.
        candidate, n = bank_id, 1
        while (root / candidate).is_dir():
            n += 1
            candidate = f"{bank_id}-{n}"
        bank_dir = root / candidate
        bank_dir.mkdir(parents=True, exist_ok=True)
        bank = _empty_bank()
        bank.meta.description = name.strip() or candidate
        bank.save(bank_dir)
        self.activate(self.active_project_id, candidate)
        return candidate

    def delete_bank(self, bank_id: str) -> None:
        """Delete a bank directory from the active project, re-selecting another.

        Irreversible: removes ``banks/<bank_id>/`` (feature arrays, marks,
        verdict recipe, thumbnails). When the deleted bank was the active one
        (or the last-active marker still points at it) the project is
        re-activated with ``bank_id=None`` so ``activate`` picks the next
        remaining bank — or, when it was the last bank, creates a fresh empty
        ``default`` — leaving ``self.bank`` / ``self.bank_dir`` always valid.

        Removal is ``ignore_errors`` like ``delete_project``: a held thumbnail
        handle (Windows) must not surface as a 500 after we intend deletion.
        If the directory survives, a ``.deleted`` tombstone hides it from
        ``list_banks`` so the selector doesn't resurrect it.
        """
        self.require_active_project()
        assert self.active_project_id is not None
        project_id = self.active_project_id
        root = self._banks_root(project_id)
        bank_dir = root / bank_id
        if not bank_dir.is_dir() or (bank_dir / ".deleted").exists():
            raise BankNotFoundError(f"bank not found: {bank_id}")
        shutil.rmtree(bank_dir, ignore_errors=True)
        if bank_dir.exists():
            try:
                (bank_dir / ".deleted").write_text("", encoding="utf-8")
            except OSError:
                pass  # best-effort; list_banks still tolerates a live dir
        marker = root / LAST_BANK_MARKER
        try:
            remembered = marker.read_text(encoding="utf-8").strip()
        except OSError:
            remembered = ""
        if self.active_bank_id == bank_id or remembered == bank_id:
            self.activate(project_id, None)

    def reset_model(self) -> None:
        """Drop the lazily-loaded backbone and device-bound tensor caches.

        Called when the configured torch device changes: ensure_model()
        early-returns once loaded, so without this a device change from the
        UI silently has no effect (old device keeps every teach/score) until
        the next restart.
        """
        with self.lock:
            self._model = None
            self._device = None
            self._dtype = None
            self._max_batch = None
            self.mark_dirty()
        # Dropping the references is not enough: the caching allocator keeps
        # every freed block reserved, so the old device's arena stays held
        # until the process exits. Best-effort, never fatal.
        self.release_vram()

    # ---- VRAM accounting + release ------------------------------------------

    def _cuda_device(self) -> str | None:
        """The configured device string when it is CUDA, else None.

        Never forces a model load and never imports torch: callers use it to
        decide whether a ``torch.cuda.*`` call is even legal. ``self._device``
        is "cpu" whenever the device resolver fell back, and every
        ``torch.cuda.*`` accessor raises on a cpu device string.
        """
        import sys

        dev = self._device
        if isinstance(dev, str) and dev.startswith("cuda"):
            return dev
        if dev is None:
            # reset_model() nulls the field before it flushes, and a release
            # can land before the first ensure_model(); fall back to whatever
            # device torch actually initialised so the arena still goes back.
            torch = sys.modules.get("torch")
            try:
                if torch is not None and torch.cuda.is_initialized():
                    return f"cuda:{torch.cuda.current_device()}"
            except Exception:  # noqa: BLE001 - probing must never raise
                return None
        return None

    def vram_stats(self) -> dict:
        """Process-local VRAM numbers for the configured device.

        ``memory_reserved`` is the one that answers "did the arena go back" —
        ``memory_allocated`` only counts live tensors and barely moves when
        the scoring scratch is freed. Machine-wide nvidia-smi figures cannot
        answer it at all on Windows/WDDM, where per-process memory reads N/A.

        Returns ``{"available": False}`` rather than importing torch: asking
        this question must never be what creates a CUDA context.
        """
        import sys

        torch = sys.modules.get("torch")
        dev = self._cuda_device()
        if torch is None or dev is None:
            return {"available": False, "device": self._device}
        try:
            return {
                "available": True,
                "device": dev,
                "allocated_mb": round(torch.cuda.memory_allocated(dev) / (1024 ** 2)),
                "reserved_mb": round(torch.cuda.memory_reserved(dev) / (1024 ** 2)),
            }
        except Exception:  # noqa: BLE001 - a stats read must never 500 a route
            return {"available": False, "device": dev}

    def release_vram(
        self, drop_bank: bool = False, drop_model: bool = False
    ) -> dict:
        """Hand the caching allocator's free arena back to the driver.

        The default release keeps BOTH the backbone and the device-resident
        bank: they are ~173 MB and ~1 GB respectively, while the bulk of a
        long-lived server's footprint is the reserved-but-free arena left at
        the high-water mark of the largest scoring distance matrix. Flushing
        that costs nothing to rebuild; dropping the bank costs a full int8
        re-quantisation of every row on the next score, for the last ~1 GB.

        Refuses while any :meth:`gpu_lease` is held. Nothing here can fail a
        request: dropping a cache only drops a reference, and an in-flight
        caller keeps its own. ``empty_cache`` cannot reclaim a referenced
        block, so a release racing real work is a no-op, not a corruption.
        """
        import sys

        with self._gpu_lock:
            if self._gpu_inflight:
                return {"released": False, "reason": "busy", "inflight": self._gpu_inflight}
            before = self.vram_stats()
            # Pure-Python drops first and outside any torch guard, so the part
            # that always works is never skipped by a torch call that raised.
            if drop_bank:
                self.mark_dirty()
            if drop_model:
                self._model = None
                self._max_batch = None
            torch = sys.modules.get("torch")
            dev = self._cuda_device()
            if torch is None or dev is None:
                self._gpu_released_at = time.monotonic()
                return {"released": False, "reason": "no_cuda", "before": before}
            try:
                if drop_bank or drop_model:
                    # nn.Module graphs and the IVF index hold reference cycles;
                    # without a collection their CUDA storages may outlive the
                    # assignment above and empty_cache() would not see them.
                    import gc

                    gc.collect()
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - best effort by design
                logger.warning("VRAM release failed: %s", exc, exc_info=True)
                return {"released": False, "reason": "error", "before": before}
            self._gpu_released_at = time.monotonic()
            after = self.vram_stats()
            freed = None
            if before.get("available") and after.get("available"):
                freed = before["reserved_mb"] - after["reserved_mb"]
            logger.info(
                "VRAM release: reserved %s -> %s MB (drop_bank=%s drop_model=%s)",
                before.get("reserved_mb"), after.get("reserved_mb"), drop_bank, drop_model,
            )
            return {
                "released": True,
                "before": before,
                "after": after,
                "freed_mb": freed,
            }

    def deactivate_project(self, project_id: str) -> None:
        """Drop the in-memory bank if it belongs to ``project_id``.

        Called by ``DELETE /projects/{id}`` before it removes the tree: the
        process-wide active bank otherwise keeps pointing into the deleted
        directory, later teaches silently rebuild it on disk, and the next
        startup orphan purge destroys everything taught in the meantime
        (the 2026-07-17 data-loss incident).
        """
        if self.active_project_id != project_id:
            return
        with self.lock:
            if self.active_project_id != project_id:
                return  # re-activated by a concurrent select while we waited
            self.bank = _empty_bank()
            self.bank_dir = None
            self.active_project_id = None
            self.active_bank_id = None
            self.mark_dirty()

    def require_active_project(self) -> None:
        """Guard for routes that need a selected project.

        Also re-checks that the active project still exists on disk: a
        ``DELETE /projects/{id}`` that raced this request (another client,
        another tab) must turn every later bank route into a clean 409
        instead of letting a write resurrect the removed tree.
        """
        if self.active_project_id is None:
            raise NoActiveProjectError(
                "no active project — POST /api/v1/bank/select first"
            )
        pdir = project_dir(self.active_project_id)
        if not pdir.is_dir() or (pdir / ".deleted").exists():
            # Raise only — never deactivate here: this guard runs both with
            # and without ``self.lock`` held (e.g. delete_bank under the
            # router's lock) and deactivate_project acquires the non-reentrant
            # lock. The dangling state is harmless: every later route hits
            # this same 409 until a fresh select rebinds it.
            raise ActiveProjectDeletedError(
                "active project was deleted — select another project"
            )

    def require_active(self) -> None:
        """Guard for routes that need a selected project + bank."""
        self.require_active_project()
        if self.bank_dir is None:
            raise NoActiveProjectError(
                "no active project — POST /api/v1/bank/select first"
            )

    def save_bank(self, parts: tuple[str, ...] | None = None) -> None:
        """Persist the bank (or just the changed ``parts``) to disk.

        Serialised by ``io_lock``, not ``lock`` — callers should do the
        in-memory mutation under ``lock``, release it, then call this so the
        (potentially multi-GB) write overlaps with, rather than blocks,
        concurrent scoring. ``parts=(tier,)`` skips rewriting untouched tiers
        (a single-image teach shouldn't rewrite the whole normal bank).
        """
        self.require_active()
        assert self.bank_dir is not None
        with self.io_lock:
            # Final choke point for every mkdir-resurrection path: a project
            # delete that landed between the caller's require_active() and
            # this write must not have its tree quietly rebuilt (the startup
            # purge would then destroy whatever we write into it).
            pid = self.active_project_id
            pdir = project_dir(pid) if pid else None
            if pdir is None or not pdir.is_dir() or (pdir / ".deleted").exists():
                # Raise only (no deactivation) — callers may hold self.lock;
                # see require_active_project.
                raise ActiveProjectDeletedError(
                    "active project was deleted — select another project"
                )
            self.bank_dir.mkdir(parents=True, exist_ok=True)
            self.bank.save(self.bank_dir, parts=parts)

    # ---- per-tier source images (thumbnails) -------------------------------

    def images_dir(self, tier: str) -> Path:
        self.require_active()
        assert self.bank_dir is not None
        return bank_images.image_roots(self.bank_dir).tier(tier)

    @staticmethod
    def safe_image_name(filename: str) -> str:
        """The one rule that decides what a taught image is called on disk.

        Delegates to :func:`bank_images.safe_image_name`, which is the rule
        both the writer here and every reader elsewhere now share. Kept as a
        method because callers (and ``test_image_name_rule``) reach it through
        the state object.
        """
        return bank_images.safe_image_name(filename)

    def save_source_image(self, tier: str, filename: str, data: bytes) -> str:
        """Persist a taught image so the UI can thumbnail it. Returns the name used."""
        d = self.images_dir(tier)
        d.mkdir(parents=True, exist_ok=True)
        safe = self.safe_image_name(filename)
        # Formats browsers render natively are served as uploaded. Everything
        # else (TIFF above all) is transcoded to PNG so the thumbnail always
        # shows; PNG is lossless, so a later re-teach from the stored copy is
        # bit-identical to the original's decoded pixels. Scoped to the
        # non-web formats so ordinary JPEGs aren't inflated into large PNGs.
        if not bank_images.has_image_ext(safe, bank_images.DISPLAYABLE_EXTS):
            try:
                import cv2
                arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if ok:
                        data = buf.tobytes()
                        safe = f"{Path(safe).stem or 'image'}.png"
            except Exception:
                pass  # undecodable here → keep raw bytes; feature extraction already validated it
        target = d / safe
        stem, suffix = target.stem, target.suffix or ".png"
        i = 1
        while target.exists():
            target = d / f"{stem}_{i}{suffix}"
            i += 1
        target.write_bytes(data)
        return target.name

    def image_path(self, tier: str, name: str) -> Path:
        """Resolve a source-image path, rejecting traversal.

        The name is mapped through ``safe_image_name`` rather than validated
        against it: callers pass the ORIGINAL filename, and the copy on disk
        is the rewritten one. Rejecting instead of mapping is what made the
        thumbnail route fail for every name with a space or a non-ASCII
        character, and it failed BEFORE the store fallback that exists to keep
        the viewer off a black rectangle.

        The traversal check is the actual guard and is unchanged: the mapping
        keeps "." and "-" (so "..", which the old regex also allowed, still
        reaches it) and strips both separators, so nothing new gets past it.
        It lives in ``bank_images.tier_image_path`` now, together with the
        name rule it has to agree with.
        """
        self.require_active()
        assert self.bank_dir is not None
        try:
            return bank_images.tier_image_path(self.bank_dir, tier, name)
        except bank_images.UnsafeImageName as exc:
            raise PathTraversalError("invalid image path") from exc

    def captures_dir(self) -> Path:
        self.require_active()
        return project_dir(self.active_project_id) / CAPTURES_SUBDIR  # type: ignore[arg-type]

    def inspections_dir(self) -> Path:
        """Per-project store for operator inspection results (log + previews).

        Lets the Operator tab survive a browser reload: results that finished
        server-side are re-listed from here instead of evaporating with the
        page's state.
        """
        self.require_active()
        return project_dir(self.active_project_id) / "inspections"  # type: ignore[arg-type]

    # ---- lazy model --------------------------------------------------------

    def ensure_model(self) -> tuple[torch.nn.Module, str, Any]:
        """Load the shared DINOv2 backbone on first use; cache thereafter."""
        # Snapshot, then test the snapshot. Re-reading the fields after the
        # guard let a concurrent release null them between the two reads and
        # return (model, None, None) -- which does not raise, it produces CPU
        # tensors, and fails much later and much further away.
        self._gpu_last_active = time.monotonic()
        model_, dev_, dtype_ = self._model, self._device, self._dtype
        if model_ is not None and dev_ is not None:
            return model_, dev_, dtype_
        import torch

        from clscore.feature_extractor import load_dinov2

        from .runtime_settings import read_runtime_settings
        from .torch_device import resolve_torch_device_or_cpu

        configured = str(read_runtime_settings().get("torch_device", "auto"))
        device = resolve_torch_device_or_cpu(configured)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = load_dinov2(self.model_name, device)
        # Actually halve it. This reported _dtype=float16 while leaving the
        # weights fp32, so every staged window and every token was twice the
        # size the rest of the code budgets for, and the tensor cores went
        # unused. load_dinov2's own docstring says callers apply .half() first;
        # nothing did. CUDA only -- fp16 matmul on CPU is slower, not faster.
        # Guarded on having parameters: a backbone whose weights live outside
        # torch (the OpenVINO runtime owns its own) exposes none to halve.
        if dtype == torch.float16 and next(iter(model.parameters()), None) is not None:
            model = model.half()
        self._model, self._device, self._dtype = model, device, dtype
        # Probe the largest window batch that fits, once, so teaching /
        # heatmaps run as fast as the device allows without OOMing on small
        # GPUs. Non-fatal: any failure leaves _max_batch None -> 32 fallback.
        try:
            from clscore.feature_extractor import probe_max_batch
            self._max_batch = probe_max_batch(model, device)
        except Exception:
            self._max_batch = None
        return model, device, dtype

    def max_batch(self) -> int:
        """Probed window batch for DINOv2 forwards (teaching / heatmaps)."""
        return self._max_batch or 32

    def cdist_chunk_for(self, n_bank: int, ivf_active: bool = False) -> int:
        """Query-row chunk for scoring against an ``n_bank``-row bank.

        Sized against the smaller of the driver's free VRAM and this server's
        own budget. Free VRAM alone was the wrong basis and is what made the
        server look like it had leaked: the distance matrix is deliberately
        grown to 60% of whatever is free, the caching allocator then holds
        that high-water mark for the life of the process, and on an otherwise
        idle 24 GB card a single score reserves most of it. Worse, the basis
        is a ratchet -- free VRAM is *larger* right after an idle release, or
        right after a neighbouring job exits, so each release made the next
        score grab more. The budget replaces the ratchet with a ceiling.

        The budget is on the PROCESS, not on the one allocation, so the sum
        must count the arena we already hold: ``mem_get_info`` reports
        driver-level free memory, which excludes our own reserved-but-unused
        blocks. Ignoring them would shrink every later request to the floor
        the moment the server reached its budget.

        CPU keeps the library default (512). ``ivf_active`` reserves headroom
        for the routing mask (one bool per distance-matrix cell, +50% of fp16).
        """
        dev = self._device
        if not isinstance(dev, str) or not dev.startswith("cuda"):
            return 512
        try:
            import torch

            from clscore.scoring import safe_cdist_chunk

            from .runtime_vram import read_vram_settings

            driver_free, _ = torch.cuda.mem_get_info(dev)
            budget_mb = int(read_vram_settings()["budget_mb"])
            if budget_mb <= 0:
                avail = int(driver_free)  # 0 = unlimited: the old behaviour
            else:
                reserved = int(torch.cuda.memory_reserved(dev))
                allocated = int(torch.cuda.memory_allocated(dev))
                # Reusable inside our arena, plus what we may still take from
                # the driver without going over budget.
                avail = (reserved - allocated) + min(
                    int(driver_free), max(0, budget_mb * 1024 * 1024 - reserved)
                )
            elem = 2 if self._dtype == torch.float16 else 4
            return safe_cdist_chunk(
                n_bank, max(0, avail), elem_bytes=elem,
                overhead=4.5 if ivf_active else 3.0,
            )
        except Exception:
            return 512

    @property
    def device(self) -> str:
        """Device string for status responses (resolved before a model loads)."""
        dev = self._device  # snapshot: a release can null it between reads
        if dev:
            return dev
        try:
            from .runtime_settings import read_runtime_settings
            from .torch_device import resolve_torch_device_or_cpu
            return resolve_torch_device_or_cpu(str(read_runtime_settings().get("torch_device", "auto")))
        except Exception:
            return "cpu"


_STATE = ClsStudioState()


def get_state() -> ClsStudioState:
    return _STATE


def gpu_leased(fn):
    """Mark a function as doing real GPU work, for the idle VRAM reaper.

    Reads the process singleton rather than an argument, so it applies
    equally to FastAPI route handlers (which resolve the state themselves)
    and to helpers that take it as their first parameter — there is exactly
    one state per process. ``functools.wraps`` keeps ``__wrapped__``, which
    is what lets FastAPI still read the original signature for dependency
    injection.

    Must decorate the function that runs ON the worker thread. A lease taken
    in an ``async`` handler around ``run_in_threadpool`` would be released
    when the client disconnects and the await is abandoned, while the worker
    keeps running — under-counting exactly when it matters.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with get_state().gpu_lease():
            return fn(*args, **kwargs)

    return wrapper


def bank_state_of(bank: Bank) -> BankState:
    """Serialise a :class:`Bank` into the API ``BankState`` shape."""
    return BankState(
        normal=int(bank.normal.shape[0]),
        critical=bank.tier_size("critical"),
        negative=bank.tier_size("negative"),
        critical_by_label=bank.label_sizes("critical"),
        negative_by_label=bank.label_sizes("negative"),
        dim=int(bank.normal.shape[1]) if bank.normal.size else DEFAULT_DINO_DIM,
    )
