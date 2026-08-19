# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Frozen DINOv2 feature extraction with sliding-window batching.

`extract_windows_tokens_batched` collapses N per-window forward passes into
ceil(N / max_batch) batched calls, which is the recommended path for any
non-trivial image. The single-window helper is kept for tests and CPU debug.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import numpy as np
import torch

from .bg_backbone import BG_BACKBONES, is_bg_backbone, load_bg_backbone
from .bg_backbone import TARGET_DIM as BG_TARGET_DIM
from .preprocess import IMAGENET_MEAN, IMAGENET_STD, normalize_window
from .sw import DINO_PATCH, WINDOW_SIZE, pad_to_min, sw_offsets

logger = logging.getLogger(__name__)


def is_cuda_oom(exc: BaseException) -> bool:
    """True for a CUDA allocator failure, whatever type it arrives as.

    Most come through as ``torch.OutOfMemoryError``, but a cuBLAS workspace
    that cannot be allocated surfaces as a plain ``RuntimeError`` whose only
    marker is the message. Matching the message alone would swallow a host
    ``MemoryError``, which needs a different remedy, so both are checked.
    """
    oom_cls = getattr(torch, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    message = str(exc).lower()
    if "out of memory" not in message:
        return False
    return "cuda" in message or "cublas" in message


def _release_cuda_cache(device: str) -> None:
    """Hand cached blocks back to the driver before retrying a smaller batch.

    Without this the allocator keeps the reserved-but-unallocated arena that
    just failed to satisfy the request, so the retry can hit the same wall
    with a batch that would otherwise fit.
    """
    if not str(device).startswith("cuda"):
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must never mask the real error
        pass

__all__ = [
    "BACKBONE_DIMS",
    "DEFAULT_DINO_DIM",
    "DEFAULT_DINO_NAME",
    "DINO_MODELS",
    "extract_image_features_for_bank",
    "extract_images_features_batched",
    "extract_window_tokens",
    "extract_windows_tokens_batched",
    "is_cuda_oom",
    "load_backbone",
    "load_dinov2",
    "maybe_compile",
    "probe_max_batch",
    "warmup_model",
]

# torch.compile mode env override. Defaults to "reduce-overhead" which uses
# CUDA graphs for the smallest per-call launch overhead — a good fit for an
# inference-only ViT with a fixed input shape (224x224 windows). Users can
# fall back to "default" if Triton on Windows misbehaves, or "off" to skip
# compilation entirely.
COMPILE_MODE_ENV = "CLS_COMPILE_MODE"
_VALID_MODES = {"default", "reduce-overhead", "max-autotune"}

DEFAULT_DINO_NAME: str = "dinov2_vitb14"
DEFAULT_DINO_DIM: int = 768

# DINOv2 backbone variants and their patch-token dimensions. A bank built
# with one variant cannot be queried with another (dimensions don't match),
# so the active bank's ``meta.model`` is the source of truth for which
# variant must be loaded at scoring time.
DINO_MODELS: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# Unified catalogue: DINOv2 variants + BG-aware backbones. All BG backbones
# produce 768-dim tokens by design (see bg_backbone.TARGET_DIM) so the bank
# / scoring code is dim-agnostic to which BG variant is active. The
# distinction matters only for selecting the right loader (load_dinov2 vs
# load_bg_backbone) — callers should prefer ``load_backbone(name, device)``
# below rather than dispatching on this dict directly.
BACKBONE_DIMS: dict[str, int] = {
    **DINO_MODELS,
    **{name: BG_TARGET_DIM for name in BG_BACKBONES},
}

# Re-exported so existing readers keep working; the values live in
# clscore.preprocess, which is the single place the contract is written.
_IMAGENET_MEAN = IMAGENET_MEAN
_IMAGENET_STD = IMAGENET_STD


def _compile_mode() -> str | None:
    """Return the requested torch.compile mode, or None to skip compilation.

    "off" / unset on CPU / unknown value => None. We only attempt compile on
    CUDA because (a) Triton on CPU is research-quality and (b) the speedup
    targets kernel-launch overhead which CPU doesn't have anyway.
    """
    raw = os.environ.get(COMPILE_MODE_ENV, "reduce-overhead").strip().lower()
    if raw in ("off", "0", "false", "no"):
        return None
    if raw not in _VALID_MODES:
        logger.warning(
            "%s=%r is not one of %s; skipping torch.compile",
            COMPILE_MODE_ENV, raw, sorted(_VALID_MODES),
        )
        return None
    return raw


def maybe_compile(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Wrap the patch-token methods used by cls-studio with ``torch.compile``.

    Two reasons we compile bound methods instead of the whole module:

    1. cls-studio calls ``model.forward_features(x)`` and
       ``model.get_intermediate_layers(...)``, never plain ``model(x)``.
       ``torch.compile(model)`` only intercepts ``__call__``, so calling a
       different method silently bypasses the compiled graph.
    2. Compiling specific methods leaves every other attribute (``parameters()``,
       dtype probing in ``extract_windows_tokens_batched``) untouched.

    Compilation is silently skipped on CPU or if Triton fails to import.
    First call after this returns will spend 10-30s building the graph;
    callers who want a snappy first request should run a warmup forward.
    """
    mode = _compile_mode()
    if mode is None or not device.startswith("cuda"):
        return model
    try:
        # Probe Triton availability up front — clearer error than the cryptic
        # one torch.compile would surface on first invocation.
        import triton  # noqa: F401
    except ImportError as exc:
        logger.warning("torch.compile disabled: triton import failed (%s)", exc)
        return model
    except Exception:
        logger.warning("torch.compile disabled: unexpected triton probe failure",
                       exc_info=True)
        return model
    try:
        model.forward_features = torch.compile(  # type: ignore[assignment]
            model.forward_features, mode=mode, fullgraph=False
        )
        model.get_intermediate_layers = torch.compile(  # type: ignore[assignment]
            model.get_intermediate_layers, mode=mode, fullgraph=False
        )
        logger.info(
            "torch.compile applied (mode=%s) to forward_features / get_intermediate_layers",
            mode,
        )
    except Exception:
        logger.warning("torch.compile failed (mode=%s) — falling back to eager",
                       mode, exc_info=True)
    return model


def load_dinov2(model_name: str = DEFAULT_DINO_NAME, device: str = "cuda:0") -> torch.nn.Module:
    """Download (or reuse cache) and put DINOv2 on `device` in eval mode.

    Returns the eager model — callers should apply dtype changes (``.half()``)
    *first*, then call :func:`maybe_compile` last. Compiling before a dtype
    swap forces a recompile on the next inference call (``reduce-overhead``
    mode bakes parameter dtypes into its CUDA graph).
    """
    logger.info("loading %s ...", model_name)
    # The ":main" is load-bearing, not decoration. Given a repo with no ref,
    # torch.hub.load has to work out the default branch, and it does that by
    # fetching github.com/<owner>/<repo>/tree/main on EVERY call -- before it
    # ever looks at the local cache. So a machine with the repo already cached
    # still cannot load a model offline, and a machine that loads models in a
    # loop (a CI run, a batch teach) gets rate-limited: _parse_repo_info
    # re-raises any HTTPError that is not a 404, and a 403 from GitHub then
    # surfaces as a KeyError deep inside urllib. With the ref given, the
    # cache-hit path makes no network call at all.
    #
    # "main" resolves to the same cache directory the unpinned call already
    # used (facebookresearch_dinov2_main), so nothing re-downloads. It pins
    # the ref, NOT a commit -- a commit pin would be stronger for
    # reproducibility but would invalidate every existing user's cache.
    model = torch.hub.load("facebookresearch/dinov2:main", model_name)
    return model.eval().to(device)


def load_backbone(name: str, device: str = "cuda:0") -> torch.nn.Module:
    """Backbone loader dispatch.

    Routes to :func:`load_dinov2` for DINOv2 variants and to
    :func:`load_bg_backbone` for distilled BG-aware backbones (everything
    starting with ``bg_``). The returned module always exposes
    ``forward_features(x)["x_norm_patchtokens"]`` so the downstream batched
    extractor doesn't need to branch.

    Use this instead of calling ``load_dinov2`` directly anywhere a user
    might pass a BG-aware backbone name (CLI ``--model``, ``/api/model/select``,
    bank-recorded model auto-load).
    """
    if is_bg_backbone(name):
        return load_bg_backbone(name, device=device)
    if name in DINO_MODELS:
        return load_dinov2(name, device=device)
    raise KeyError(
        f"unknown backbone {name!r}; valid names: {sorted(BACKBONE_DIMS)}"
    )


@torch.no_grad()
def warmup_model(model: torch.nn.Module, device: str, max_batch: int = 32) -> None:
    """Force torch.compile to trace + build kernels before the first request.

    Runs one batched forward at the same shape/dtype as the real scoring
    path. On CPU or with compile disabled this is a cheap no-op pass that
    just primes CUDA caches; on first compiled run it pays the 10-30s
    Triton autotune up front so the first user request isn't penalised.
    """
    if not isinstance(device, str):
        return
    target_dtype = next(model.parameters()).dtype
    dummy = torch.zeros(
        (max_batch, 3, WINDOW_SIZE, WINDOW_SIZE),
        device=device, dtype=target_dtype,
    )
    try:
        _ = model.forward_features(dummy)["x_norm_patchtokens"]
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        logger.warning("warmup forward failed", exc_info=True)


@torch.no_grad()
def probe_max_batch(
    model: torch.nn.Module,
    device: str,
    candidates: tuple[int, ...] = (64, 48, 32, 24, 16, 8),
    min_batch: int = 4,
) -> int:
    """Largest window batch that runs a full 518² forward without OOM.

    Dry-run probe (largest-first probing to avoid OOM while keeping
    throughput high): try each candidate largest-first, run one real forward,
    and return the first that fits. A too-large fixed batch OOMs on small
    GPUs; a too-small one wastes a big GPU — this picks the sweet spot per
    device. Non-CUDA devices have no CUDA-OOM semantics, so the largest
    candidate is returned unprobed. Result is meant to be cached by the
    caller (it costs a few forwards). ``empty_cache`` between tries so a
    failed attempt's reserved blocks don't poison the next.
    """
    if not isinstance(device, str) or not device.startswith("cuda") or not torch.cuda.is_available():
        return candidates[0]
    first_param = next(iter(model.parameters()), None)
    target_dtype = first_param.dtype if first_param is not None else torch.float32
    for b in candidates:
        try:
            dummy = torch.zeros((b, 3, WINDOW_SIZE, WINDOW_SIZE), device=device, dtype=target_dtype)
            _ = model.forward_features(dummy)["x_norm_patchtokens"]
            torch.cuda.synchronize(device)
            del dummy
            torch.cuda.empty_cache()
            logger.info("probe_max_batch: selected window batch %d on %s", b, device)
            return b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception:
            # A non-OOM failure (bad shape, compile issue) is not something a
            # smaller batch would fix — fall back to the conservative floor.
            logger.warning("probe_max_batch: non-OOM forward failure, using min_batch", exc_info=True)
            torch.cuda.empty_cache()
            return min_batch
    return min_batch


def _normalize_window(window_bgr: np.ndarray) -> torch.Tensor:
    """One BGR uint8 window -> CHW float32 tensor in ImageNet normalization (no batch dim).

    Thin wrapper over :func:`clscore.preprocess.normalize_window`; the chain
    itself lives there so an exported runtime reads the same numbers instead
    of a copy that can drift.
    """
    return torch.from_numpy(normalize_window(window_bgr))


@torch.no_grad()
def extract_image_features_for_bank(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: str,
    max_batch: int = 32,
) -> np.ndarray:
    """Sliding-window forward + flatten — shared bank-append feature pipeline.

    Every code path that appends one image's patches to a bank (Gradio UI,
    FastAPI ``/api/score`` / ``/api/bank/append/*``) needs the exact same
    preprocessing chain so the resulting features land in the bank's
    feature space: pad to the minimum SW grid, cut into overlapping
    ``WINDOW_SIZE`` crops, run a batched DINOv2 forward, and flatten patch
    tokens to a flat ``float32`` matrix.

    Args:
        model: Loaded backbone exposing
            ``forward_features(x)["x_norm_patchtokens"]``.
        image_bgr: ``HxWx3`` BGR ``uint8`` image as decoded by OpenCV.
        device: Device string for the forward pass.

    Returns:
        ``[N_patches, D]`` ``float32`` array, ready for ``Bank.append``.
    """
    padded, _ = pad_to_min(image_bgr)
    offsets = sw_offsets(*padded.shape[:2])
    crops = [padded[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE] for (y, x) in offsets]
    if not crops:
        return np.zeros((0, 0), dtype=np.float32)
    # Stream into the output instead of building the whole token tensor first.
    # This used to take the default keep_on_device=True, so every window's
    # tokens sat in VRAM — 673 MB for a 6000x4000 image, doubled by the final
    # concatenation — purely to be copied straight out to the host on the next
    # line, on a card this box shares with a sibling service. The old
    # `.astype(np.float32)` was a second full copy on top: the backbone is
    # fp32, and astype always copies. Writing each mini-batch into a
    # preallocated array costs one copy and no accumulation (2026-07-31).
    side = WINDOW_SIZE // DINO_PATCH
    out: np.ndarray | None = None
    pos = 0
    for tok in iter_windows_tokens_batched(
        model, crops, device, max_batch=max_batch, keep_on_device=False
    ):
        arr = tok.reshape(-1, tok.shape[-1]).numpy()
        if out is None:
            out = np.empty((len(crops) * side * side, arr.shape[1]), dtype=np.float32)
        out[pos : pos + arr.shape[0]] = arr  # assignment casts if the model is fp16
        pos += arr.shape[0]
        del arr
    return out[:pos] if out is not None else np.zeros((0, 0), dtype=np.float32)


@torch.no_grad()
def extract_images_features_batched(
    model: torch.nn.Module,
    images_bgr: list[np.ndarray],
    device: str,
    max_batch: int = 32,
) -> list[np.ndarray]:
    """Bank features for MANY images, batching every image's windows through
    the backbone together.

    Per-image teaching runs one small forward (~6 windows) then stalls on CPU
    work, so the GPU never fills. This collects the sliding windows from all
    images into a single ``extract_windows_tokens_batched`` call — which packs
    them into full ``max_batch`` forwards — then splits the tokens back per
    image. Each returned array is bit-identical to
    :func:`extract_image_features_for_bank` on that image (same pad / crop /
    reshape); only the forward is shared, so banks stay consistent.

    Returns a list aligned with ``images_bgr`` (``[N_patches, D]`` float32).

    Holds every image's features at once by construction. Callers that reduce
    each image right away (teach coreset-caps to a couple of thousand patches)
    should iterate :func:`iter_images_features_batched` instead and never
    materialise the group — see the host-memory note there.
    """
    return [feats for _i, feats in iter_images_features_batched(model, images_bgr, device, max_batch)]


@torch.no_grad()
def iter_images_features_batched(
    model: torch.nn.Module,
    images_bgr: list[np.ndarray],
    device: str,
    max_batch: int = 32,
) -> Iterator[tuple[int, np.ndarray]]:
    """Streaming form of :func:`extract_images_features_batched`.

    Yields ``(index, features)`` the moment an image's windows have all been
    through the backbone, so the group's tokens never exist together. The
    forwards, their order and every array are identical to the list form —
    only the lifetime differs.

    This matters because host memory here scales with the GROUP, not with
    ``max_batch``. One 6000x4000 image is ~160 windows and a window's tokens
    are 4.2 MB in fp32, so a full 24-image group is ~16 GB of tokens plus the
    same again as float32 arrays. Measured on the dev box: a single 24 MP teach
    peaked +6.2 GB and a group of four peaked +7.5 GB, none of which the
    allocator hands back to the OS afterwards — the process simply stays that
    large, which is what reads as a leak (2026-07-31).
    """
    if not images_bgr:
        return
    all_crops: list[np.ndarray] = []
    counts: list[int] = []
    for image_bgr in images_bgr:
        padded, _ = pad_to_min(image_bgr)
        offsets = sw_offsets(*padded.shape[:2])
        crops = [padded[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE] for (y, x) in offsets]
        counts.append(len(crops))
        all_crops.extend(crops)

    # keep_on_device=False: every token here becomes a host-side float32 array,
    # so there is no reason for the full set to pass through VRAM on the way.
    # This is what a batch teach of large images used to run out of memory
    # doing. Mini-batches are consumed as they arrive rather than concatenated,
    # which is what keeps the host side bounded too.
    idx = 0
    buf: list[torch.Tensor] = []
    buf_n = 0
    for tok in iter_windows_tokens_batched(
        model, all_crops, device, max_batch=max_batch, keep_on_device=False
    ):
        buf.append(tok)
        buf_n += int(tok.shape[0])
        # A mini-batch can straddle an image boundary, so drain every image the
        # buffer now completes before asking for more windows.
        while idx < len(counts) and buf_n >= counts[idx]:
            need = counts[idx]
            merged = buf[0] if len(buf) == 1 else torch.cat(buf, dim=0)
            t = merged[:need]
            yield idx, t.reshape(-1, t.shape[-1]).cpu().numpy().astype(np.float32)
            # clone(), not a slice: a view would pin the whole merged tensor —
            # the very allocation this is trying to release. The leftover is
            # under one mini-batch, so the copy is cheap.
            rest = merged[need:]
            buf = [rest.clone()] if int(rest.shape[0]) else []
            buf_n = int(buf[0].shape[0]) if buf else 0
            idx += 1
            del merged, t, rest


@torch.no_grad()
def extract_window_tokens(
    model: torch.nn.Module,
    window_bgr: np.ndarray,
    device: str,
    layers: list[int] | None = None,
) -> np.ndarray:
    """Patch tokens for one window. Output shape: [Hp, Wp, D_out].

    layers=None       -> last block only,           D_out = D
    layers=[9, 11]    -> concat of the listed blocks, D_out = D * len(layers)
    """
    x = _normalize_window(window_bgr).unsqueeze(0).to(device)
    side = WINDOW_SIZE // DINO_PATCH
    if layers is None:
        out = model.forward_features(x)
        tokens = out["x_norm_patchtokens"].squeeze(0)
        return tokens.view(side, side, -1).cpu().numpy()
    layer_outs = model.get_intermediate_layers(x, n=layers, return_class_token=False)
    cat = torch.cat([t.squeeze(0).view(side, side, -1) for t in layer_outs], dim=-1)
    return cat.cpu().numpy()


@torch.no_grad()
def extract_windows_tokens_batched(
    model: torch.nn.Module,
    windows_bgr: list[np.ndarray],
    device: str,
    layers: list[int] | None = None,
    max_batch: int = 32,
    keep_on_device: bool = True,
) -> torch.Tensor:
    """Batched patch-token extraction. Returns a tensor of shape [N, Hp, Wp, D_out].

    A single forward kernel-launch handles up to `max_batch` windows; larger
    inputs are processed in mini-batches. The batched path is markedly faster
    than per-window forwards because CUDA launch overhead dominates the small
    per-window work for large images.

    The staged INPUTS are bounded by ``max_batch``, so device memory for the
    forward does not scale with image size: roughly
    ``max_batch * 3.2 MB`` plus the model.

    The returned tokens are NOT bounded — they are every window's, which is the
    point of the call. Where they live depends on ``keep_on_device``, and either
    way the total is ``n_windows * 4.2 MB`` (fp32) with a second copy alive at
    the final concatenation. For a 24-image group of 24 MP images that is tens
    of GB on the host, so the teach path uses
    :func:`iter_windows_tokens_batched` and consumes mini-batches as they
    arrive instead of calling this (2026-07-31).

    Args:
        keep_on_device: Return the tokens on ``device`` (default). Scoring
            wants that, because it goes straight into a distance computation.
            Teach ends up in host memory regardless, so it passes False and
            the tokens never accumulate in VRAM.

    Input is cast to the model's parameter dtype (fp16 / fp32) so an
    fp16-converted model works without dtype-mismatch errors.
    """
    chunks = list(
        iter_windows_tokens_batched(
            model, windows_bgr, device, layers=layers, max_batch=max_batch,
            keep_on_device=keep_on_device,
        )
    )
    if not chunks:
        return torch.empty(0, device=device)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def iter_windows_tokens_batched(
    model: torch.nn.Module,
    windows_bgr: list[np.ndarray],
    device: str,
    layers: list[int] | None = None,
    max_batch: int = 32,
    keep_on_device: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield one token tensor per mini-batch instead of one concatenated result.

    Same forwards, same order, same OOM back-off as
    :func:`extract_windows_tokens_batched` — it is now a thin ``torch.cat`` over
    this. Callers that can consume incrementally avoid both the accumulator and
    the concatenation copy.
    """
    if not windows_bgr:
        return
    side = WINDOW_SIZE // DINO_PATCH
    # ``next(model.parameters())`` raises StopIteration when the module
    # has no torch parameters — the OpenVINO backbone is one such case
    # (the IR runtime owns its own weights and exposes nothing to the
    # autograd graph). In an async FastAPI handler that bare StopIteration
    # gets promoted to ``RuntimeError: coroutine raised StopIteration``
    # by PEP 479 and surfaces as a 500. Use the sentinel-default form and
    # fall back to fp32, which is the input dtype the OV plugin expects
    # regardless of the IR's internal weight precision.
    first_param = next(iter(model.parameters()), None)
    target_dtype = first_param.dtype if first_param is not None else torch.float32
    # Stage each mini-batch to the device separately. Stacking every window
    # first put the whole set in VRAM before a single forward ran, and
    # ``max_batch`` only ever bounded the forward: one 518x518x3 window is
    # 3.2 MB in fp32, so a batch teach of large images reached thousands of
    # windows and asked for 11 GiB in one allocation. Observed as
    # ``OutOfMemoryError: Tried to allocate 11.23 GiB`` on a 24 GB card.
    #
    # A bound computed up front is still a guess: another process can take
    # the card between two batches (this machine runs a sibling service on
    # the same GPU), and a forward's peak is not analytically predictable the
    # way the distance matrix in ``safe_cdist_chunk`` is. So halve on OOM and
    # retry the same window range. The smaller size sticks — a card that just
    # ran out is not one to push again mid-job.
    batch_size = max(1, int(max_batch))
    start = 0
    while start < len(windows_bgr):
        # The yield lives OUTSIDE the try: suspending inside it would put the
        # consumer's exceptions — and its GeneratorExit on early close — through
        # the OOM back-off handler below.
        out: torch.Tensor | None = None
        try:
            batch = torch.stack(
                [_normalize_window(w) for w in windows_bgr[start : start + batch_size]]
            ).to(device, dtype=target_dtype)
            if layers is None:
                tok = model.forward_features(batch)["x_norm_patchtokens"]
                tok = tok.view(batch.shape[0], side, side, -1)
            else:
                layer_outs = model.get_intermediate_layers(batch, n=layers, return_class_token=False)
                cat = torch.cat(layer_outs, dim=-1)  # [B, side*side, D*len(layers)]
                tok = cat.view(batch.shape[0], side, side, -1)
            # The output is the other half of the same problem: tokens are
            # 4.2 MB per window, so holding every window's tokens on the
            # device costs more than the inputs did. Callers that end up in
            # host memory anyway (teach) pass keep_on_device=False and pay
            # nothing for it.
            out = tok if keep_on_device else tok.cpu()
            del batch, tok
            start += batch_size
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            if not is_cuda_oom(exc):
                raise
            _release_cuda_cache(device)
            if batch_size == 1:
                raise RuntimeError(
                    "GPU ran out of memory extracting features, even one window "
                    "at a time. Free VRAM on the device (another process may be "
                    "holding it), use a smaller image, or switch the inference "
                    "device to CPU in Settings."
                ) from exc
            batch_size = max(1, batch_size // 2)
            logger.warning(
                "CUDA OOM during feature extraction; retrying with batch=%d "
                "(window %d of %d)", batch_size, start, len(windows_bgr),
            )
            continue  # nothing to hand out — retry the same window range
        yield out
        del out
