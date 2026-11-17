# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Shared backbone-bring-up sequence for the API server.

The order ``load -> half -> compile -> warmup`` matters: ``torch.compile``
in ``reduce-overhead`` mode bakes parameter dtypes into its CUDA graph, so
the ``.half()`` swap must happen *before* compilation. Warmup pays the
10-30s Triton autotune cost at startup instead of the first request.

Server startup and any live model hot-swap should both go through
:func:`prepare_model` so the two flows cannot silently drift.
"""

from __future__ import annotations

import logging
import time

import torch

from .backends import DEFAULT_BACKEND, SUPPORTED_BACKENDS, BackendName
from .feature_extractor import load_backbone, maybe_compile, warmup_model

logger = logging.getLogger(__name__)

__all__ = ["prepare_model"]


def prepare_model(
    name: str,
    device: str,
    dtype: torch.dtype,
    backend: BackendName = DEFAULT_BACKEND,
) -> torch.nn.Module:
    """Load a backbone variant and apply the full server-side bring-up.

    Steps, in order:

    1. ``load_backbone(name, device=device)`` — dispatches to DINOv2 or
       BG-aware loader based on the variant name.
    2. ``.half()`` if ``dtype == float16`` and CUDA is the target device
       (CPU fp16 is not faster on most consumer CPUs).
    3. :func:`maybe_compile` to apply ``torch.compile`` when supported.
    4. :func:`warmup_model` on CUDA to absorb the 10-30s autotune.

    Args:
        name: Backbone variant name (DINOv2 hub model or BG-aware key).
        device: Torch device string (``"cuda:0"``, ``"cpu"``, ...).
        dtype: Compute dtype. ``torch.float16`` only takes effect on CUDA.
        backend: Which inference backend to use. ``"torch"`` (default) runs
            the PyTorch model directly. ``"openvino"`` loads an exported
            IR through the OpenVINO runtime — useful for CPU / Intel iGPU
            / NPU deployments without CUDA.

    Returns:
        A ready-to-serve module exposing
        ``forward_features(x)["x_norm_patchtokens"]``.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {SUPPORTED_BACKENDS}"
        )
    if backend == "openvino":
        # Defer the import so torch-only users never pay for the optional
        # openvino dep. The wrapper raises a clean ImportError pointing
        # at ``pip install cls-studio[openvino]`` if the package is missing.
        from .backends.openvino_backbone import load_openvino_backbone
        model = load_openvino_backbone(name, device=device)
        logger.info("backend=openvino model=%r", model)
        return model
    model = load_backbone(name, device=device)
    if dtype == torch.float16 and device.startswith("cuda"):
        model = model.half()
        logger.info("forward dtype: fp16")
    else:
        logger.info("forward dtype: fp32")
    # Compile + warm Triton/CUDA-graph caches *after* dtype is finalised so
    # the recorded CUDA graph matches the live parameter dtype.
    model = maybe_compile(model, device)
    if device.startswith("cuda"):
        t0 = time.perf_counter()
        warmup_model(model, device)
        logger.info("warmup took %.1fs", time.perf_counter() - t0)
    return model
