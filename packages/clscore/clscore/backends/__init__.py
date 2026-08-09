# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Pluggable inference backends.

cls-studio defaults to a PyTorch backbone (DINOv2 via ``torch.hub``, or any of
the vendored BG-aware distilled student models). For edge / CPU-only
deployments — particularly Intel Iris Xe / Arc / NPU hardware where CUDA is
unavailable — an OpenVINO backend can be selected via
``--backend openvino`` on the CLI / API server.

The backend layer is intentionally narrow: a backend just has to expose a
:class:`Backbone` (a callable with the same input contract as the torch
backbone). All scoring / cdist code remains backend-agnostic; only the
backbone forward step is swapped.

Currently registered backends
-----------------------------
- ``torch`` (default): the PyTorch backbone prepared by
  :func:`clscore.model_runtime.prepare_model`.
- ``openvino``: runs an exported OpenVINO IR on CPU / Iris Xe / Arc / NPU
  (see :mod:`clscore.backends.openvino_backbone`). Requires the optional
  OpenVINO runtime: ``pip install clscore[openvino]``.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["BackendName", "SUPPORTED_BACKENDS", "DEFAULT_BACKEND"]

BackendName = Literal["torch", "openvino"]

SUPPORTED_BACKENDS: tuple[str, ...] = ("torch", "openvino")
DEFAULT_BACKEND: BackendName = "torch"
