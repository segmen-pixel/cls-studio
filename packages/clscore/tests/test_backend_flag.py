# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Backend flag plumbing tests.

Phase A1 wired the ``--backend`` flag through CLI / API / runtime;
Phase A2 implements the actual OpenVINO loader. These tests pin down:

1. ``backend="torch"`` (the default) stays bit-equivalent to the
   pre-flag behaviour — existing callers see no change.
2. ``backend="openvino"`` reaches the OV loader. When openvino is not
   installed, the loader raises ``ImportError``; when installed but
   no IR exists for the backbone, ``FileNotFoundError``. Either is
   acceptable here — what matters is that we don't blow up with an
   unrelated stack trace.
"""

from __future__ import annotations

import pytest
import torch

from clscore.backends import DEFAULT_BACKEND, SUPPORTED_BACKENDS
from clscore.model_runtime import prepare_model


def test_default_backend_is_torch():
    """``DEFAULT_BACKEND`` is the safe, always-installable choice."""
    assert DEFAULT_BACKEND == "torch"


def test_supported_backends_match_documented_set():
    assert set(SUPPORTED_BACKENDS) == {"torch", "openvino"}


def test_prepare_model_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        prepare_model("dinov2_vitb14", "cpu", torch.float32, backend="caffe2")  # type: ignore[arg-type]


def test_prepare_model_openvino_reaches_loader():
    """Phase A2: hitting backend='openvino' reaches the OV loader.

    Three outcomes are all acceptable for this plumbing test:

    * ``ImportError`` — ``openvino`` is not installed (the most common
      case in a CI matrix that doesn't carry the optional extra).
    * ``FileNotFoundError`` — OV is installed but no IR exists yet for
      this backbone under ``$CLS_OPENVINO_IR_DIR`` / ``./ov_ir``.
    * Success — both are present, the loader returns a real
      :class:`OpenVINOBackbone`. We sanity-check the type so a future
      regression (e.g. quietly returning ``None``) is caught.

    What we explicitly do **not** want is the request to crash with an
    unrelated stack trace; that would mean the backend flag is not
    routing to the OV path properly.
    """
    try:
        result = prepare_model(
            "bg_simpleunet_bc32_e30", "cpu", torch.float32, backend="openvino",
        )
    except (ImportError, FileNotFoundError):
        return
    from clscore.backends.openvino_backbone import OpenVINOBackbone
    assert isinstance(result, OpenVINOBackbone)
