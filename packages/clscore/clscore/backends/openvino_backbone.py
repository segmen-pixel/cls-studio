# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""OpenVINO IR loader wrapped in a torch.nn.Module-compatible facade.

The rest of cls-studio (``scoring``, ``api/app``, ``runtime``) treats the
backbone as a torch ``nn.Module`` exposing ``forward_features(x) -> {
"x_norm_patchtokens": tokens}``. This wrapper preserves that contract
while delegating the actual forward to an OpenVINO ``CompiledModel`` so
no caller has to branch on the backend.

Inputs land as torch.Tensors on whatever device cls-studio picked for
the rest of the pipeline (typically ``cpu`` when the OV backend is in
use; ``cuda:*`` is allowed but the round-trip through numpy is wasted
work). Outputs come back as torch.Tensors so downstream cdist / per-
label scoring stays bit-exact.

The IR file path comes from the env var ``CLS_OPENVINO_IR_DIR``,
or from the ``ir_dir`` keyword to :func:`load_openvino_backbone`. IRs
are exported ahead of time with OpenVINO's own converter and must be
named ``<backbone>__bs<N>_<precision>.xml`` (e.g.
``bg_simpleunet_bc32_e30__bs1_fp16.xml``), with the matching ``.bin``
weights file alongside it in the same directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["OpenVINOBackbone", "load_openvino_backbone", "OV_DEVICE_MAP"]


# cls-studio uses torch device strings (``cpu``, ``cuda:0``); OpenVINO
# names its plugins ``CPU``, ``GPU``, ``NPU``, ``AUTO``. The map below is
# used when the operator passes a torch-style string to ``--device`` but
# selected ``--backend openvino``; any value that is already an OV plugin
# name passes through unchanged.
OV_DEVICE_MAP = {
    "cpu": "CPU",
    "auto": "AUTO",
    "cuda": "GPU",
    "cuda:0": "GPU",
    "cuda:1": "GPU",
}


def _resolve_ov_device(device: str) -> str:
    """Translate a torch device string (or OV plugin name) to OV plugin name.

    ``cuda:N`` maps to ``GPU`` because OpenVINO's Intel GPU plugin
    ignores the index — we have at most one iGPU per box anyway.
    Unknown strings are passed through verbatim so a user can specify
    ``MULTI:CPU,GPU`` or other exotic OV device specs without us
    standing in the way.
    """
    key = device.strip().lower()
    return OV_DEVICE_MAP.get(key, device)


def _ir_filename(backbone: str, batch_size: int = 1, precision: str = "fp16") -> str:
    return f"{backbone}__bs{batch_size}_{precision}.xml"


def _port_label(port) -> str:
    """Best-effort human label for an OpenVINO port; never raises.

    ``ConstOutput.any_name`` raises when the underlying tensor has no
    names set — which happens for IRs produced by the torch frontend
    without an explicit name pin. Fall back to the static shape so
    the diagnostic still tells the operator what they're looking at.
    """
    try:
        return port.any_name
    except RuntimeError:
        try:
            return f"<unnamed shape={list(port.shape)}>"
        except Exception:
            return "<unnamed>"


def _find_ir(backbone: str, ir_dir: Path) -> Path:
    """Locate the IR for ``backbone`` under ``ir_dir``.

    Preference order: int8 > fp16 > fp32. INT8 is picked first because
    it's strictly the lowest-latency variant on AVX2 / DP4a CPUs (the
    common Intel-edge target); fp16 / fp32 are the conventional
    accuracy-focused fallbacks for an accuracy-aware deploy.

    Raises ``FileNotFoundError`` listing every candidate so the
    operator immediately sees which variant to export.
    """
    candidates = [
        ir_dir / _ir_filename(backbone, 1, "int8"),
        ir_dir / _ir_filename(backbone, 1, "fp16"),
        ir_dir / _ir_filename(backbone, 1, "fp32"),
    ]
    for c in candidates:
        if c.exists():
            return c
    expected = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"no OpenVINO IR found for backbone {backbone!r} under {ir_dir}. "
        f"Expected one of:\n  {expected}\n"
        "Export the backbone to OpenVINO IR first and place the .xml file "
        "(with its matching .bin weights) under that directory, or point "
        "CLS_OPENVINO_IR_DIR at the directory that holds it."
    )


class OpenVINOBackbone(nn.Module):
    """torch.nn.Module facade over an OpenVINO CompiledModel.

    Exposes the subset of the DINOv2 hub API that cls-studio consumes:
    ``forward_features`` and ``get_intermediate_layers``. The class
    inherits from ``nn.Module`` only for duck-typing — there are no
    trainable parameters, ``state_dict`` is empty, and ``.to(device)``
    is a no-op (the OV plugin owns its own memory).
    """

    def __init__(
        self,
        ir_path: Path,
        ov_device: str = "CPU",
        target_dim: int = 768,
        grid: int = 37,
    ):
        super().__init__()
        try:
            from openvino import Core
        except ImportError as exc:
            raise ImportError(
                "OpenVINO is not installed. Install with "
                "'pip install cls-studio[openvino]' to use the openvino backend."
            ) from exc

        self._ir_path = ir_path
        self._ov_device = ov_device
        self.target_dim = target_dim
        self.grid = grid

        core = Core()
        model = core.read_model(str(ir_path))
        self._compiled = core.compile_model(model, ov_device)
        # The exporter does not pin input / output names because the
        # OpenVINO torch frontend names them after the traced leaf op
        # rather than the function arg. We use port index 0 instead,
        # which is well-defined and matches the single-input /
        # single-output graph the exporter produces.
        self._input_port = self._compiled.input(0)
        self._output_port = self._compiled.output(0)
        logger.info(
            "OpenVINO backbone ready: ir=%s device=%s inputs=%s outputs=%s",
            ir_path.name, ov_device,
            [_port_label(p) for p in self._compiled.inputs],
            [_port_label(p) for p in self._compiled.outputs],
        )

    # ---- DINOv2-compatible forward ---------------------------------------

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the IR on ``x`` and re-pack the result as the DINOv2 dict.

        ``x`` must be a torch.Tensor of shape ``[B, 3, H, W]`` (typically
        ``[1, 3, 518, 518]`` for cls-studio). We round-trip via numpy
        because the OpenVINO Python API is built around it; the copy is
        negligible compared to the inference itself.
        """
        np_x = x.detach().cpu().contiguous().numpy()
        result = self._compiled([np_x])
        # OpenVINO returns an OVDict keyed by ConstOutput port objects;
        # indexing by the port we recorded at __init__ avoids string
        # name lookup and works regardless of how the frontend named
        # the tensor.
        np_tokens = result[self._output_port]
        tokens = torch.from_numpy(np_tokens)
        return {"x_norm_patchtokens": tokens}

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: list[int] | int = 1,
        return_class_token: bool = False,
    ) -> list[torch.Tensor]:
        """Stub for the DINOv2 multi-layer API.

        BG-aware backbones expose only the final-stage tokens; we mirror
        :class:`BGBackboneWrapper.get_intermediate_layers` by repeating
        the same tensor ``len(n)`` (or ``n``) times. Callers that want
        true intermediate layers fall through to DINOv2 directly.
        """
        out = self.forward_features(x)["x_norm_patchtokens"]
        count = len(n) if isinstance(n, list) else int(n)
        return [out for _ in range(count)]

    # ---- nn.Module overrides ---------------------------------------------

    def half(self) -> OpenVINOBackbone:
        """No-op: precision is baked into the IR, runtime dtype handled by OV."""
        return self

    def eval(self) -> OpenVINOBackbone:
        """No-op: OV CompiledModel has no eval/train mode."""
        return self

    def to(self, *args, **kwargs) -> OpenVINOBackbone:  # noqa: D401, ARG002
        """No-op: OV plugin owns its own memory."""
        return self

    def __repr__(self) -> str:
        return (
            f"OpenVINOBackbone(ir={self._ir_path.name}, "
            f"device={self._ov_device}, grid={self.grid}, dim={self.target_dim})"
        )


def load_openvino_backbone(
    name: str,
    device: str = "cpu",
    ir_dir: str | os.PathLike[str] | None = None,
) -> OpenVINOBackbone:
    """Load the OpenVINO IR for ``name`` and return a torch-compatible wrapper.

    Args:
        name: Backbone name (must be a registered BG-aware backbone; the
            DINOv2 hub models do not have an exported IR yet).
        device: Torch device string (``cpu`` / ``cuda:N``) or OpenVINO
            plugin name (``CPU`` / ``GPU`` / ``NPU`` / ``AUTO``). torch
            strings are mapped via :data:`OV_DEVICE_MAP`.
        ir_dir: Where to look for the ``.xml`` / ``.bin`` pair. Defaults
            to ``$CLS_OPENVINO_IR_DIR``, then ``./ov_ir``.

    Returns:
        An :class:`OpenVINOBackbone` ready to serve.
    """
    if ir_dir is None:
        env = os.environ.get("CLS_OPENVINO_IR_DIR")
        ir_dir = env if env else "./ov_ir"
    ir_dir_path = Path(ir_dir).expanduser().resolve()
    ir_path = _find_ir(name, ir_dir_path)
    ov_device = _resolve_ov_device(device)
    return OpenVINOBackbone(ir_path=ir_path, ov_device=ov_device)
