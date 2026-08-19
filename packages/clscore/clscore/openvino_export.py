# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Export the frozen encoder as OpenVINO IR, for CPU and Intel iGPU.

The Core ML path and this one answer the same question for different
hardware, and they are not symmetric.

Core ML has to define its own contract, because the reader is an app outside
this repo: the package says what shape comes out and in what layout. Here the
reader already exists -- ``clscore.backends.openvino_backbone`` loads an IR and
re-packs its single output as ``x_norm_patchtokens`` -- so the export has to
match a contract that was written first. That means tokens as
``forward_features`` returns them, ``[B, N, D]``, not the patch grid the Core
ML export emits, and a filename of ``<backbone>__bs<N>_<precision>.xml``.

The other asymmetry is better news: OpenVINO converts and runs on the same
Windows box the banks are built on, so the parity gate is not a thing that has
to happen somewhere else. An IR is checked against the bank it must match
before anyone is handed it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["OpenVINOUnavailable", "openvino_availability", "export_ir", "ir_filename"]


class OpenVINOUnavailable(RuntimeError):
    """Conversion cannot run here, with the reason the caller should relay."""


def openvino_availability() -> dict[str, Any]:
    """Whether this machine can convert an IR, and run one.

    Unlike Core ML these are the same answer everywhere OpenVINO installs, but
    the shape is kept identical so the UI has one thing to read.
    """
    try:
        import openvino
    except ImportError:
        return {
            "available": False,
            "convert": False,
            "predict": False,
            "reason": (
                "openvino is not installed; add it with the installer's "
                "--with-openvino option"
            ),
        }
    return {"available": True, "convert": True, "predict": True,
            "version": openvino.__version__, "reason": None}


def ir_filename(backbone: str, batch_size: int = 1, precision: str = "fp32") -> str:
    """The name the loader looks for. Kept here so both ends agree."""
    return f"{backbone}__bs{batch_size}_{precision}.xml"


def _token_encoder(backbone: Any) -> Any:
    """One window in, ``[B, N, D]`` tokens out -- the loader's own shape."""
    import torch

    class _TokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone.forward_features(x)["x_norm_patchtokens"]

    return _TokenEncoder().eval()


def export_ir(
    *,
    model_name: str,
    window: int,
    patch: int,
    dest_dir: os.PathLike[str] | str,
    layers: list[int] | None = None,
    bank: Any = None,
    sample_windows: list | None = None,
    samples: int = 4,
    precision: str = "fp32",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Convert, check against the bank, and write the .xml/.bin pair.

    Raises :class:`OpenVINOUnavailable` if the conversion cannot be verified
    here -- an IR nobody has compared to the bank is not a deliverable, and on
    this platform there is no excuse for skipping the comparison.
    """
    import json
    from pathlib import Path

    import numpy as np
    import openvino as ov
    import torch

    from .coreml_export import _load_backbone  # allowlist + XFORMERS_DISABLED
    from .coreml_parity import compare_features
    from .preprocess import normalize_window, preprocess_spec

    cap = openvino_availability()
    if not cap["available"]:
        raise OpenVINOUnavailable(cap["reason"] or "OpenVINO export is unavailable here")
    if precision not in ("fp32", "fp16"):
        raise OpenVINOUnavailable("precision must be fp32 or fp16")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    encoder = _token_encoder(_load_backbone(model_name))
    example = torch.zeros(batch_size, 3, window, window, dtype=torch.float32)

    model = ov.convert_model(encoder, example_input=example)
    # convert_model leaves the spatial dims dynamic. The bank's rows only exist
    # in one geometry, so a graph that will accept any size is a graph that can
    # be fed the wrong one and answer without complaint.
    model.reshape({0: ov.PartialShape([batch_size, 3, window, window])})

    xml = dest_dir / ir_filename(model_name, batch_size, precision)
    ov.save_model(model, str(xml), compress_to_fp16=(precision == "fp16"))

    # Parity against the torch path, scored on the real bank.
    from .feature_extractor import extract_window_tokens

    core = ov.Core()
    # Compile the file that was just written, not the object it came from.
    # compress_to_fp16 changes the artifact and not the in-memory model, so
    # gating the latter would measure something nobody is going to run -- and
    # would have reported the fp32 numbers for an fp16 export, which is
    # exactly the shape of mistake this gate exists to catch.
    compiled = core.compile_model(core.read_model(str(xml)), "CPU")
    out_port = compiled.output(0)
    torch_backbone = encoder.backbone
    wins = sample_windows[:samples] if sample_windows else None
    if not wins:
        rng = np.random.default_rng(0)
        wins = [rng.integers(0, 256, size=(window, window, 3), dtype=np.uint8) for _ in range(samples)]

    dim_ref, ref, got = None, [], []
    for w in wins:
        with torch.no_grad():
            t = extract_window_tokens(torch_backbone, w, "cpu", layers=layers)
        dim_ref = t.shape[-1]
        ref.append(t.reshape(-1, dim_ref))
        x = normalize_window(w)[None].astype(np.float32)
        got.append(np.asarray(compiled([x])[out_port]).reshape(-1, dim_ref))
    parity = compare_features(
        np.concatenate(ref), np.concatenate(got),
        bank=None if bank is None else np.asarray(bank),
    )

    artifact = {
        "backend": "openvino",
        "model": model_name,
        "window": int(window),
        "patch": int(patch),
        "layers": layers,
        "batch_size": int(batch_size),
        "precision": precision,
        "ir": xml.name,
        "weights": xml.with_suffix(".bin").name,
        "input": {"shape": [batch_size, 3, window, window], "dtype": "float32"},
        "output": {"shape": [batch_size, (window // patch) ** 2, dim_ref], "layout": "BND"},
        "openvino": cap.get("version"),
        "preprocess": preprocess_spec(),
        "parity": parity,
        "windows_probed": len(wins),
        "windows_were_real_crops": bool(sample_windows),
    }
    (dest_dir / (xml.stem + ".json")).write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    if not parity["passed"]:
        raise OpenVINOUnavailable(
            "converted IR failed the parity gate against the bank it must match: "
            + json.dumps({k: v for k, v in parity.items() if k != "reason"})
        )
    logger.info("openvino IR written: %s (%s)", xml.name, precision)
    return artifact
