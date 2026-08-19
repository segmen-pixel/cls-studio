# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Export the frozen encoder so a phone can put queries in the bank's space.

The edge package ships the bank and a description of the encoder that built
it, and states outright that a device which cannot reproduce that encoder
cannot use the package. This closes that gap for Apple hardware.

What crosses the boundary is deliberately small: one window in, one patch-grid
of features out. Tiling, quantisation and the nearest-neighbour search stay on
the device, where the package already describes them. Reproducing the sliding
window inside a traced graph would mean reproducing reflect padding and the
edge-anchored last window bit for bit, for arbitrary image sizes, and there is
no parity argument that survives that.

The input is a pre-normalised tensor rather than an image. Core ML's ImageType
applies one scalar scale with per-channel bias, so the three ImageNet standard
deviations cannot be expressed in it; folding them to an average is a
systematic error that clscore.coreml_parity exists to catch, and it is easier
not to introduce it. The host normalises -- the iOS side already carries the
same per-channel constants -- and hands over a float32 MLMultiArray.

Nothing here is imported at module load: coremltools has no Windows build, and
the server this usually runs on is Windows. Callers get a clear failure at the
point of use instead of an import error at startup.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CoreMLUnavailable", "coreml_availability", "export_encoder", "traceable_encoder"]


class CoreMLUnavailable(RuntimeError):
    """Conversion cannot run here, with the reason the caller should relay."""


def coreml_availability() -> dict[str, Any]:
    """Whether this machine can convert, and predict, and why not.

    Reported rather than inferred: converting without being able to predict
    produces an artifact nobody has checked, which is not a deliverable.
    """
    import platform

    system = platform.system()
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "convert": False,
            "predict": False,
            "reason": (
                "coremltools is not installed"
                + ("; it publishes no Windows build" if system == "Windows" else "")
            ),
        }
    macos = system == "Darwin"
    return {
        "available": macos,
        "convert": True,
        "predict": macos,
        "reason": None if macos else (
            f"coremltools cannot run a model on {system}, so a converted "
            "encoder could not be checked against the bank it must match"
        ),
    }


def _check_model_name(model_name: str) -> None:
    """Refuse anything that is not a plain DINOv2 backbone, before any fetch.

    The hub repo also carries backbones released under non-commercial terms.
    Nothing in cls-studio asks for them, but a name arriving from a bank's
    metadata is data, and the check has to happen before torch.hub is given a
    chance to go and get one.
    """
    from .feature_extractor import DINO_MODELS

    if model_name not in DINO_MODELS:
        raise CoreMLUnavailable(
            f"{model_name!r} is not an exportable backbone; expected one of "
            + ", ".join(sorted(DINO_MODELS))
        )


def traceable_encoder(backbone: Any, side: int, layers: list[int] | None = None) -> Any:
    """Wrap the backbone so one window in gives one patch grid out.

    The output layout matches what the server's own extractor returns, so
    parity is a direct comparison rather than a comparison modulo a reshape
    someone has to get right twice.
    """
    import torch

    class _WindowEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.side = int(side)
            self.layers = layers

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.layers is None:
                tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
            else:
                outs = self.backbone.get_intermediate_layers(
                    x, n=self.layers, return_class_token=False
                )
                tokens = torch.cat(outs, dim=-1)
            return tokens.reshape(tokens.shape[0], self.side, self.side, -1)

    return _WindowEncoder().eval()


def _load_backbone(model_name: str) -> Any:
    """Allowlisted load on CPU, with the xformers path disabled."""
    _check_model_name(model_name)
    # MemEffAttention takes the xformers path when it is importable, and that
    # path survives neither tracing nor export.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    from .feature_extractor import load_dinov2

    return load_dinov2(model_name, device="cpu")


def trace_encoder(model_name: str, window: int, patch: int, layers: list[int] | None = None) -> Any:
    """Load the backbone on CPU and return a traced graph plus its output shape.

    Runs everywhere torch runs, which is the point: the half of the export
    that can go wrong silently is checkable on the machine the bank was built
    on, before anyone opens a Mac.
    """
    import torch

    wrapper = traceable_encoder(_load_backbone(model_name), side=window // patch, layers=layers)
    example = torch.zeros(1, 3, window, window, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
        shape = tuple(int(v) for v in traced(example).shape)
    logger.info("traced %s: %s -> %s", model_name, tuple(example.shape), shape)
    return traced, shape


def _sample_windows(window: int, count: int, supplied: list | None) -> list:
    """Windows to probe parity with.

    Real crops from the bank's own images are preferred: they exercise the
    activation range the encoder actually sees. Deterministic noise is the
    fallback, and it is a weaker probe -- it says the graph computes the same
    thing, not that it does so over the distribution that matters.
    """
    import numpy as np

    if supplied:
        return list(supplied)[:count]
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(window, window, 3), dtype=np.uint8) for _ in range(count)]


def export_encoder(
    *,
    model_name: str,
    window: int,
    patch: int,
    dest: os.PathLike[str] | str,
    layers: list[int] | None = None,
    bank: Any = None,
    sample_windows: list | None = None,
    samples: int = 4,
    precision: str = "float32",
) -> dict[str, Any]:
    """Convert the encoder, check it against the bank, and record both.

    Returns the artifact record. Raises :class:`CoreMLUnavailable` when this
    machine cannot both convert and predict: an unchecked conversion is not
    something to hand anyone, so it is not produced at all.

    ``precision`` defaults to float32 on measurement, not on caution. Converted
    at float16 the same graph drifts scores by 0.099% on average and 1.1% at
    worst, against 0.00006% at float32 -- fine for a classifier, and the same
    order as a real preprocessing bug for a bank scored by distance. float16
    halves the artifact (173 MB against 347 MB for vitb14) and is a reasonable
    trade to make deliberately, so it is offered; it is not the default, and
    it does not pass the gate on its own numbers.
    """
    import json
    from pathlib import Path

    import numpy as np
    import torch

    cap = coreml_availability()
    if not cap["available"]:
        raise CoreMLUnavailable(cap["reason"] or "Core ML export is unavailable here")

    import coremltools as ct

    from .coreml_parity import compare_features
    from .preprocess import normalize_window, preprocess_spec

    dest = Path(dest)
    traced, out_shape = trace_encoder(model_name, window, patch, layers=layers)

    # Conversion goes through torch.export, not the TorchScript graph above.
    # coremltools 9.0 cannot take this model as TorchScript: DINOv2 reads its
    # own shapes (vision_transformer.py "B, nc, w, h = x.shape",
    # attention.py "C // self.num_heads") and the frontend hits those as
    # aten::Int over a value it cannot reduce to a scalar, raising
    # "only 0-dimensional arrays can be converted to Python scalars" 22 ops in.
    # torch.jit.freeze folds the graph from 979 nodes to 556 without removing
    # them. The exported program has no such nodes. run_decompositions is not
    # optional -- without it the dialect is TRAINING and convert refuses.
    exported = torch.export.export(
        traceable_encoder(_load_backbone(model_name), window // patch, layers),
        (torch.zeros(1, 3, window, window, dtype=torch.float32),),
    ).run_decompositions({})

    mlmodel = ct.convert(
        exported,
        compute_precision=(
            ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
        ),
        minimum_deployment_target=ct.target.iOS17,
    )
    # torch.export names the graph's own placeholders ("x", "view_25"); the
    # package documents stable names, so say them here rather than in prose.
    spec = mlmodel.get_spec()
    ct.utils.rename_feature(spec, spec.description.input[0].name, "window")
    ct.utils.rename_feature(spec, spec.description.output[0].name, "features")
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)
    mlmodel.short_description = (
        "cls-studio patch encoder. Input is one pre-normalised window, "
        "output is a patch-feature grid to score against the bank by distance."
    )
    mlmodel.save(str(dest))

    # Parity: the traced graph is the server's own numbers (verified bit-exact
    # against extract_window_tokens), so it is the reference the converted
    # model has to reproduce.
    wins = _sample_windows(window, samples, sample_windows)
    ref, got = [], []
    for w in wins:
        x = normalize_window(w)[None, ...]
        with torch.no_grad():
            ref.append(traced(torch.from_numpy(x)).numpy().reshape(-1, out_shape[-1]))
        pred = mlmodel.predict({"window": x})["features"]
        got.append(np.asarray(pred).reshape(-1, out_shape[-1]))
    parity = compare_features(
        np.concatenate(ref), np.concatenate(got),
        bank=None if bank is None else np.asarray(bank),
    )

    artifact = {
        "model": model_name,
        "window": int(window),
        "patch": int(patch),
        "layers": layers,
        "output_shape": list(out_shape),
        "compute_precision": precision,
        "input": {"name": "window", "shape": [1, 3, window, window], "dtype": "float32"},
        "output": {"name": "features", "layout": "NHWC"},
        "preprocess": preprocess_spec(),
        "parity": parity,
        "windows_probed": len(wins),
        "windows_were_real_crops": bool(sample_windows),
    }
    Path(str(dest) + ".json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not parity["passed"]:
        raise CoreMLUnavailable(
            "converted encoder failed the parity gate against the bank it must "
            f"match: {parity.get('reason')} ({json.dumps({k: v for k, v in parity.items() if k != 'reason'})})"
        )
    return artifact
