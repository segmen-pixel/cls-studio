# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Core ML export, on a machine that cannot do Core ML.

Almost all of this is checkable without coremltools, which matters because the
server it runs on is Windows and coremltools publishes no Windows build. What
is checked here: that a name arriving from bank metadata cannot send torch.hub
after a backbone nobody asked for, that the wrapper hands back the layout the
server uses, and that an unavailable platform says so instead of producing an
artifact nobody could verify.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from clscore.coreml_export import (
    CoreMLUnavailable,
    coreml_availability,
    export_encoder,
    trace_encoder,
    traceable_encoder,
)

NON_COMMERCIAL = ["cell_dino_hpa_vitl16", "cell_dino_cp_vits8", "xray_dino_vitl16"]


def test_a_backbone_nobody_asked_for_is_refused_before_the_network(monkeypatch):
    """The hub repo carries backbones under non-commercial terms alongside the
    ones this project uses. A model name comes out of a bank's metadata, which
    is data, so the refusal has to land before torch.hub is given the chance
    to go and fetch one."""
    def explode(*a, **k):
        raise AssertionError("torch.hub was called for a name that should have been refused")

    monkeypatch.setattr(torch.hub, "load", explode)
    for name in NON_COMMERCIAL + ["resnet50", "", "../etc/passwd"]:
        with pytest.raises(CoreMLUnavailable, match="not an exportable backbone"):
            trace_encoder(name, 518, 14)


def test_availability_reports_convert_and_predict_separately():
    """Converting without being able to run the result yields an artifact
    nobody has checked, so the two capabilities are not one flag."""
    cap = coreml_availability()
    assert set(cap) >= {"available", "convert", "predict", "reason"}
    if not cap["available"]:
        assert cap["reason"]


def test_export_refuses_rather_than_producing_something_unverified(tmp_path):
    if coreml_availability()["available"]:
        pytest.skip("this machine can convert; the refusal path is not the one under test")
    with pytest.raises(CoreMLUnavailable):
        export_encoder(model_name="dinov2_vitb14", window=518, patch=14,
                       dest=tmp_path / "enc.mlpackage")


class _FakeBackbone(torch.nn.Module):
    """Stands in for DINOv2: same call surface, no download."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.proj = torch.nn.Linear(3 * 14 * 14, dim)

    def _tokens(self, x):
        b = x.shape[0]
        side = x.shape[-1] // 14
        patches = x.unfold(2, 14, 14).unfold(3, 14, 14)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, side * side, -1)
        return self.proj(patches)

    def forward_features(self, x):
        return {"x_norm_patchtokens": self._tokens(x)}

    def get_intermediate_layers(self, x, n, return_class_token=False):
        t = self._tokens(x)
        return [t for _ in (n if isinstance(n, list) else range(n))]


def test_the_wrapper_returns_the_layout_the_server_uses():
    """Patch grid first, channels last -- the shape extract_window_tokens
    returns, so parity is a direct comparison and not a comparison modulo a
    reshape someone has to get right in two languages."""
    enc = traceable_encoder(_FakeBackbone(dim=8), side=4, layers=None)
    out = enc(torch.zeros(1, 3, 56, 56))
    assert tuple(out.shape) == (1, 4, 4, 8)


def test_concatenated_layers_widen_the_last_axis():
    enc = traceable_encoder(_FakeBackbone(dim=8), side=4, layers=[9, 11])
    assert tuple(enc(torch.zeros(1, 3, 56, 56)).shape) == (1, 4, 4, 16)


def test_the_wrapper_survives_tracing_and_keeps_its_numbers():
    """Whatever the conversion does afterwards, the traced graph is what it
    starts from, so it has to agree with the eager module first."""
    enc = traceable_encoder(_FakeBackbone(dim=8), side=4, layers=None)
    x = torch.rand(1, 3, 56, 56)
    with torch.no_grad():
        eager = enc(x).numpy()
        traced = torch.jit.trace(enc, torch.zeros_like(x), strict=False)(x).numpy()
    assert np.array_equal(eager, traced)
