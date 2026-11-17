# SPDX-License-Identifier: Apache-2.0
"""Tests for the BG-aware backbone wrapper and registry.

These tests run on CPU and do *not* require the distilled checkpoints — the
wrapper is exercised at random init, which is enough to validate the shape
contract (the part cls-studio' bank / scoring code actually depends on).

An optional integration test that loads the real ckpts runs only when
``BG_BACKBONE_CKPT_DIR`` is set and the file exists on disk; see
``test_load_bg_backbone_real_ckpt``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from clscore.bg_backbone import (
    BG_BACKBONES,
    GRID,
    TARGET_DIM,
    BGBackboneSpec,
    BGBackboneWrapper,
    is_bg_backbone,
    load_bg_backbone,
)
from clscore.feature_extractor import BACKBONE_DIMS, load_backbone
from clscore.sw import DINO_PATCH, WINDOW_SIZE


# ---------------------------------------------------------------------------- #
# Constants + registry sanity
# ---------------------------------------------------------------------------- #
def test_grid_and_target_dim_match_cls_window() -> None:
    """The wrapper's output shape must match what bank / scoring expects."""
    assert GRID == WINDOW_SIZE // DINO_PATCH == 37
    assert TARGET_DIM == 768  # matches dinov2_vitb14


def test_registry_nonempty() -> None:
    assert len(BG_BACKBONES) >= 3, "expect at least SimpleUNet bc32/bc128/bc256"


def test_registry_simpleunet_present() -> None:
    """The 3 SimpleUNet entries are the production-recommended set (Apache 2.0)."""
    for name in ("bg_simpleunet_bc32_e30", "bg_simpleunet_bc128_e30", "bg_simpleunet_bc256_e30"):
        assert name in BG_BACKBONES, f"{name} missing from registry"
        spec = BG_BACKBONES[name]
        assert spec.family == "simpleunet"
        assert spec.license == "Apache-2.0"


def test_is_bg_backbone_dispatch() -> None:
    assert is_bg_backbone("bg_simpleunet_bc128_e30") is True
    assert is_bg_backbone("dinov2_vitb14") is False
    assert is_bg_backbone("nonexistent_model") is False


def test_backbone_dims_merged() -> None:
    """BACKBONE_DIMS unifies DINOv2 + BG; every BG entry contributes dim=768."""
    assert "dinov2_vitb14" in BACKBONE_DIMS
    for name in BG_BACKBONES:
        assert BACKBONE_DIMS[name] == TARGET_DIM


# ---------------------------------------------------------------------------- #
# Wrapper shape contract — the kNN code only cares about this part
# ---------------------------------------------------------------------------- #
def _simpleunet_spec_with_tiny_bc() -> BGBackboneSpec:
    """A spec we can instantiate without touching disk.

    The bc value is arbitrary for shape tests — we never load a ckpt. We do
    register a stub ckpt_relpath so any code path that touches ``ckpt_path``
    fails clearly rather than mysteriously.
    """
    return BGBackboneSpec(
        name="bg_simpleunet_bc8_test",
        family="simpleunet",
        ckpt_relpath="nonexistent/bc8_test.pt",
        arch_or_bc=8,  # tiny: 8 / 16 / 32 channels — fast on CPU
        out_channels=32,
        spatial_reduction=4,
        param_mb=0.1,
        license="Apache-2.0",
        note="test-only fixture",
    )


def test_wrapper_output_shape_matches_dinov2() -> None:
    """Wrapper produces ``(B, 37*37, 768)`` regardless of backbone family.

    This is the single contract cls-studio' scoring depends on. A mismatch
    here means the bank / cdist code would crash or silently mis-score.
    """
    spec = _simpleunet_spec_with_tiny_bc()
    model = BGBackboneWrapper(spec).eval()
    x = torch.randn(2, 3, WINDOW_SIZE, WINDOW_SIZE)
    out = model.forward_features(x)
    tokens = out["x_norm_patchtokens"]
    assert tokens.shape == (2, GRID * GRID, TARGET_DIM)


def test_wrapper_handles_non_window_input() -> None:
    """Pool-then-project means any HW > 4× tolerates passing through.

    ClsStudio always feeds 518×518, but downstream debug paths may try
    arbitrary shapes — the adapter should normalise them without crashing.
    """
    spec = _simpleunet_spec_with_tiny_bc()
    model = BGBackboneWrapper(spec).eval()
    x = torch.randn(1, 3, 224, 224)  # smaller window
    tokens = model.forward_features(x)["x_norm_patchtokens"]
    assert tokens.shape == (1, GRID * GRID, TARGET_DIM)


def test_wrapper_intermediate_layers_stub() -> None:
    """``get_intermediate_layers`` returns one repeated entry per layer.

    Real multi-layer extraction isn't meaningful for BG backbones (we only
    distill the deepest stage), so the stub repeats the final tokens —
    enough to keep callers that pass ``layers=[9, 11]`` from crashing.
    """
    spec = _simpleunet_spec_with_tiny_bc()
    model = BGBackboneWrapper(spec).eval()
    x = torch.randn(1, 3, WINDOW_SIZE, WINDOW_SIZE)
    layers = model.get_intermediate_layers(x, n=[9, 11], return_class_token=False)
    assert len(layers) == 2
    assert layers[0].shape == (1, GRID * GRID, TARGET_DIM)


# ---------------------------------------------------------------------------- #
# Loader error paths
# ---------------------------------------------------------------------------- #
def test_load_bg_backbone_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown BG backbone"):
        load_bg_backbone("bg_totally_made_up", device="cpu")


def test_load_bg_backbone_missing_ckpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pointing the root at an empty dir produces a clear FileNotFoundError.

    We test via the env-var override path because that's also how production
    deploys point the loader at a pinned ckpt directory.
    """
    monkeypatch.setenv("CLS_BG_BACKBONE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="ckpt missing"):
        load_bg_backbone("bg_simpleunet_bc128_e30", device="cpu")


def test_load_backbone_dispatch_unknown_name() -> None:
    """``load_backbone`` rejects names that aren't DINOv2 or BG-registered."""
    with pytest.raises(KeyError, match="unknown backbone"):
        load_backbone("not_a_real_backbone", device="cpu")


# ---------------------------------------------------------------------------- #
# Optional integration: real ckpt round-trip
# ---------------------------------------------------------------------------- #
_CKPT_DIR = os.environ.get("BG_BACKBONE_CKPT_DIR") or os.environ.get("CLS_BG_BACKBONE_DIR")


_real_ckpt_available = bool(
    _CKPT_DIR
    and (Path(_CKPT_DIR) / "backbones_e30" / "simpleunet_bc32_seed1.pt").exists()
    and (Path(_CKPT_DIR) / "backbones_e30" / "simpleunet_bc128_seed1.pt").exists()
)

_needs_real_ckpt = pytest.mark.skipif(
    not _real_ckpt_available,
    reason="set BG_BACKBONE_CKPT_DIR pointing at a directory containing backbones_e30/*.pt",
)


@_needs_real_ckpt
def test_load_bg_backbone_real_ckpt(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: load the smallest real ckpt + forward at 518."""
    assert _CKPT_DIR is not None  # narrowed for mypy
    monkeypatch.setenv("CLS_BG_BACKBONE_DIR", _CKPT_DIR)
    model = load_bg_backbone("bg_simpleunet_bc32_e30", device="cpu")
    x = torch.randn(1, 3, WINDOW_SIZE, WINDOW_SIZE)
    tokens = model.forward_features(x)["x_norm_patchtokens"]
    assert tokens.shape == (1, GRID * GRID, TARGET_DIM)
    assert torch.isfinite(tokens).all()


# ---------------------------------------------------------------------------- #
# API smoke: /api/model + /api/model/select with a real BG backbone
#
# Uses TestClient to drive the FastAPI app in-process — no uvicorn, no
# network. Mirrors the pattern in tests/test_e2e_api.py but starts with a
# real BG backbone so the swap path actually exercises load_bg_backbone.
# ---------------------------------------------------------------------------- #
@pytest.mark.skip(reason="exercises clscore.api (FastAPI app), which is ported in v2 phase R2")
@_needs_real_ckpt
def test_api_smoke_bg_backbone_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start with bg_simpleunet_bc32_e30, swap to bc128, verify the state."""
    import numpy as np
    from fastapi.testclient import TestClient

    from clscore.api.app import build_app
    from clscore.api.state import APIState
    from clscore.bank import Bank

    assert _CKPT_DIR is not None
    monkeypatch.setenv("CLS_BG_BACKBONE_DIR", _CKPT_DIR)

    # Empty bank (dim=0). /api/model/select gates on bank_dim matching the
    # incoming variant's dim, but ignores the check when bank_dim==0; that
    # is the path a brand-new install hits before its first /api/banks/create.
    bank_dir = tmp_path / "banks" / "active"
    bank_dir.mkdir(parents=True)
    bank = Bank(normal=np.zeros((0, 0), dtype=np.float32))
    bank.save(bank_dir)

    initial_name = "bg_simpleunet_bc32_e30"
    model = load_bg_backbone(initial_name, device="cpu")
    state = APIState(
        bank=Bank.load(bank_dir),
        bank_dir=bank_dir,
        model=model,
        device="cpu",
        model_name=initial_name,
    )
    client = TestClient(build_app(state))

    # /api/model: BG metadata is surfaced via available_meta
    r = client.get("/api/model")
    assert r.status_code == 200
    info = r.json()
    assert info["name"] == initial_name
    assert info["dim"] == TARGET_DIM
    meta = info["available_meta"]
    assert meta[initial_name]["family"] == "simpleunet"
    assert meta[initial_name]["license"] == "Apache-2.0"
    assert meta["dinov2_vitb14"]["family"] == "dinov2"

    # /api/model/select: swap to bc128 (same dim, different ckpt)
    target_name = "bg_simpleunet_bc128_e30"
    r = client.post("/api/model/select", params={"name": target_name})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == target_name
    assert state.model_name == target_name

    # Unknown name still errors cleanly
    r = client.post("/api/model/select", params={"name": "bg_not_a_real_one"})
    assert r.status_code == 400
