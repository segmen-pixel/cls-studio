# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""BG-aware backbones distilled from DINOv2-giant on industrial background pools.

These backbones are drop-in replacements for ``dinov2_vitb14`` from cls-studio'
point of view: they expose ``forward_features(x)["x_norm_patchtokens"]``
returning ``(B, 37*37, 768)`` so the bank / scoring / sliding-window code
needs zero changes.

Why these exist
---------------
DINOv2-base is 327 MB and 14-patch ViT, which is overkill on small industrial
fixtures (typically one bolt / connector / film roll under controlled lighting).
We distilled smaller convolutional encoders against DINOv2-giant on a
background image pool (Unsplash Lite + OpenImages V7); the resulting features
preserve enough texture/edge structure for PatchCore-style kNN while shrinking
the model by ~20-200× and the inference cost similarly.

How the shape match works
-------------------------
cls-studio' sliding window is 518×518 with ``DINO_PATCH=14`` → a 37×37 patch
grid at 768 dim (matching ``dinov2_vitb14``). Distilled encoders produce
features at their own spatial reduction (SimpleUNet at H/4, timm encoders at
H/32). The wrapper bolts on an adapter::

    backbone last stage  (B, C, h, w)
       ↓ adaptive_avg_pool2d → 37×37
       ↓ nn.Conv2d(C, 768, 1)
    DINOv2-shaped tokens (B, 1369, 768)

The adapter is initialised randomly. The kNN distance space is built when the
normal bank is constructed, so what matters is *kNN locality* on the backbone's
feature manifold — not whether the adapter outputs values on the same numerical
scale as DINOv2. The trade this buys is size: SimpleUNet-bc128 is 20 MB against
DINOv2-base's 327 MB.

License attribution (per backbone)
----------------------------------
* SimpleUNet encoder: Apache 2.0 (originally cls-studio).
* MobileViT v2 weights (timm) : Apple ML license.
* EfficientFormer v2 (timm)   : Apache 2.0 (Snap Inc.).
* ConvNeXt (timm)             : MIT (Facebook Research).

A registered backbone ckpt is produced by distilling DINOv2 features into the
smaller encoder. No ckpt ships with this repository: BG backbones are loaded
from the directory named by ``$CLS_BG_BACKBONE_DIR``, which has no
default — loading one raises ``RuntimeError`` when the variable is unset. If
you redistribute a ckpt you distilled, carry the NOTICE entry for whichever
upstream encoder it is based on (see the licence list above).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sw import DINO_PATCH, WINDOW_SIZE

logger = logging.getLogger(__name__)


# ClsStudio-native token layout. These are the numbers ``dinov2_vitb14``
# happens to produce; we mirror them so the bank / scoring code is oblivious
# to which backbone is actually running.
GRID: int = WINDOW_SIZE // DINO_PATCH  # 37
TARGET_DIM: int = 768  # matches dinov2_vitb14


# ---------------------------------------------------------------------------- #
# SimpleUNet encoder
#
# A small self-contained UNet-style convolutional encoder (~150 LoC, no
# external model dependency) used as the SimpleUNet family of BG-aware
# backbones in the registry below.
# ---------------------------------------------------------------------------- #
def _norm_layer(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with auto group count (max 32, divisor-safe)."""
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.gap(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class SimpleUNetEncoder(nn.Module):
    """Three-stage convolutional encoder (UNet-style, no decoder, no head).

    Returns 3-stage features ``(e1, e2, e3)`` at H, H/2, H/4.
    BG-aware distillation supervises e3 against DINOv2-giant features, so
    that is the stage the wrapper feeds into the adapter.
    """

    def __init__(self, base_channels: int = 32, use_se: bool = True):
        super().__init__()
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        self.channels = (ch1, ch2, ch3)

        self.enc1 = self._block(3, ch1)
        self.se1 = _SEBlock(ch1) if use_se else nn.Identity()
        self.enc2 = self._block(ch1, ch2)
        self.se2 = _SEBlock(ch2) if use_se else nn.Identity()
        self.enc3 = self._block(ch2, ch3)
        self.se3 = _SEBlock(ch3) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.se1(self.enc1(x))
        e2 = self.se2(self.enc2(self.pool(e1)))
        e3 = self.se3(self.enc3(self.pool(e2)))
        return e1, e2, e3


# ---------------------------------------------------------------------------- #
# Last-stage wrappers (encoder → single feature map)
# ---------------------------------------------------------------------------- #
class _SimpleUNetLastStage(nn.Module):
    """Adapter that surfaces only e3 from SimpleUNetEncoder."""

    def __init__(self, base_channels: int):
        super().__init__()
        self.encoder = SimpleUNetEncoder(base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, e3 = self.encoder(x)
        return e3


class _TimmLastStage(nn.Module):
    """timm ``features_only`` encoder returning only the deepest stage.

    timm is imported lazily so that cls-studio installs that don't use the
    timm-backed registry entries (everything except mobilevitv2 / convnext /
    efficientformer) don't need to ship the dependency. The user gets a clear
    ImportError pointing at the right install command when they try to load
    one of those backbones without timm.
    """

    def __init__(self, arch: str):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                f"Loading BG backbone {arch!r} requires the optional "
                f"'timm' package. Install with: pip install timm>=1.0"
            ) from exc
        self.encoder = timm.create_model(arch, pretrained=False, features_only=True, in_chans=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return feats[-1]


# ---------------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------------- #
def _ckpt_root() -> Path:
    """Where to look for distilled-backbone ckpts.

    ``$CLS_BG_BACKBONE_DIR`` is required; a :class:`RuntimeError` is
    raised when it is not set. Files inside the directory keep their original
    names (``simpleunet_bc128_seed1.pt`` etc.) so the registry can reference
    them by basename.
    """
    env = os.environ.get("CLS_BG_BACKBONE_DIR")
    if not env:
        raise RuntimeError(
            "BG-aware backbones require CLS_BG_BACKBONE_DIR to be set "
            "to the directory containing the distilled encoder checkpoints. "
            "There is no default location."
        )
    return Path(env).expanduser().resolve()


@dataclass(frozen=True)
class BGBackboneSpec:
    """One row in the registry.

    Fields
    ------
    name             : registry key (also goes into ``BankMeta.model``)
    family           : ``"simpleunet"`` (vendored) or ``"timm"`` (lazy dep)
    ckpt_relpath     : path under ``_ckpt_root()`` to the encoder weights
    arch_or_bc       : ``base_channels`` for SimpleUNet, timm arch name for timm
    out_channels     : last-stage channel count (drives the adapter input)
    spatial_reduction: H/spatial_reduction is the encoder's output grid; used
                       only for diagnostics — the adapter pools to ``GRID``
                       regardless, so this field is informational
    param_mb         : approximate ckpt size on disk, for the UI / docs
    license          : SPDX-ish tag; surfaced in /api/model so the operator
                       knows what they're shipping
    note             : short human-readable hint
    """

    name: str
    family: Literal["simpleunet", "timm"]
    ckpt_relpath: str
    arch_or_bc: str | int
    out_channels: int
    spatial_reduction: int
    param_mb: float
    license: str
    note: str = ""

    @property
    def ckpt_path(self) -> Path:
        return _ckpt_root() / self.ckpt_relpath


# Registry of the backbones this loader knows how to build. The SimpleUNet
# family is the one to reach for first; the timm entries take their features
# from a deeper stage (H/32), which costs spatial detail that patch-level
# anomaly scoring depends on, so they are registered mainly for comparison.
# Evaluate any of them on your own data before relying on it.
BG_BACKBONES: dict[str, BGBackboneSpec] = {
    "bg_simpleunet_bc32_e30": BGBackboneSpec(
        name="bg_simpleunet_bc32_e30",
        family="simpleunet",
        ckpt_relpath="backbones_e30/simpleunet_bc32_seed1.pt",
        arch_or_bc=32,
        out_channels=128,
        spatial_reduction=4,
        param_mb=1.5,
        license="Apache-2.0",
        note="iPhone-deploy tier (1.5 MB)",
    ),
    "bg_simpleunet_bc128_e30": BGBackboneSpec(
        name="bg_simpleunet_bc128_e30",
        family="simpleunet",
        ckpt_relpath="backbones_e30/simpleunet_bc128_seed1.pt",
        arch_or_bc=128,
        out_channels=512,
        spatial_reduction=4,
        param_mb=19.6,
        license="Apache-2.0",
        note="balanced default for desktop GPUs",
    ),
    "bg_simpleunet_bc256_e30": BGBackboneSpec(
        name="bg_simpleunet_bc256_e30",
        family="simpleunet",
        ckpt_relpath="backbones_e30/simpleunet_bc256_seed1.pt",
        arch_or_bc=256,
        out_channels=1024,
        spatial_reduction=4,
        param_mb=75.4,
        license="Apache-2.0",
        note="largest SimpleUNet variant; try when accuracy matters more than size",
    ),
    "bg_mobilevitv2_100": BGBackboneSpec(
        name="bg_mobilevitv2_100",
        family="timm",
        ckpt_relpath="backbones_timm/mobilevitv2_100_seed1.pt",
        arch_or_bc="mobilevitv2_100",
        out_channels=512,
        spatial_reduction=32,
        param_mb=18.2,
        license="Apple-ML",
        note="deepest stage is coarse for patch-level scoring",
    ),
    "bg_convnext_tiny": BGBackboneSpec(
        name="bg_convnext_tiny",
        family="timm",
        ckpt_relpath="backbones_timm/convnext_tiny_seed1.pt",
        arch_or_bc="convnext_tiny",
        out_channels=768,
        spatial_reduction=32,
        param_mb=108.4,
        license="MIT",
        note="low distillation loss, but verify on your data before use",
    ),
    "bg_efficientformerv2_s1": BGBackboneSpec(
        name="bg_efficientformerv2_s1",
        family="timm",
        ckpt_relpath="backbones_timm/efficientformerv2_s1_seed1.pt",
        arch_or_bc="efficientformerv2_s1",
        out_channels=224,
        spatial_reduction=32,
        param_mb=22.0,
        license="Apache-2.0",
        note="NOT recommended: 224-hard-coded attention breaks at 518 input",
    ),
}


def is_bg_backbone(name: str) -> bool:
    """Whether ``name`` is a BG-aware backbone (vs a DINOv2 variant)."""
    return name in BG_BACKBONES


# ---------------------------------------------------------------------------- #
# DINOv2-compatible wrapper
# ---------------------------------------------------------------------------- #
class BGBackboneWrapper(nn.Module):
    """Encoder + adapter producing DINOv2-shaped patch tokens.

    Exposes ``forward_features(x) -> {"x_norm_patchtokens": (B, GRID*GRID, TARGET_DIM)}``
    and a ``get_intermediate_layers`` stub, matching the subset of the DINOv2
    hub API that cls-studio consumes in feature_extractor.py.
    """

    def __init__(
        self,
        spec: BGBackboneSpec,
        target_dim: int = TARGET_DIM,
        grid: int = GRID,
    ):
        super().__init__()
        self.spec = spec
        self.target_dim = target_dim
        self.grid = grid

        if spec.family == "simpleunet":
            if not isinstance(spec.arch_or_bc, int):
                raise TypeError(
                    f"SimpleUNet spec needs int base_channels, got "
                    f"{type(spec.arch_or_bc).__name__}"
                )
            self.backbone: nn.Module = _SimpleUNetLastStage(base_channels=spec.arch_or_bc)
        elif spec.family == "timm":
            if not isinstance(spec.arch_or_bc, str):
                raise TypeError(
                    f"timm spec needs str arch name, got "
                    f"{type(spec.arch_or_bc).__name__}"
                )
            self.backbone = _TimmLastStage(arch=spec.arch_or_bc)
        else:  # pragma: no cover - dataclass Literal already prevents this
            raise ValueError(f"unknown backbone family: {spec.family!r}")

        # Pool-first-then-project: cheaper than projecting full-res features
        # to 768d and then pooling, and numerically equivalent for the kNN
        # space we care about (linear adapter commutes with avg_pool up to
        # a constant when the conv has no bias variation across positions).
        self.adapter = nn.Conv2d(spec.out_channels, target_dim, kernel_size=1)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """DINOv2-compatible forward.

        ``x`` is expected at 518×518 (cls-studio window size). Other sizes work
        as long as the backbone accepts them; the adapter always pools to
        ``self.grid`` so the output token count stays compatible with the bank.
        """
        feats = self.backbone(x)                                     # (B, C, h, w)
        pooled = F.adaptive_avg_pool2d(feats, output_size=self.grid)  # (B, C, GRID, GRID)
        projected = self.adapter(pooled)                              # (B, TARGET_DIM, GRID, GRID)
        b, d, h, w = projected.shape
        tokens = projected.permute(0, 2, 3, 1).reshape(b, h * w, d)   # (B, GRID*GRID, TARGET_DIM)
        return {"x_norm_patchtokens": tokens}

    @torch.no_grad()
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: list[int] | int = 1,
        return_class_token: bool = False,
    ) -> list[torch.Tensor]:
        """Stub for the DINOv2 multi-layer API.

        BG backbones don't expose intermediate ViT blocks; we return the same
        final-stage tokens repeated ``len(n)`` (or ``n``) times so that callers
        passing ``layers=[9, 11]`` to ``extract_window_tokens`` get a tensor of
        the expected length rather than a crash. The result is not meaningfully
        multi-layer — the concat is just the last-stage tokens repeated — so
        prefer ``forward_features`` for real scoring.
        """
        tokens = self.forward_features(x)["x_norm_patchtokens"]
        n_repeats = n if isinstance(n, int) else len(n)
        return [tokens] * n_repeats


# ---------------------------------------------------------------------------- #
# Loader
# ---------------------------------------------------------------------------- #
def load_bg_backbone(
    name: str,
    device: str = "cuda:0",
    target_dim: int = TARGET_DIM,
    grid: int = GRID,
) -> BGBackboneWrapper:
    """Build a ``BGBackboneWrapper`` and load distilled encoder weights.

    The adapter intentionally stays at random init: it was not trained
    (distillation targeted DINOv2-giant features at a different layout), and
    PatchCore's normal-bank construction shapes the kNN distance space around
    whatever scale the random projection happens to land at.
    """
    if name not in BG_BACKBONES:
        raise KeyError(
            f"unknown BG backbone {name!r}. Available: {sorted(BG_BACKBONES)}"
        )
    spec = BG_BACKBONES[name]
    if not spec.ckpt_path.exists():
        raise FileNotFoundError(
            f"BG backbone ckpt missing: {spec.ckpt_path}. "
            f"Set $CLS_BG_BACKBONE_DIR to the directory containing the "
            f"distilled weights."
        )

    model = BGBackboneWrapper(spec, target_dim=target_dim, grid=grid)
    ckpt = torch.load(spec.ckpt_path, map_location="cpu", weights_only=True)

    # The BG distillation pipeline saves the bare encoder
    # under "encoder_state_dict"; the distillation projection heads are
    # discarded (they targeted DINOv2-giant 1536d at 4 stages, not our
    # 37×37×768 cls-studio layout, so reusing those weights would be worse
    # than the random adapter init we ship instead).
    if "encoder_state_dict" not in ckpt:
        raise KeyError(
            f"ckpt {spec.ckpt_path} has keys {list(ckpt)}; expected "
            f"'encoder_state_dict'. BG backbone ckpts must be produced by "
            f"the cls-studio BG distillation pipeline."
        )
    enc_state = ckpt["encoder_state_dict"]
    missing, unexpected = model.backbone.encoder.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        # Light warning is enough; some timm versions tweak buffer names.
        logger.warning(
            "load_bg_backbone(%s): missing=%d unexpected=%d on encoder.load_state_dict",
            name, len(missing), len(unexpected),
        )

    model = model.eval().to(device)
    logger.info(
        "loaded BG backbone %s (family=%s, %.1fMB, license=%s) on %s",
        name, spec.family, spec.param_mb, spec.license, device,
    )
    return model
