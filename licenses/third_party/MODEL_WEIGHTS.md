# Model Weights — License & Attribution

Cls-Studio redistributes **no** model weights. The pre-trained weight below is
downloaded from its upstream host onto your own machine on first use; it is
documented here for attribution and license clarity. It is licensed under
**Apache License 2.0** (see the root `LICENSE` file for the full text).

This is the complete list: no other model weights are bundled with, downloaded
by, or mirrored for Cls-Studio.

---

## DINOv2 (patch feature extractor)

| File | `dinov2_vitb14_pretrain.pth` (weights only) |
|---|---|
| Upstream | https://github.com/facebookresearch/dinov2 |
| Copyright | Copyright (c) Meta Platforms, Inc. and affiliates |
| Weights license | Apache-2.0 |
| LICENSE source | https://github.com/facebookresearch/dinov2/blob/main/LICENSE |
| Pretrained on | ImageNet-22k + LVD-142M (curated by Meta) |
| Obtained by | `torch.hub` on your own machine at first use — downloaded from `https://dl.fbaipublicfiles.com/dinov2/` and cached under `TORCH_HOME` |

> ⚠️ **Source tree note.** Neither the weight nor the source tree is
> redistributed with Cls-Studio. The DINOv2 torch-hub source tree
> (`facebookresearch_dinov2_main/`) is **not** bundled because recent
> versions mix Apache-2.0 with non-commercial fragments
> (`LICENSE_CELL_DINO_CODE`: CC-BY-NC-4.0 and `LICENSE_XRAY_DINO_MODEL`:
> FAIR Noncommercial), which cannot be re-shipped under Apache-2.0.
> The model-definition Python files are fetched at runtime via
> `torch.hub.load('facebookresearch/dinov2', ...)` on the user's machine,
> staying outside Cls-Studio's redistribution surface.

---

## License compliance summary

The weight above is governed by the same Apache License 2.0 that applies to
Cls-Studio itself. Because Cls-Studio ships source only and the weight is
fetched by the user's own machine, section 4 does not attach to this project;
the attributions are reproduced anyway:

- **Section 4(a) — recipients receive a copy of the license**: satisfied by the
  root `LICENSE` file shipped alongside every distribution.
- **Section 4(c) — copyright/attribution notices preserved**: listed in this
  file and in `THIRD_PARTY_NOTICES.md`.
- **Section 4(d) — NOTICE attributions preserved**: the DINOv2 repository
  publishes no separate `NOTICE` file (re-verified 2026-07-28). If one is added
  upstream, it will be incorporated at the next refresh.

The weight file is used verbatim; no modification has been performed.
