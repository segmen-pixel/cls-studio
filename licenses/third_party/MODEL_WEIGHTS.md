# Redistributed Model Weights — License & Attribution

The Cls-Studio installer redistributes the following pre-trained model weight.
It is licensed under **Apache License 2.0** (see the root `LICENSE` file for the
full text) and is reproduced with the upstream copyright notice preserved as
required by section 4(a)-(c).

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
| Obtained by | `scripts/build_installer.py` — copied from the local torch hub cache, or downloaded from `https://dl.fbaipublicfiles.com/dinov2/` at installer build time |

> ⚠️ **Source tree note.** Only the pretrained weight file is redistributed
> with Cls-Studio. The DINOv2 torch-hub source tree
> (`facebookresearch_dinov2_main/`) is **not** bundled because recent
> versions mix Apache-2.0 with non-commercial fragments
> (`LICENSE_CELL_DINO_CODE`: CC-BY-NC-4.0 and `LICENSE_XRAY_DINO_MODEL`:
> FAIR Noncommercial), which cannot be re-shipped under Apache-2.0.
> The model-definition Python files are fetched at runtime via
> `torch.hub.load('facebookresearch/dinov2', ...)` on the user's machine,
> staying outside Cls-Studio's redistribution surface.

---

## License compliance summary

The redistributed weight above is governed by the same Apache License 2.0 that
applies to Cls-Studio itself. The required obligations are met as follows:

- **Section 4(a) — recipients receive a copy of the license**: satisfied by the
  root `LICENSE` file shipped alongside every distribution.
- **Section 4(c) — copyright/attribution notices preserved**: listed in this
  file and in `THIRD_PARTY_NOTICES.md`.
- **Section 4(d) — NOTICE attributions preserved**: the DINOv2 repository
  publishes no separate `NOTICE` file (re-verified 2026-07-28). If one is added
  upstream, it will be incorporated at the next refresh.

The weight file is reproduced verbatim as a binary "Object form" artifact; no
modification has been performed.
