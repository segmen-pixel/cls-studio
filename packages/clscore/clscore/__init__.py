# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""Cls-Studio — industrial visual OK/NG classification with a 3-tier memory bank.

Top-level public API:

    from clscore import (
        Bank, BankMeta,                         # 3-tier memory bank
        load_dinov2,                            # frozen feature backbone
        score_image, image_auroc,               # one-shot scoring + metric
        extract_distance_components,            # alpha/beta-independent cache
        compose_score_grid,                     # alpha/beta combination
    )

Internals (`feature_extractor`, `sw`, `io`) remain importable but are subject
to change.
"""

from __future__ import annotations

from .bank import (
    Bank,
    BankMeta,
    coreset_reduce,
    coreset_reduce_indexed,
    kcenter_greedy,
    safe_label,
)
from .feature_extractor import (
    DEFAULT_DINO_DIM,
    DEFAULT_DINO_NAME,
    DINO_MODELS,
    extract_window_tokens,
    extract_windows_tokens_batched,
    iter_images_features_batched,
    iter_windows_tokens_batched,
    load_dinov2,
)
from .incident import (
    DEFAULT_SEVERITY,
    SEVERITY_HEAVY,
    SEVERITY_LIGHT,
    SEVERITY_MEDIUM,
    IncidentMetaArray,
)
from .scoring import (
    LabelWinner,
    attribution_per_label,
    compose_score_grid,
    extract_distance_components,
    image_auroc,
    per_label_winners,
    score_image,
)
from .sw import DINO_PATCH, WINDOW_SIZE, WINDOW_STRIDE, pad_to_min, sw_offsets

__version__ = "0.2.0.dev0"

__all__ = [
    "__version__",
    # bank
    "Bank",
    "BankMeta",
    "coreset_reduce",
    "coreset_reduce_indexed",
    "kcenter_greedy",
    "safe_label",
    # incident metadata (time-aware memory)
    "IncidentMetaArray",
    "DEFAULT_SEVERITY",
    "SEVERITY_LIGHT",
    "SEVERITY_MEDIUM",
    "SEVERITY_HEAVY",
    # backbone
    "DEFAULT_DINO_DIM",
    "DEFAULT_DINO_NAME",
    "DINO_MODELS",
    "load_dinov2",
    "extract_window_tokens",
    "extract_windows_tokens_batched",
    "iter_images_features_batched",
    "iter_windows_tokens_batched",
    # scoring
    "score_image",
    "image_auroc",
    "extract_distance_components",
    "compose_score_grid",
    "attribution_per_label",
    "per_label_winners",
    "LabelWinner",
    # sliding window
    "DINO_PATCH",
    "WINDOW_SIZE",
    "WINDOW_STRIDE",
    "sw_offsets",
    "pad_to_min",
]
