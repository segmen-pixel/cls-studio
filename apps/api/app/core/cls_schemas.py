# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Pydantic v2 models exposed at the cls-studio API boundary.

Ported from cls-studio v0.1 (``cls-studio/api/schemas.py``); the memory-bank
and scoring shapes are unchanged, only the project scoping around them moved
to the cls-studio-native project model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["normal", "critical", "negative"]
ProjectionMode = Literal["normal", "anomaly"]


class BankState(BaseModel):
    """Tier sizes and per-label breakdowns at a point in time."""

    normal: int = Field(..., description="Patches in the normal bank")
    critical: int = Field(..., description="Total patches in the critical bank")
    negative: int = Field(..., description="Total patches in the negative bank")
    critical_by_label: dict[str, int] = Field(default_factory=dict)
    negative_by_label: dict[str, int] = Field(default_factory=dict)
    dim: int = Field(..., description="Feature dimension (DINOv2 output dim)")


class ScoreResult(BaseModel):
    """Per-image scoring result. ``heatmap_png_base64`` is a base64 RGBA PNG."""

    max_score: float
    mean_score: float
    p99_score: float
    topk_score: float = Field(
        0.0,
        description=(
            "Mean of the k hottest per-patch composite scores — the same "
            "statistic as the Develop tab's separation check, so its "
            "threshold applies to this value directly"
        ),
    )
    n_exemplar_rows: int = Field(
        0, description="Critical exemplar rows the alpha term used (0 = alpha off or no exemplars)"
    )
    heatmap_png_base64: str = Field(..., description="JET overlay encoded as base64 PNG")
    original_jpeg_base64: str = Field(
        "",
        description=(
            "Downscaled JPEG preview of the uploaded image (long edge <=1280). "
            "The heatmap-off view: browsers cannot render TIFF and friends, so "
            "the server hands back a web-displayable copy of whatever it decoded"
        ),
    )
    inspection_id: str = Field(
        "", description="Id of the persisted inspection-log entry ('' when persist=false or persistence failed)"
    )
    critical_attribution: dict[str, float] = Field(default_factory=dict)
    negative_attribution: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


class AppendResult(BaseModel):
    tier: Tier
    label: str = Field("", description="Sanitised label used on disk (empty for normal tier)")
    appended_patches: int
    bank: BankState


class BatchAppendResult(BaseModel):
    tier: Tier
    label: str = ""
    appended_patches: int
    images_processed: int
    images_failed: list[str]
    bank: BankState


class BankCapacityInfo(BaseModel):
    """Runtime memory-bank budget and how full the normal tier is against it."""

    capacity: Literal["small", "medium", "large"] = "medium"
    ceiling: int = Field(..., description="Max total normal patches allowed at this tier")
    normal: int = Field(..., description="Current normal-bank patches")
    pct: float = Field(..., description="normal / ceiling * 100, clamped to 100")
    est_vram_mb: int = Field(..., description="Rough resident VRAM of the normal bank at this fill (fp16)")
    labeled: int = Field(
        0,
        description=(
            "Current critical + negative patches. These tiers are resident too "
            "but are NOT capped and do not count against the ceiling"
        ),
    )
    est_vram_total_mb: int = Field(
        0, description="Rough resident VRAM of ALL tiers at this fill (fp16)"
    )


class BankCapacitySet(BaseModel):
    """Request body for changing the active bank's size budget."""

    capacity: Literal["small", "medium", "large"]


class LabelsResponse(BaseModel):
    """List of currently populated labels per labelled tier."""

    critical: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class BankInfo(BaseModel):
    """One memory bank in a project (cheap summary from its meta)."""

    id: str
    name: str
    images: dict[str, int] = Field(default_factory=dict, description="per-tier image counts")


class BankList(BaseModel):
    banks: list[BankInfo] = Field(default_factory=list)
    active_bank_id: str | None = None


class SelectResult(BaseModel):
    """Result of activating a project + bank for the API process."""

    project_id: str
    bank_id: str
    bank_dir: str
    device: str
    bank: BankState
    banks: list[BankInfo] = Field(default_factory=list)


class ProjectionPoint(BaseModel):
    """One patch's 2D coordinates in the bank projection."""

    tier: Tier
    label: str = Field("", description="Sub-bank label ('' for normal/unlabeled)")
    x: float
    y: float
    image: str = Field(
        "",
        description=(
            "Source image filename this patch was taught from, resolved via the "
            "per-image row-range index ('' for legacy rows without an index)"
        ),
    )
    score: float | None = Field(
        None,
        description=(
            "Top-k mean distance to the normal bank (only when with_scores is "
            "requested; leave-own-image-out for normal-tier rows)"
        ),
    )


class StoredImageEval(BaseModel):
    """Image-level anomaly stats computed from the image's stored patch rows.

    No model forward — the patches taught into the bank are scored directly
    against the normal bank (leave-own-image-out for normal-tier images).
    """

    name: str
    tier: Tier
    label: str = ""
    patches: int
    score_max: float
    score_p99: float
    score_mean: float
    top_scores: list[float] = Field(
        default_factory=list,
        description=(
            "The image's patch scores sorted descending, truncated to 256 "
            "values — lets a client recompute any top-k statistic (top-k "
            "mean, k-th value) without re-scoring"
        ),
    )
    top_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Local row indices (0-based, within this image's stored rows) "
            "aligned with top_scores — lets the projection guarantee the "
            "highest-scoring patches survive downsampling. Empty for cache "
            "entries written before this field existed"
        ),
    )


class RuntimeConfig(BaseModel):
    """Deployable verdict recipe, persisted as ``runtime_config.json`` in the
    bank directory so a bank export carries everything a runtime needs:
    the features (bank arrays), the exemplar marks (severity sidecars) and
    this file — load the package, read the config, inspect.
    """

    version: int = 1
    metric: Literal["topk_mean"] = "topk_mean"
    topk: int = Field(10, ge=1, le=256, description="k of the image-level top-k mean")
    k: int = Field(5, ge=1, le=50, description="Normal-bank neighbours per patch")
    alpha: float = Field(0.0, ge=0.0, description="Exemplar-boost weight")
    beta: float = Field(0.0, ge=0.0)
    exemplar_alpha: bool = Field(True, description="Alpha measures distance to exemplar rows only")
    threshold: float | None = Field(None, description="NG when topk_score exceeds this")
    bank_capacity: Literal["small", "medium", "large"] = Field(
        "medium",
        description=(
            "Runtime memory-bank size budget. Caps the total normal-tier patch "
            "count (and thus the resident scoring VRAM) at teach time: "
            "small / medium / large. Operational knob, not part of the verdict "
            "recipe — changing it never marks the config stale."
        ),
    )
    saved_at: str = Field("", description="Server-stamped ISO timestamp")
    bank_fingerprint: str = Field(
        "", description="Content fingerprint of the bank this recipe was tuned on (server-stamped)"
    )
    stale: bool = Field(
        False,
        description=(
            "Set on read: the bank changed (teach/delete/mark) since this "
            "recipe was saved — re-run the separation check and save again"
        ),
    )


class AnnotationRect(BaseModel):
    """One defect-mark rectangle, normalized to the original image size."""

    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., gt=0.0, le=1.0)
    h: float = Field(..., gt=0.0, le=1.0)


class AnnotateResult(BaseModel):
    """Outcome of marking an image's defect regions as exemplar rows."""

    tier: Tier
    label: str = ""
    name: str
    rows_marked: int = Field(..., description="Bank rows set to heavy severity")
    rects: list[AnnotationRect] = Field(default_factory=list)


class ImageCmin(BaseModel):
    """Per-image min distances from its cached top patches to the exemplar set.

    ``top_cmin`` is aligned with the cached eval's ``top_indices`` /
    ``top_scores`` so a client can compose ``score + alpha / (eps + cmin)``
    and sweep alpha without re-scoring. Rows of the evaluated image itself
    are excluded from the exemplar set (leave-own-image-out).
    """

    name: str
    tier: Tier
    label: str = ""
    top_cmin: list[float] = Field(default_factory=list)


class ProjectionAxisInfo(BaseModel):
    """Axis metadata for the visualisation panel.

    Mirrors the ``axis_info`` dict returned by both projectors in
    ``clscore.projection`` so the UI can render mode-appropriate labels.
    ``pc*_pct`` is populated for normal PCA (explained variance %); the
    ``pc*_contrast`` fields are populated for cPCA (raw eigenvalues, which
    are contrasts rather than variances and are best shown as relative bars).
    """

    mode: ProjectionMode
    pc1_pct: float | None = None
    pc2_pct: float | None = None
    pc1_contrast: float | None = None
    pc2_contrast: float | None = None
    n_normal: int
    n_ng: int
    alpha: float | None = None


class ProjectionResponse(BaseModel):
    """2D projection of the active bank for the visualisation panel."""

    mode: Literal["normal", "anomaly", "empty"] = Field(
        ..., description="'empty' when nothing could be projected"
    )
    granularity: Literal["patch", "image"] = Field(
        "patch", description="'image' when each point aggregates one taught image"
    )
    guaranteed: int = Field(
        0,
        description=(
            "Points force-included from cached per-image top patches "
            "(patch granularity only; 0 when no eval cache was available)"
        ),
    )
    axis_info: ProjectionAxisInfo | None = None
    points: list[ProjectionPoint] = Field(default_factory=list)
    total: dict[str, int] = Field(
        default_factory=dict, description="patch counts per tier before sampling"
    )
    sampled: dict[str, int] = Field(
        default_factory=dict, description="points actually returned per tier"
    )


# ---- feature store / label sets -------------------------------------------
# The split shape of teaching: an image is ingested once (expensive), assigned
# a tier as often as the operator changes their mind (free), and the bank is
# assembled from the two on demand (numpy). See clscore.store.


class StoreImageInfo(BaseModel):
    """One ingested image and what the active label set says it is."""

    id: str = Field(..., description="Opaque store handle; filenames are not unique")
    name: str
    rows: int = Field(..., description="Feature rows this image contributes")
    grid_rows: int = Field(
        0,
        description=(
            "Patches the image's sliding-window grid yields. Defect marks "
            "address this grid, so it differs from ``rows`` for any image the "
            "per-image coreset cap reduced"
        ),
    )
    width: int = 0
    height: int = 0
    has_image: bool = Field(False, description="A source image is available to display")
    tier: str = Field("", description="'' when the image is not assigned yet")
    label: str = ""
    severity: int = Field(0, description="0 when unassigned")
    marks: int = Field(0, description="Grid patches marked as the defect")
    rects: list[AnnotationRect] = Field(
        default_factory=list,
        description=(
            "The rectangles the marks were drawn from, normalised to the "
            "image. Carried so the editor can redraw and adjust them; the "
            "grid patches in ``marks`` are the scoring-facing artifact"
        ),
    )
    group: str = Field("", description="Manual validation group, if assigned")


class StoreListResponse(BaseModel):
    images: list[StoreImageInfo] = Field(default_factory=list)
    total_rows: int = 0
    dim: int = 0
    model: str = ""
    labelset_id: str = ""


class AssemblyStatus(BaseModel):
    """Whether the loaded bank still matches the store + active label set.

    Assembly is explicit, so the UI needs to know when the bank it is scoring
    against is behind the labelling the operator has been doing.
    """

    labelset_id: str = ""
    labelset_name: str = ""
    store_images: int = 0
    store_rows: int = 0
    assigned: int = 0
    unassigned: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    fingerprint: str = ""
    stale: bool = Field(
        False, description="Assignments changed since the bank was last assembled"
    )
    assembled_from: str = Field("", description="Label set the loaded bank was built from")
    migrated: bool = Field(False, description="This bank has a feature store")


class IngestResult(BaseModel):
    ingested: int
    rows: int
    failed: list[str] = Field(default_factory=list, description="Undecodable uploads")
    status: AssemblyStatus


class MigrationResult(BaseModel):
    """Outcome of carving an existing bank into a store + label set."""

    images: int
    rows: int
    labelset_id: str = ""
    status: AssemblyStatus


class LabelSetInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    counts: dict[str, int] = Field(default_factory=dict)
    updated_at: str = ""


class LabelSetList(BaseModel):
    labelsets: list[LabelSetInfo] = Field(default_factory=list)
    active_id: str = ""


class AssignResult(BaseModel):
    changed: int
    status: AssemblyStatus


class MarkResult(BaseModel):
    id: str
    marks: int = Field(..., description="Grid patches the rectangles cover")
    status: AssemblyStatus


class AssembleResult(BaseModel):
    bank: BankState
    status: AssemblyStatus


class GroupPreview(BaseModel):
    """What a grouping rule would split the store into.

    ``ungrouped`` is the number the rule could not place. It is the figure
    that says whether the naming convention was guessed right, so it is
    reported rather than folded into the group list.
    """

    mode: str
    groups: dict[str, list[str]] = Field(default_factory=dict)
    grouped: int = 0
    ungrouped: int = 0
