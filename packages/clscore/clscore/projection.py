# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""2D projectors for the bank visualisation panel.

Two strategies are exposed behind a tiny common interface:

* ``NormalPCAProjector`` — vanilla PCA fit on the normal bank only.
  Stable when there are no NG samples; the axes capture the dominant
  *appearance variation among normals* and never adapt to anomalies.

* ``ContrastivePCAProjector`` — contrastive PCA (cPCA). Solves the
  eigenproblem ``Cov(NG) - alpha * Cov(normal)`` and keeps the top two
  eigenvectors. Highlights directions in DINOv2 feature space where the
  NG bank has *more* variance than the normal bank — i.e. the axes
  literally "grow" toward the anomaly direction as new NG patches are
  appended. With ``alpha=0`` this collapses to PCA-on-NG; with
  ``alpha→inf`` it approaches the directions of *least* normal variance.

Both projectors expose ``transform(features) -> [N, 2]`` and an
``axis_info`` dict consumed by the UI (mode label + per-axis variance
percentages, when available).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Projector(Protocol):
    """Common interface for 2D bank projectors used by the API and UI.

    Implementations:
        - ``NormalPCAProjector``: plain PCA on the normal bank.
        - ``ContrastivePCAProjector``: cPCA contrasting NG against normal.
    """

    def transform(self, features: np.ndarray) -> np.ndarray: ...

    @property
    def axis_info(self) -> dict: ...


# ---------- normal-only PCA -----------------------------------------------


@dataclass
class NormalPCAProjector:
    """Thin wrapper around ``sklearn.decomposition.PCA`` so both modes
    share a single call site in ``api/app.py``."""

    pca: object  # sklearn PCA instance, kept opaque to dodge import cost
    n_normal: int

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.pca.transform(features)  # type: ignore[attr-defined]

    @property
    def axis_info(self) -> dict:
        ratios = getattr(self.pca, "explained_variance_ratio_", None)
        pct = [float(r) * 100.0 for r in ratios] if ratios is not None else [0.0, 0.0]
        return {
            "mode": "normal",
            "pc1_pct": pct[0] if len(pct) > 0 else 0.0,
            "pc2_pct": pct[1] if len(pct) > 1 else 0.0,
            "n_normal": self.n_normal,
            "n_ng": 0,
            "alpha": None,
        }


def fit_normal_pca(normal: np.ndarray) -> NormalPCAProjector:
    """Fit a 2D PCA projector on normal-bank features.

    Args:
        normal: Normal-bank feature matrix of shape ``[N, D]``.

    Returns:
        A fitted ``NormalPCAProjector`` whose ``transform`` maps any
        feature row to a 2D coordinate in the same projection.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42).fit(normal)
    return NormalPCAProjector(pca=pca, n_normal=int(normal.shape[0]))


# ---------- contrastive PCA -----------------------------------------------


@dataclass
class ContrastivePCAProjector:
    """Two cPCA components plus the centring vector.

    ``components`` is shape ``[2, D]`` (rows are eigenvectors, applied as
    ``(x - mean) @ components.T``). ``mean`` is the NG-bank centroid we
    centre on; this matters because the projection mostly visualises
    where NG samples scatter and we want them roughly centred in the
    plot.
    """

    components: np.ndarray  # [2, D] float32
    mean: np.ndarray        # [D]    float32
    eigenvalues: np.ndarray # [2]    float32  (cPCA contrast values)
    n_normal: int
    n_ng: int
    alpha: float

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = features.astype(np.float32, copy=False) - self.mean
        return x @ self.components.T

    @property
    def axis_info(self) -> dict:
        # cPCA "explained variance" doesn't have the same meaning as PCA's
        # (it's a contrast, not a variance fraction), so we just expose
        # the raw eigenvalues — the UI shows them as relative bars.
        evs = [float(v) for v in self.eigenvalues]
        return {
            "mode": "anomaly",
            "pc1_contrast": evs[0] if len(evs) > 0 else 0.0,
            "pc2_contrast": evs[1] if len(evs) > 1 else 0.0,
            "n_normal": self.n_normal,
            "n_ng": self.n_ng,
            "alpha": float(self.alpha),
        }


def fit_contrastive_pca(
    normal: np.ndarray,
    ng: np.ndarray,
    alpha: float = 1.0,
) -> ContrastivePCAProjector:
    """Fit cPCA: top-2 eigenvectors of ``Cov(ng) - alpha * Cov(normal)``.

    Both inputs must be ``[N, D]`` float arrays. ``ng`` should be the
    concatenation of every critical/negative sub-bank (any label) so the
    axes reflect *all* anomalies the user has taught so far. We take the
    *real* part of the eigendecomposition because the contrast matrix is
    symmetric and any imaginary residue is numerical noise.

    Caller is expected to have verified ``ng.shape[0] >= 2`` and that
    feature dims match — both tiers must come from the same DINOv2
    backbone.
    """
    if ng.ndim != 2 or normal.ndim != 2 or ng.shape[1] != normal.shape[1]:
        raise ValueError(
            f"shape mismatch: normal={normal.shape}, ng={ng.shape}"
        )

    # Centring choice matters: if both tiers are centred on their own
    # means, cPCA only sees *spread* differences and misses the
    # mean-shift component — fatal here, where NG is typically a tight
    # cluster shifted from the normal manifold rather than spread out
    # over it. We centre Cov(NG) on the *normal* mean so the shift is
    # captured as variance, while Cov(normal) stays centred on its own
    # mean (its true variance). The plot then anchors on the normal
    # centroid, which is also where the user expects "the OK cluster"
    # to sit.
    mean_normal = normal.mean(axis=0).astype(np.float32)
    nm_c = normal.astype(np.float32) - mean_normal
    ng_c = ng.astype(np.float32) - mean_normal

    cov_ng = (ng_c.T @ ng_c) / max(ng_c.shape[0], 1)
    cov_nm = (nm_c.T @ nm_c) / max(nm_c.shape[0], 1)
    contrast = cov_ng - alpha * cov_nm  # [D, D]

    # eigh: symmetric eigendecomposition, returns ascending eigenvalues.
    # We want the *largest* (most NG-variance, least normal-variance).
    eigvals, eigvecs = np.linalg.eigh(contrast.astype(np.float64))
    # Take the top 2 in descending order
    top = np.argsort(eigvals)[::-1][:2]
    components = eigvecs[:, top].T.astype(np.float32)  # [2, D]
    top_eigvals = eigvals[top].astype(np.float32)

    # Sign convention: flip components so the NG centroid lands on the
    # positive side of each axis. Otherwise eigh's arbitrary signs make
    # the plot mirror itself between bank-switches, which is jarring.
    ng_centroid = (ng.astype(np.float32) - mean_normal).mean(axis=0)
    centroid_xy = ng_centroid @ components.T
    for i in range(2):
        if centroid_xy[i] < 0:
            components[i] *= -1.0

    return ContrastivePCAProjector(
        components=components,
        mean=mean_normal,
        eigenvalues=top_eigvals,
        n_normal=int(normal.shape[0]),
        n_ng=int(ng.shape[0]),
        alpha=float(alpha),
    )
