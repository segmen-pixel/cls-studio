# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Tests for the 2D projectors used by the bank visualisation panel.

We test the *math contracts* — explained-variance shape, contrast
eigenvalue ordering, sign convention, NG-shift recovery — rather than
exact numerical values, because eigendecompositions on random data
have a sign ambiguity even after our flip-to-positive convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from clscore.projection import (
    NormalPCAProjector,
    fit_contrastive_pca,
    fit_normal_pca,
)


def _ring(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Compact, near-isotropic blob; good stand-in for a normal bank."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def _shifted(n: int, dim: int = 16, shift: float = 5.0, seed: int = 1) -> np.ndarray:
    """Blob translated along one axis; stand-in for an NG cluster."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim), dtype=np.float32) * 0.5
    x[:, 0] += shift
    return x


# ---- normal-only PCA --------------------------------------------------------


def test_fit_normal_pca_returns_2d_projection():
    proj = fit_normal_pca(_ring(50))
    assert isinstance(proj, NormalPCAProjector)
    out = proj.transform(_ring(8, seed=99))
    assert out.shape == (8, 2)


def test_normal_pca_axis_info_reports_explained_variance():
    proj = fit_normal_pca(_ring(50))
    info = proj.axis_info
    assert info["mode"] == "normal"
    assert 0.0 <= info["pc1_pct"] <= 100.0
    assert 0.0 <= info["pc2_pct"] <= 100.0
    # PC1 should hold at least as much variance as PC2 by construction.
    assert info["pc1_pct"] >= info["pc2_pct"]
    assert info["n_normal"] == 50
    assert info["n_ng"] == 0
    assert info["alpha"] is None


# ---- contrastive PCA --------------------------------------------------------


def test_fit_contrastive_pca_recovers_shift_direction():
    """An NG bank shifted along axis 0 should yield a top component
    whose first coordinate dominates: the cPCA top axis should align
    with the direction of NG-vs-normal disagreement."""
    normal = _ring(200, dim=8, seed=2)
    ng = _shifted(60, dim=8, shift=8.0, seed=3)
    proj = fit_contrastive_pca(normal, ng, alpha=1.0)
    # Component 1 should be primarily along axis 0.
    c1 = proj.components[0]
    assert abs(c1[0]) > max(abs(v) for v in c1[1:])


def test_contrastive_pca_eigenvalues_descending():
    proj = fit_contrastive_pca(_ring(120, dim=8), _shifted(40, dim=8), alpha=1.0)
    assert proj.eigenvalues.shape == (2,)
    assert proj.eigenvalues[0] >= proj.eigenvalues[1]


def test_contrastive_pca_axis_info_shape():
    proj = fit_contrastive_pca(_ring(120, dim=8), _shifted(40, dim=8), alpha=2.5)
    info = proj.axis_info
    assert info["mode"] == "anomaly"
    assert info["n_normal"] == 120
    assert info["n_ng"] == 40
    assert info["alpha"] == pytest.approx(2.5)
    # Both contrast values should be finite floats.
    assert np.isfinite(info["pc1_contrast"])
    assert np.isfinite(info["pc2_contrast"])


def test_contrastive_pca_sign_convention_puts_ng_on_positive_side():
    """The flip-to-positive convention: after fitting, the NG centroid
    must land in the positive quadrant. Without this, every bank toggle
    in the UI would mirror the plot for no reason — a real bug we
    fixed once and want to keep fixed."""
    normal = _ring(200, dim=8)
    ng = _shifted(50, dim=8)
    proj = fit_contrastive_pca(normal, ng, alpha=1.0)
    ng_xy = proj.transform(ng).mean(axis=0)
    assert ng_xy[0] >= 0
    assert ng_xy[1] >= 0


def test_contrastive_pca_centres_on_normal_mean():
    """transform()'s subtraction is the *normal* mean, so the normal
    centroid lands close to the origin in the projected plane."""
    normal = _ring(200, dim=8, seed=10)
    ng = _shifted(50, dim=8, seed=11)
    proj = fit_contrastive_pca(normal, ng, alpha=1.0)
    nm_xy = proj.transform(normal).mean(axis=0)
    np.testing.assert_allclose(nm_xy, [0, 0], atol=1e-3)


def test_contrastive_pca_alpha_zero_collapses_to_pca_on_ng():
    """alpha=0 removes the normal-variance penalty, so the top axis is
    just the dominant variance direction of the NG bank itself."""
    normal = _ring(200, dim=8)
    ng = _shifted(50, dim=8)
    proj = fit_contrastive_pca(normal, ng, alpha=0.0)
    # The NG bank's variance is dominated by axis 0 (shift direction);
    # cPCA(alpha=0) should pick that out as the strongest.
    c1 = proj.components[0]
    assert abs(c1[0]) > 0.5  # axis 0 dominates


def test_contrastive_pca_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_contrastive_pca(np.zeros((5, 8), np.float32), np.zeros((3, 9), np.float32))


def test_contrastive_pca_transform_output_shape():
    proj = fit_contrastive_pca(_ring(60, dim=10), _shifted(20, dim=10))
    out = proj.transform(np.zeros((4, 10), dtype=np.float32))
    assert out.shape == (4, 2)
