# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Unit tests for IncidentMetaArray (per-patch metadata)."""

from __future__ import annotations

import numpy as np
import pytest

from clscore.incident import (
    DEFAULT_SEVERITY,
    DEFAULT_TIER,
    SEVERITY_HEAVY,
    SEVERITY_LIGHT,
    SEVERITY_MEDIUM,
    IncidentMetaArray,
)


def test_empty_has_zero_length():
    m = IncidentMetaArray.empty()
    assert len(m) == 0
    assert m.severity.dtype == np.uint8
    assert m.registered_at_inspection.dtype == np.uint64
    assert m.last_hit_at_inspection.dtype == np.uint64
    assert m.hit_count.dtype == np.uint32
    assert m.tier.dtype == np.uint8


def test_defaults_for_uses_medium_severity_and_short_tier():
    m = IncidentMetaArray.defaults_for(5, registered_at=42)
    assert len(m) == 5
    assert (m.severity == DEFAULT_SEVERITY).all()
    assert (m.tier == DEFAULT_TIER).all()
    assert (m.hit_count == 0).all()
    # registered == last_hit so the very first freshness check sees delta=0.
    assert (m.registered_at_inspection == 42).all()
    assert (m.last_hit_at_inspection == 42).all()


def test_append_extends_in_lockstep():
    m = IncidentMetaArray.empty()
    m.append(3, severity=SEVERITY_HEAVY, inspection_count=10)
    m.append(2, severity=SEVERITY_LIGHT, inspection_count=20)
    assert len(m) == 5
    assert m.severity.tolist() == [SEVERITY_HEAVY] * 3 + [SEVERITY_LIGHT] * 2
    assert m.registered_at_inspection.tolist() == [10, 10, 10, 20, 20]
    assert m.last_hit_at_inspection.tolist() == [10, 10, 10, 20, 20]
    assert m.hit_count.tolist() == [0] * 5


def test_append_clips_severity_out_of_range():
    """A buggy caller passing severity=99 must not poison the scale."""
    m = IncidentMetaArray.empty()
    m.append(1, severity=99, inspection_count=0)
    m.append(1, severity=-5, inspection_count=0)
    assert m.severity.tolist() == [SEVERITY_HEAVY, SEVERITY_LIGHT]


def test_append_zero_rows_is_a_noop():
    m = IncidentMetaArray.empty()
    m.append(0, severity=SEVERITY_MEDIUM, inspection_count=0)
    assert len(m) == 0


def test_take_subsets_in_parallel():
    m = IncidentMetaArray.defaults_for(4, registered_at=0)
    m.severity = np.array([1, 2, 3, 2], dtype=np.uint8)
    m.hit_count = np.array([10, 20, 30, 40], dtype=np.uint32)
    sub = m.take(np.array([0, 2]))
    assert sub.severity.tolist() == [1, 3]
    assert sub.hit_count.tolist() == [10, 30]


def test_save_load_roundtrip(tmp_path):
    m = IncidentMetaArray.empty()
    m.append(2, severity=SEVERITY_HEAVY, inspection_count=7)
    m.append(1, severity=SEVERITY_LIGHT, inspection_count=8)
    p = tmp_path / "scratch.meta.npz"
    m.save(p)
    loaded = IncidentMetaArray.load(p)
    assert loaded.severity.tolist() == m.severity.tolist()
    assert loaded.registered_at_inspection.tolist() == m.registered_at_inspection.tolist()
    assert loaded.last_hit_at_inspection.tolist() == m.last_hit_at_inspection.tolist()
    assert loaded.hit_count.tolist() == m.hit_count.tolist()
    assert loaded.tier.tolist() == m.tier.tolist()


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    m = IncidentMetaArray.defaults_for(3)
    p = tmp_path / "atomic.meta.npz"
    m.save(p)
    # ``.tmp`` sibling should never linger after a successful save.
    assert not p.with_suffix(p.suffix + ".tmp").exists()
    assert p.exists()


def test_assert_matches_passes_when_lengths_agree():
    m = IncidentMetaArray.defaults_for(3)
    m.assert_matches(3, "scratch")


def test_assert_matches_raises_on_mismatch():
    m = IncidentMetaArray.defaults_for(3)
    with pytest.raises(ValueError, match="does not match"):
        m.assert_matches(5, "scratch")
