# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""OpenVINO IR export: the filename contract, and refusing what cannot be checked.

The conversion itself needs openvino installed, which is opt-in, so the tests
that need it are guarded. What is not guarded is the part that has to hold
either way: the loader already decides what an IR must be called and what
shape it must emit, and the exporter is the side that has to comply.
"""

from __future__ import annotations

import pytest

from clscore.backends.openvino_backbone import _ir_filename
from clscore.openvino_export import (
    OpenVINOUnavailable,
    export_ir,
    ir_filename,
    openvino_availability,
)

HAS_OV = openvino_availability()["available"]
needs_ov = pytest.mark.skipif(not HAS_OV, reason="openvino is not installed (opt-in)")


def test_the_exporter_names_the_file_the_loader_looks_for():
    """Two modules, one filename convention. If they drift the loader raises
    'export the backbone first' at a file that is sitting right there."""
    for backbone in ("dinov2_vitb14", "dinov2_vits14"):
        for bs in (1, 4):
            for prec in ("fp16", "fp32"):
                assert ir_filename(backbone, bs, prec) == _ir_filename(backbone, bs, prec)


def test_availability_has_the_same_shape_as_the_coreml_one():
    """The UI reads one structure whichever backend it is asking about."""
    from clscore.coreml_export import coreml_availability

    assert set(openvino_availability()) >= set(coreml_availability()) - {"version"}


def test_an_unknown_precision_is_refused():
    if not HAS_OV:
        pytest.skip("needs openvino to get past the availability check")
    with pytest.raises(OpenVINOUnavailable, match="precision"):
        export_ir(model_name="dinov2_vitb14", window=518, patch=14,
                  dest_dir="unused", precision="int4")


def test_a_backbone_nobody_asked_for_is_refused(monkeypatch):
    """Same allowlist as the Core ML path -- the model name comes out of a
    bank's metadata, and the hub repo carries backbones under non-commercial
    terms next to the ones this project uses."""
    if not HAS_OV:
        pytest.skip("needs openvino to get past the availability check")
    import torch

    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail("fetched anyway"))
    from clscore.coreml_export import CoreMLUnavailable

    with pytest.raises((CoreMLUnavailable, OpenVINOUnavailable), match="not an exportable"):
        export_ir(model_name="cell_dino_hpa_vitl16", window=518, patch=14, dest_dir="unused")


@needs_ov
def test_the_input_shape_is_pinned_to_the_geometry_the_bank_has(tmp_path):
    """convert_model leaves height and width dynamic. A graph that accepts any
    size is one that can be fed the wrong one and answer without complaint,
    and the bank's rows only exist in the 518/14 geometry."""
    import openvino as ov

    art = export_ir(model_name="dinov2_vitb14", window=518, patch=14,
                    dest_dir=tmp_path, samples=1, precision="fp16")
    shape = list(ov.Core().read_model(str(tmp_path / art["ir"])).inputs[0].get_partial_shape())
    assert [str(d) for d in shape] == ["1", "3", "518", "518"]
    assert art["output"]["shape"] == [1, 1369, 768]
    assert art["output"]["layout"] == "BND"


@needs_ov
def test_the_gate_measures_the_file_that_was_written(tmp_path):
    """Not the object it came from. compress_to_fp16 changes the artifact and
    not the in-memory model, so compiling the latter reported the fp32 numbers
    for an fp16 export -- a hundredfold difference, hidden."""
    fp32 = export_ir(model_name="dinov2_vitb14", window=518, patch=14,
                     dest_dir=tmp_path / "a", samples=2, precision="fp32")
    fp16 = export_ir(model_name="dinov2_vitb14", window=518, patch=14,
                     dest_dir=tmp_path / "b", samples=2, precision="fp16")
    assert fp32["parity"]["passed"] and fp16["parity"]["passed"]
    assert fp16["parity"]["rel_l2_max"] > fp32["parity"]["rel_l2_max"] * 10
    a = (tmp_path / "a" / fp32["ir"]).with_suffix(".bin").stat().st_size
    b = (tmp_path / "b" / fp16["ir"]).with_suffix(".bin").stat().st_size
    assert b < a * 0.6
