# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The Core ML endpoints, on a machine that cannot do Core ML.

That is the normal case, not an edge case: this server runs on Windows and
coremltools publishes no Windows build. So the interesting behaviour is the
declining, and it has to decline in a way the UI can act on before the
operator presses anything.
"""

from __future__ import annotations

import pytest

from app.core.cls_model_export import encoder_args_from_meta, package_encoder
from clscore.coreml_export import coreml_availability


class _Meta:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_capability_is_reported_before_the_operator_presses_anything(client):
    """The UI disables the control on this, rather than offering it and
    failing afterwards."""
    r = client.get("/api/v1/hardware/coreml")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"available", "convert", "predict", "reason"}
    assert isinstance(body["available"], bool)
    if not body["available"]:
        assert body["reason"], "an unavailable capability has to say why"


def test_the_export_declines_with_a_reason_it_can_relay(client, project_id):
    if coreml_availability()["available"]:
        pytest.skip("this machine can convert; the declining path is the one under test")
    r = client.get("/api/v1/bank/export/coreml")
    # 501 once a bank is active, 4xx before that -- either way it must not be
    # a 500, and it must not produce a file.
    assert r.status_code != 200
    assert r.headers.get("content-type", "").startswith("application/json")


def test_an_unknown_precision_is_refused(client):
    r = client.get("/api/v1/bank/export/coreml", params={"precision": "int4"})
    assert r.status_code == 400


def test_the_encoder_is_taken_from_the_bank_not_from_a_default():
    """A package whose encoder does not match its rows is worse than none."""
    args = encoder_args_from_meta(
        _Meta(model="dinov2_vitl14", window=518, patch=14, layers=[9, 11])
    )
    assert args == {"model_name": "dinov2_vitl14", "window": 518, "patch": 14,
                    "layers": [9, 11]}


@pytest.mark.parametrize("meta", [
    _Meta(model=None, window=518, patch=14),
    _Meta(model="dinov2_vitb14", window=0, patch=14),
    _Meta(model="dinov2_vitb14", window=518, patch=0),
])
def test_a_bank_that_cannot_name_its_encoder_is_refused(meta):
    with pytest.raises(ValueError, match="does not record which encoder"):
        encoder_args_from_meta(meta)


def test_the_download_carries_the_parity_numbers_beside_the_model(tmp_path):
    """Whoever receives the zip can see what the export was gated on instead
    of taking the file on trust."""
    import json
    import zipfile

    pkg = tmp_path / "encoder.mlpackage"
    (pkg / "Data").mkdir(parents=True)
    (pkg / "Manifest.json").write_text("{}", encoding="utf-8")
    (pkg / "Data" / "weights.bin").write_bytes(b"\x00" * 16)

    dest = package_encoder(pkg, {"parity": {"passed": True}}, tmp_path / "out.zip")
    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        assert "encoder_artifact.json" in names
        assert "encoder.mlpackage/Manifest.json" in names
        assert "encoder.mlpackage/Data/weights.bin" in names
        assert json.loads(zf.read("encoder_artifact.json"))["parity"]["passed"] is True
