# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Edge export package: contents, manifest contract, and dequantisation.

The device never sees this code, so the tests stand in for it: they open the
zip the way a reader would, parse only ``.npy`` and JSON, and check that the
numbers come back close enough to the fp16 bank to score the same way.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from app.core.cls_edge_export import (
    ANOMALY_CLASS_COLOR,
    CLASSES_JSON,
    EDGE_FORMAT,
    IVF_CENTROIDS,
    IVF_ROW_CLUSTER,
    MANIFEST_NAME,
    MODEL_MANIFEST_JSON,
    NORMAL_CODES,
    NORMAL_SCALE,
    POSTPROCESS_KIND,
    build_edge_package,
)
from clscore.bank import Bank, BankMeta

RUNTIME_CONFIG_FILE = "runtime_config.json"
IVF_INDEX_FILE = "ivf_index.npz"


def _bank(rows: int = 64, dim: int = 16, *, with_critical: bool = True) -> Bank:
    rng = np.random.default_rng(0)
    # Small-mean blobs: large means make float distances lose precision to
    # cancellation, which would mask a real quantisation error.
    normal = rng.normal(0.0, 0.3, size=(rows, dim)).astype(np.float32)
    critical = {"scratch": rng.normal(0.0, 0.3, size=(8, dim)).astype(np.float32)} if with_critical else None
    meta = BankMeta(model="dinov2_vitb14", dim=dim, window=518, stride=256, patch=14)
    return Bank(normal, critical=critical, meta=meta)


ROOT = "b1"  # bank_id used throughout, and therefore the zip's top-level dir


def _read(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    import io

    return np.load(io.BytesIO(zf.read(f"{ROOT}/{name}")))


def _json(zf: zipfile.ZipFile, name: str) -> dict:
    return json.loads(zf.read(f"{ROOT}/{name}"))


def test_package_carries_the_encoder_contract(tmp_path):
    dest = tmp_path / "b.clsedge.zip"
    manifest = build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id=ROOT, dest=dest,
        runtime_config_file=RUNTIME_CONFIG_FILE,
    )
    assert manifest["format"] == EDGE_FORMAT
    # Without the encoder spec the features cannot be reproduced on device,
    # so every other field is worthless.
    assert manifest["encoder"] == {
        "model": "dinov2_vitb14", "dim": 16, "window": 518,
        "stride": 256, "patch": 14, "layers": None,
    }
    with zipfile.ZipFile(dest) as zf:
        assert _json(zf, MANIFEST_NAME) == manifest


def test_dequantised_rows_match_the_bank(tmp_path):
    bank = _bank()
    dest = tmp_path / "b.clsedge.zip"
    build_edge_package(
        bank=bank, bank_dir=tmp_path, bank_id=ROOT, dest=dest,
        runtime_config_file=RUNTIME_CONFIG_FILE,
    )
    with zipfile.ZipFile(dest) as zf:
        codes = _read(zf, NORMAL_CODES)
        scale = _read(zf, NORMAL_SCALE)
    assert codes.dtype == np.int8 and scale.dtype == np.float32
    restored = codes.astype(np.float32) * scale
    original = np.asarray(bank.normal, dtype=np.float32)
    # int8 over a symmetric per-dim range: the worst case is half a step.
    assert np.abs(restored - original).max() <= (np.abs(original).max(axis=0) / 127.0).max()


def test_images_and_full_precision_features_are_not_shipped(tmp_path):
    (tmp_path / "_images").mkdir()
    (tmp_path / "_images" / "taught.png").write_bytes(b"not-a-real-png")
    (tmp_path / "bank.npy").write_bytes(b"fp16-bank")
    dest = tmp_path / "b.clsedge.zip"
    build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id=ROOT, dest=dest,
        runtime_config_file=RUNTIME_CONFIG_FILE,
    )
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    assert not any("_images" in n for n in names)
    assert not any(n.endswith("/bank.npy") for n in names)
    # Everything sits under one directory so a host app can import the folder.
    assert all(n.startswith(f"{ROOT}/") for n in names), names


def test_verdict_recipe_rides_along_when_saved(tmp_path):
    (tmp_path / RUNTIME_CONFIG_FILE).write_text(
        json.dumps({"k": 3, "threshold": 12.5, "metric": "l2"}), encoding="utf-8"
    )
    manifest = build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
    )
    assert manifest["verdict"]["threshold"] == 12.5


def test_missing_verdict_recipe_is_null_not_an_error(tmp_path):
    manifest = build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
    )
    assert manifest["verdict"] is None


def test_exemplars_are_quantised_per_label(tmp_path):
    manifest = build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
        exemplars={"scratch": [(0, "img-a"), (2, "img-b"), (2, "img-b")]},
    )
    # The duplicate row must collapse: exemplars are a set of rows.
    assert manifest["exemplars"]["scratch"]["rows"] == 2
    with zipfile.ZipFile(tmp_path / "b.clsedge.zip") as zf:
        assert _read(zf, "exemplars/scratch.npy").dtype == np.int8


def test_out_of_range_exemplar_rows_are_dropped(tmp_path):
    manifest = build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
        exemplars={"scratch": [(0, "a"), (9999, "stale")]},
    )
    assert manifest["exemplars"]["scratch"]["rows"] == 1


def test_ivf_is_copied_when_the_index_matches(tmp_path):
    bank = _bank(rows=64, dim=16)
    np.savez_compressed(
        tmp_path / IVF_INDEX_FILE,
        centroids=np.random.default_rng(1).normal(size=(4, 16)).astype(np.float16),
        row_cluster=np.random.default_rng(2).integers(0, 4, size=64).astype(np.int32),
    )
    manifest = build_edge_package(
        bank=bank, bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
        ivf_index_file=IVF_INDEX_FILE, nprobe=8,
    )
    assert manifest["ivf"]["clusters"] == 4 and manifest["ivf"]["nprobe"] == 8
    with zipfile.ZipFile(tmp_path / "b.clsedge.zip") as zf:
        assert _read(zf, IVF_ROW_CLUSTER).shape == (64,)
        assert _read(zf, IVF_CENTROIDS).shape == (4, 16)


def test_stale_ivf_index_is_dropped_rather_than_shipped(tmp_path):
    # Row count moved on since the index was built: shipping it would send
    # the device to the wrong candidate rows.
    np.savez_compressed(
        tmp_path / IVF_INDEX_FILE,
        centroids=np.zeros((4, 16), dtype=np.float16),
        row_cluster=np.zeros(8, dtype=np.int32),
    )
    manifest = build_edge_package(
        bank=_bank(rows=64, dim=16), bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
        ivf_index_file=IVF_INDEX_FILE, nprobe=8,
    )
    assert manifest["ivf"] is None
    with zipfile.ZipFile(tmp_path / "b.clsedge.zip") as zf:
        assert f"{ROOT}/{IVF_CENTROIDS}" not in zf.namelist()


def test_empty_bank_is_refused(tmp_path):
    empty = Bank(np.zeros((0, 16), dtype=np.float32), meta=BankMeta(dim=16))
    with pytest.raises(ValueError, match="no normal rows"):
        build_edge_package(
            bank=empty, bank_dir=tmp_path, bank_id=ROOT,
            dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
        )


def test_host_sidecars_are_written_for_a_coreml_inspection_app(tmp_path):
    build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
        exemplars={"scratch": [(0, "a")]},
    )
    with zipfile.ZipFile(tmp_path / "b.clsedge.zip") as zf:
        classes = _json(zf, CLASSES_JSON)
        mm = _json(zf, MODEL_MANIFEST_JSON)

    # classes.json: class 0 is normal, defect labels share the anomaly colour.
    assert [c["id"] for c in classes["classes"]] == [0, 1]
    assert classes["classes"][0]["name"] == "normal"
    assert classes["classes"][1]["name"] == "scratch"
    assert classes["classes"][1]["color"] == ANOMALY_CLASS_COLOR

    # model_manifest.json: the displayed class count must NOT be the encoder's
    # channel width, or a host would read the feature map as class logits.
    assert mm["num_classes"] == 2
    assert mm["feature_dim"] == 16
    assert mm["logits_shape"] == [1, 16, 37, 37]
    assert mm["input_size"] == [518, 518]
    assert mm["postprocess"] == POSTPROCESS_KIND


def test_sidecars_name_a_generic_anomaly_class_when_no_labels_exist(tmp_path):
    build_edge_package(
        bank=_bank(with_critical=False), bank_dir=tmp_path, bank_id=ROOT,
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
    )
    with zipfile.ZipFile(tmp_path / "b.clsedge.zip") as zf:
        classes = _json(zf, CLASSES_JSON)
    assert [c["name"] for c in classes["classes"]] == ["normal", "anomaly"]


def test_bank_id_is_sanitised_into_the_directory_name(tmp_path):
    build_edge_package(
        bank=_bank(), bank_dir=tmp_path, bank_id="line 3/connector",
        dest=tmp_path / "b.clsedge.zip", runtime_config_file=RUNTIME_CONFIG_FILE,
    )
    with zipfile.ZipFile(tmp_path / "b.clsedge.zip") as zf:
        tops = {n.split("/")[0] for n in zf.namelist()}
    assert tops == {"line_3_connector"}
