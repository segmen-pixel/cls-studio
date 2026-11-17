# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Edge inference package: everything a phone needs, nothing it does not.

``/bank/export`` ships the bank the way the server keeps it — fp16 features,
the taught source images, the eval cache. That archive is for moving a bank
between cls-studio installs, and it is far too heavy for a device.

This module writes the other kind of package. It carries the normal bank
already quantised to int8 (so the device never re-quantises), the IVF
centroids that narrow the search, one exemplar block per defect label, and a
manifest describing the encoder the features came from. Nothing here depends
on torch: a reader only needs to parse ``.npy`` and do a distance
computation.

The manifest is the contract. A device that cannot reproduce the encoder
described there cannot use the package at all — the features would live in a
different space — so ``encoder`` is written even when the rest is empty. See
``docs/edge_export.md`` for the format.

The zip holds one top-level directory so that unzipping it yields a folder a
host app can import as a unit. Alongside the cls-studio manifest it writes
``classes.json`` and ``model_manifest.json``, the two sidecars that CoreML
inspection apps in this family already read: a host that knows nothing about
cls-studio still gets the class names and colours right, and the arrays it
does not understand are simply files it never opens.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

EDGE_FORMAT = "cls-studio-edge/1"
EDGE_EXPORT_SUFFIX = ".clsedge.zip"

MANIFEST_NAME = "manifest.json"
NORMAL_CODES = "normal_codes.npy"
NORMAL_SCALE = "normal_scale.npy"
IVF_CENTROIDS = "ivf_centroids.npy"
IVF_ROW_CLUSTER = "ivf_row_cluster.npy"
EXEMPLAR_DIR = "exemplars"

# Host-app sidecars. Only these two filenames are read by the CoreML
# inspection apps in this family, so they carry the display-facing facts.
CLASSES_JSON = "classes.json"
MODEL_MANIFEST_JSON = "model_manifest.json"
# The host also decodes this one. It is where the feature-space contract goes:
# a device that normalises differently lands its queries in a different space
# than the bank, and nothing about that failure looks like an error.
PREPROCESS_JSON = "preprocess.json"

# Verdict classes as a host app sees them: normal is drawn transparent, so
# only the anomaly colour matters. Vermilion, not purple — the palette has to
# stay legible for red-green colour vision deficiency, where purple and blue
# are the pair that collapse.
NORMAL_CLASS_COLOR = [0, 0, 0]
ANOMALY_CLASS_COLOR = [213, 94, 0]  # #D55E00
IGNORE_INDEX = 255

# Marks this as an anomaly package rather than a segmentation one. A host that
# checks this key knows the channel dimension is a feature width, not a class
# count, and that scoring means "distance to the bank", not "argmax".
POSTPROCESS_KIND = "anomaly-knn"


def _npy_bytes(arr: np.ndarray) -> bytes:
    """Serialise one array in .npy format (v1, so any reader can parse it)."""
    import io

    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), version=(1, 0))
    return buf.getvalue()


def _encoder_spec(meta: Any) -> dict[str, Any]:
    """The feature-space contract: without this the package is unusable."""
    return {
        "model": getattr(meta, "model", None),
        "dim": int(getattr(meta, "dim", 0)),
        "window": int(getattr(meta, "window", 0)),
        "stride": int(getattr(meta, "stride", 0)),
        "patch": int(getattr(meta, "patch", 0)),
        "layers": getattr(meta, "layers", None),
    }


def _read_verdict_recipe(bank_dir: Path, filename: str) -> dict[str, Any] | None:
    """The saved verdict recipe, or None when the bank has never been tuned.

    A package without one is still valid — the device just has no threshold
    to apply, which is exactly the state the Operator UI shows as "—".
    """
    path = bank_dir / filename
    if not path.is_file():
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _host_classes(labels: list[str]) -> dict[str, Any]:
    """``classes.json`` as a CoreML inspection host expects it.

    Class 0 is normal and is drawn transparent, so every defect label shares
    the anomaly colour rather than getting one of its own: the verdict is
    binary, and the label only says *which kind* once something is flagged.
    """
    classes = [{"id": 0, "name": "normal", "color": NORMAL_CLASS_COLOR, "active": True}]
    for i, label in enumerate(labels, start=1):
        classes.append({"id": i, "name": label, "color": ANOMALY_CLASS_COLOR, "active": True})
    if not labels:
        classes.append({"id": 1, "name": "anomaly", "color": ANOMALY_CLASS_COLOR, "active": True})
    return {"version": 1, "ignore_index": IGNORE_INDEX, "classes": classes}


def _host_model_manifest(encoder: dict[str, Any], n_labels: int) -> dict[str, Any]:
    """``model_manifest.json`` as a CoreML inspection host expects it.

    ``num_classes`` is the count a host *displays* (normal plus the defect
    labels), which is deliberately not the encoder's channel width — that
    lives in ``feature_dim``. A host that treats the channel dimension as a
    class count would misread the encoder output, so ``postprocess`` says
    outright that this package is scored by distance, not by argmax.
    """
    window = encoder.get("window") or 0
    patch = encoder.get("patch") or 0
    grid = (window // patch) if window and patch else 0
    return {
        "input_size": [window, window],
        "output_stride": patch,
        "num_classes": 1 + max(1, n_labels),
        "feature_dim": encoder.get("dim"),
        # Not logits and not channels-first: the encoder emits one feature
        # vector per patch, laid out the way the server's own extractor
        # returns them. The key names come from the host's existing schema.
        "logits_layout": "NHWC",
        "logits_shape": [1, grid, grid, encoder.get("dim")],
        "postprocess": POSTPROCESS_KIND,
        "note": (
            "Encoder output is a patch-feature map, not class logits. Score by "
            "nearest-neighbour distance against the bank in this package."
        ),
    }


def _built_rows(index_path: Path) -> int:
    """Row count the k-means was fitted on, from the index's own metadata.

    Returns 0 when it cannot be read: the capacity block then says only that
    appending is possible, which is the honest answer for an index whose
    provenance we do not have.
    """
    try:
        with np.load(index_path, allow_pickle=False) as z:
            if "meta_json" not in z:
                return 0
            meta = json.loads(bytes(np.asarray(z["meta_json"]).tobytes()).decode("utf-8"))
        return int(meta.get("built_rows", 0))
    except (OSError, ValueError, KeyError, UnicodeDecodeError):
        return 0


def _tiling_spec(encoder: dict[str, Any]) -> dict[str, Any]:
    """How the server cut the image the bank rows came from.

    The device has to reproduce this or its row indices mean something else.
    Padding is bottom/right only and the last window of each axis is anchored
    to the edge rather than left short, so windows overlap more at the far
    edge than the stride alone implies.
    """
    window = int(encoder.get("window") or 0)
    patch = int(encoder.get("patch") or 0)
    grid = (window // patch) if window and patch else 0
    return {
        "window": window,
        "stride": int(encoder.get("stride") or 0),
        "patch": patch,
        "grid": grid,
        "rows_per_window": grid * grid,
        "pad": "reflect-101, bottom and right only, up to one window",
        "last_window": "anchored to the far edge",
        "row_order": "window-major, then row-major within the window grid",
    }


def _capacity_spec(rows: int, built_rows: int, rows_per_window: int) -> dict[str, Any]:
    """What is left before the device has to hand the bank back.

    Rows taught after the index was built ride on nearest-centroid assignment,
    which the server does too. That holds until the bank outgrows the size it
    was clustered at, and past that point the routing is answering with the
    wrong candidates rather than failing, so the ceiling is worth stating
    where the device can see it.
    """
    from clscore.compress import REBUILD_GROWTH

    if built_rows <= 0:
        # No index: the device scans everything, so nothing caps an append.
        return {
            "appendable": True,
            "limit": None,
            "reason": "no IVF index in this package; a full scan has no routing to invalidate",
        }
    ceiling = int(built_rows * REBUILD_GROWTH)
    remaining = max(0, ceiling - int(rows))
    return {
        "appendable": True,
        "rows": int(rows),
        "built_rows": int(built_rows),
        "rebuild_growth": float(REBUILD_GROWTH),
        "max_rows": ceiling,
        "rows_remaining": remaining,
        "windows_remaining": remaining // rows_per_window if rows_per_window else None,
        "note": (
            "Append rows by assigning each to its nearest centroid, quantised "
            "with the shipped per-dim scale. Past max_rows the clustering no "
            "longer covers the bank: return it for a rebuild."
        ),
    }


def build_edge_package(
    *,
    bank: Any,
    bank_dir: Path,
    bank_id: str,
    dest: Path,
    runtime_config_file: str,
    exemplars: dict[str, list[tuple[int, str]]] | None = None,
    ivf_index_file: str | None = None,
    nprobe: int | None = None,
) -> dict[str, Any]:
    """Write the edge package to ``dest`` and return its manifest.

    Args:
        bank: Loaded bank (``normal`` array, ``critical`` dict, ``meta``).
        bank_dir: Directory the bank lives in, for sidecar files.
        bank_id: Identifier recorded in the manifest.
        dest: Zip path to write.
        runtime_config_file: Filename of the verdict recipe inside bank_dir.
        exemplars: ``{label: [(row, source_key), ...]}`` from the exemplar
            selection the server already uses, so device and server agree on
            what counts as a defect example.
        ivf_index_file: Filename of the persisted IVF index, if one exists.
        nprobe: Probe count to record as the recommended default.

    Raises:
        ValueError: If the bank has no normal rows — there is nothing to
            score against, so a package would be silently useless.
    """
    from clscore.compress import quantize_int8
    from clscore.preprocess import preprocess_spec

    normal = np.asarray(bank.normal)
    if normal.size == 0 or normal.shape[0] == 0:
        raise ValueError("bank has no normal rows - teach at least one image first")

    codes, scale = quantize_int8(normal.astype(np.float32, copy=False))
    encoder = _encoder_spec(bank.meta)
    tiling = _tiling_spec(encoder)
    manifest: dict[str, Any] = {
        "format": EDGE_FORMAT,
        "bank_id": bank_id,
        "encoder": encoder,
        "preprocess": preprocess_spec(),
        "tiling": tiling,
        "normal": {
            "rows": int(codes.shape[0]),
            "dim": int(codes.shape[1]),
            "quantization": "int8-symmetric-per-dim",
            "dequantize": "value = code * scale[dim]",
        },
        "ivf": None,
        "exemplars": {},
        "verdict": _read_verdict_recipe(bank_dir, runtime_config_file),
    }

    # One top-level directory, so unzipping gives a host app a folder it can
    # import as a unit rather than loose files in whatever it unzipped into.
    root = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in bank_id) or "bank"

    def put(name: str, data: bytes | str) -> None:
        zf.writestr(f"{root}/{name}", data)

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        put(NORMAL_CODES, _npy_bytes(codes.astype(np.int8, copy=False)))
        put(NORMAL_SCALE, _npy_bytes(scale.astype(np.float32, copy=False)))

        # IVF is optional: a device with a small bank can scan everything.
        # Copy the centroids straight out of the persisted index rather than
        # re-clustering, so the device narrows to the same candidates the
        # server would.
        if ivf_index_file:
            idx_path = bank_dir / ivf_index_file
            if idx_path.is_file():
                try:
                    with np.load(idx_path) as z:
                        centroids = np.asarray(z["centroids"])
                        row_cluster = np.asarray(z["row_cluster"])
                    if row_cluster.shape[0] == codes.shape[0]:
                        put(IVF_CENTROIDS, _npy_bytes(centroids.astype(np.float16, copy=False)))
                        put(IVF_ROW_CLUSTER, _npy_bytes(row_cluster.astype(np.int32, copy=False)))
                        manifest["capacity"] = _capacity_spec(
                            int(codes.shape[0]), _built_rows(idx_path), tiling["rows_per_window"]
                        )
                        manifest["ivf"] = {
                            "clusters": int(centroids.shape[0]),
                            "nprobe": int(nprobe) if nprobe else None,
                            "note": "candidate rows = those whose cluster is among the nprobe nearest centroids",
                        }
                    # A stale index (row count moved on) is dropped rather
                    # than shipped: wrong candidates are worse than none.
                except (OSError, ValueError, KeyError):
                    pass

        if "capacity" not in manifest:
            manifest["capacity"] = _capacity_spec(int(codes.shape[0]), 0, tiling["rows_per_window"])

        shown = getattr(bank.meta, "label_display", {}) or {}
        for label, rows in (exemplars or {}).items():
            feats = bank.critical.get(label)
            if feats is None or len(rows) == 0:
                continue
            idx = np.asarray(sorted({int(r) for r, _ in rows if 0 <= int(r) < feats.shape[0]}), dtype=np.int64)
            if idx.size == 0:
                continue
            block = np.asarray(feats)[idx].astype(np.float32, copy=False)
            e_codes, e_scale = quantize_int8(block)
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in label) or "label"
            put(f"{EXEMPLAR_DIR}/{safe}.npy", _npy_bytes(e_codes.astype(np.int8, copy=False)))
            put(f"{EXEMPLAR_DIR}/{safe}.scale.npy", _npy_bytes(e_scale.astype(np.float32, copy=False)))
            manifest["exemplars"][label] = {
                "file": safe,
                "rows": int(e_codes.shape[0]),
                # What the operator called this class. The key beside it is
                # the on-disk stem, which is unique but unreadable for a
                # non-ASCII name, and this file is read on the device.
                "display": shown.get(label, label),
            }

        # Host-app sidecars last, once the label set is known.
        labels = sorted(manifest["exemplars"].keys())
        put(CLASSES_JSON, json.dumps(
            _host_classes([shown.get(lab, lab) for lab in labels]),
            ensure_ascii=False, indent=2,
        ))
        put(MODEL_MANIFEST_JSON, json.dumps(_host_model_manifest(encoder, len(labels)),
                                            ensure_ascii=False, indent=2))
        put(PREPROCESS_JSON, json.dumps(preprocess_spec(), ensure_ascii=False, indent=2))
        put(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))

    return manifest
