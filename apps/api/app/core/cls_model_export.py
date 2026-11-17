# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Server-side wrapper around the encoder exports.

Each conversion lives in clscore so it can be driven from a script with no
server running -- on a Mac for Core ML, on this box for OpenVINO. This module
is only what the API adds on top: turning the bank's recorded encoder into
export arguments, and packaging the result into something a browser can
download.

Both artifacts are directories -- a .mlpackage is one by definition, an IR is
an .xml and a .bin that have to travel together. Handing either over HTTP
means zipping it, and the artifact record goes in beside it so whoever
receives it can see the parity numbers the export was gated on rather than
taking the file on trust.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

ENCODER_SUFFIX = ".clsencoder.zip"
IR_SUFFIX = ".clsir.zip"
ARTIFACT_NAME = "encoder_artifact.json"


def encoder_args_from_meta(meta: Any) -> dict[str, Any]:
    """Export arguments for the encoder a bank was actually built with.

    Read off the bank rather than defaulted: a package whose encoder does not
    match its rows is worse than no package, and the bank is the only thing
    that knows which one it used.
    """
    model = getattr(meta, "model", None)
    window = int(getattr(meta, "window", 0) or 0)
    patch = int(getattr(meta, "patch", 0) or 0)
    if not model or window <= 0 or patch <= 0:
        raise ValueError(
            "this bank does not record which encoder built it, so an exported "
            "encoder could not be matched to its rows"
        )
    return {"model_name": model, "window": window, "patch": patch,
            "layers": getattr(meta, "layers", None)}


def package_encoder(source: Path, artifact: dict[str, Any], dest: Path,
                    root: str | None = None) -> Path:
    """Zip a directory of model files with its artifact record beside it."""
    root = root or Path(source).name
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(Path(source).rglob("*")):
            if f.is_file():
                zf.write(f, arcname=f"{root}/{f.relative_to(source).as_posix()}")
        zf.writestr(ARTIFACT_NAME, json.dumps(artifact, ensure_ascii=False, indent=2))
    return dest
