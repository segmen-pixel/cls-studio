#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Collect the exact copyleft (LGPL/MPL) sources for a binary release.

Usage:
    python scripts/release/collect_lgpl_sources.py <staging_dir_or_manifest> \
        [--out DIR]

Reads the built installer's ``release_manifest.json`` (written by
scripts/build_installer.py), finds every shipped DLL that matches a pattern in
``lgpl_sources.json``, downloads the pinned upstream source archive for each,
verifies its SHA-256, and writes a release-attachable bundle:

    <out>/
      lgpl_sources_manifest.json   # DLL -> component/version/source mapping
      <component>-<version>.tar.*  # verified upstream source archives

Fail-closed behaviour (each aborts with exit 1):
  * a shipped DLL matches a component whose version/source_url/sha256 pin is
    not filled in yet,
  * a shipped DLL matches no component entry and is not in the reviewed
    non-copyleft list below.

The second gate is the point of the script. A binary release that quietly
omits a copyleft library's sources is a licence violation, and the failure
mode is silence -- nothing crashes, nothing looks wrong. So an unrecognised
DLL stops the release rather than being assumed harmless.

Attach the output directory as ``lgpl-sources-v<version>.zip`` to the GitHub
Release of the binary build.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import urllib.request
from pathlib import Path, PurePosixPath

# Console encoding: this script prints non-ASCII while reporting which DLLs it
# matched. On a non-UTF-8 console -- cp932 on a Japanese Windows install -- the
# default locale codec raises mid-report, so the release step that proves the
# copyleft obligation is discharged could not be run on the machine that builds
# the release.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
SOURCES_JSON = HERE / "lgpl_sources.json"

# DLLs that ship in the installer and are NOT copyleft. Every entry needs a
# reason, because each one narrows the gate above.
#
# Reviewed 2026-08-06 against the release_manifest.json of a win64 build.
# Patterns are scoped to the distribution that installs them: that is the unit
# a licence applies to, and it keeps a DLL appearing somewhere new falling
# through to triage instead of being swallowed by a broad name pattern.
KNOWN_NON_COPYLEFT: list[str] = [
    # --- the embedded CPython (python-build-standalone) ---
    "python/python3.dll",           # CPython itself, PSF-2.0
    "python/python31*.dll",         # ditto, versioned
    "python/vcruntime140*.dll",     # Microsoft VC++ runtime, redistributable
    "python/dlls/libcrypto-3-x64.dll",  # OpenSSL 3, Apache-2.0
    "python/dlls/libssl-3-x64.dll",     # OpenSSL 3, Apache-2.0
    "python/dlls/libffi-8.dll",         # libffi, MIT
    "python/dlls/sqlite3.dll",          # SQLite, public domain
    "python/dlls/tcl86t.dll",           # Tcl, BSD-style
    "python/dlls/tk86t.dll",            # Tk, BSD-style
    "python/tcl/*/*.dll",               # Tcl extensions (dde/reg/tix), BSD-style

    # --- numpy's bundled BLAS ---
    "python/lib/site-packages/numpy.libs/libscipy_openblas*.dll",  # OpenBLAS, BSD-3
    "python/lib/site-packages/numpy.libs/msvcp140*.dll",           # MSVC runtime

    # --- torch and the CUDA runtime it vendors ---
    # torch is BSD-3; the cu* / nv* DLLs are NVIDIA redistributables, already
    # covered by the allowlist in scripts/ci/dep-license-allowlist.txt; the
    # remaining ones are permissive (libiomp5 Apache-2.0-with-LLVM-exception,
    # zlibwapi zlib, uv MIT, fbgemm/asmjit BSD/Apache).
    "python/lib/site-packages/torch/lib/*.dll",
    "python/lib/site-packages/torch/bin/*.dll",

    # --- scipy / scikit-learn bundled runtimes ---
    "python/lib/site-packages/scipy.libs/libscipy_openblas*.dll",  # OpenBLAS, BSD-3
    "python/lib/site-packages/sklearn/.libs/msvcp140.dll",         # MSVC runtime
    "python/lib/site-packages/sklearn/.libs/vcomp140.dll",         # MSVC OpenMP runtime
]


def _matches(patterns: list[str], rel_path: str) -> bool:
    """Match a pattern against the file name or the whole staged path.

    Name-only matching cannot express "everything torch installed", and a
    per-file list of a few hundred DLLs is a list nobody re-reads. Path
    patterns let an entry name the distribution it covers, which is the unit
    a licence actually applies to -- while a DLL appearing somewhere new
    still falls through to the triage gate.
    """
    name = PurePosixPath(rel_path).name.lower()
    low = rel_path.replace("\\", "/").lower()
    return any(fnmatch.fnmatch(name, p.lower()) or fnmatch.fnmatch(low, p.lower())
               for p in patterns)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", help="installer staging dir or release_manifest.json")
    ap.add_argument("--out", default="dist/lgpl-sources")
    args = ap.parse_args()

    manifest_path = Path(args.staging)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "release_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found - build the installer first")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))

    dlls = [f for f in manifest["files"] if f["path"].lower().endswith(".dll")]
    print(f"{len(dlls)} DLLs in release manifest")

    matched: dict[str, list[dict]] = {}
    unmatched: list[str] = []
    for f in dlls:
        comp = next(
            (c for c in spec["components"] if _matches(c["dll_patterns"], f["path"])),
            None,
        )
        if comp is not None:
            matched.setdefault(comp["component"], []).append(f)
        elif not _matches(KNOWN_NON_COPYLEFT, f["path"]):
            unmatched.append(f["path"])

    if unmatched:
        print("DLLs with no copyleft mapping (triage each - add to "
              "lgpl_sources.json, or to KNOWN_NON_COPYLEFT with the licence "
              "you verified):")
        for p in unmatched:
            print(f"  ? {p}")
        print("ERROR: unclassified DLLs present - refusing to declare the "
              "copyleft source bundle complete")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = []
    incomplete = []
    for comp in spec["components"]:
        shipped = matched.get(comp["component"])
        if not shipped:
            continue  # component not present in this build
        if not (comp.get("version") and comp.get("source_url") and comp.get("sha256")):
            incomplete.append(comp["component"])
            continue
        archive = out_dir / Path(comp["source_url"]).name
        if not archive.exists() or _sha256(archive) != comp["sha256"]:
            print(f"  downloading {comp['component']} {comp['version']} source...")
            tmp = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(comp["source_url"], tmp)
            actual = _sha256(tmp)
            if actual != comp["sha256"]:
                tmp.unlink()
                print(f"ERROR: source hash mismatch for {comp['component']}: {actual}")
                return 1
            tmp.replace(archive)
        bundle.append({
            "component": comp["component"],
            "license": comp["license"],
            "version": comp["version"],
            "source_archive": archive.name,
            "source_sha256": comp["sha256"],
            "patches": comp.get("patches", []),
            "shipped_dlls": shipped,
        })

    if incomplete:
        print("ERROR: shipped copyleft DLLs whose source pin is not filled in "
              "(edit scripts/release/lgpl_sources.json):")
        for c in incomplete:
            print(f"  ! {c}")
        return 1

    (out_dir / "lgpl_sources_manifest.json").write_text(
        json.dumps({
            "cls_studio_version": manifest["version"],
            "platform": manifest["platform"],
            "components": bundle,
        }, indent=1),
        encoding="utf-8",
    )
    print(f"OK - {len(bundle)} component source(s) collected into {out_dir}")
    print("Attach this directory to the GitHub Release as "
          f"lgpl-sources-v{manifest['version']}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
