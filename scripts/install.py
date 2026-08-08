#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""cls-studio installer — sets up Python dependencies and builds the UI.

Everything installs from PyPI wheels; no git checkouts are required. The
DINOv2 weights are downloaded automatically on first use via torch.hub.

Usage:
    python scripts/install.py                  # Install deps + build UI
    python scripts/install.py --offline-pack DIR   # Wheel bundle for air-gapped installs
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "apps" / "api" / "requirements.txt"

# Torch CUDA index. Default cu128 (Turing/RTX 20xx and newer, incl. Blackwell).
# For older GPUs (Maxwell/Pascal/Volta, e.g. GTX 10xx / Tesla V100) use cu124.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def run(cmd: list[str], **kw) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, **kw)


def install_python_deps() -> None:
    print("\n=== Installing Python dependencies ===")
    if sys.platform == "darwin":
        # macOS: default PyPI wheels include MPS support (no CUDA index)
        run([sys.executable, "-m", "pip", "install", "torch"])
    else:
        # Windows/Linux: use CUDA index
        run([sys.executable, "-m", "pip", "install",
             "torch", "--index-url", TORCH_INDEX])
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    run([sys.executable, "-m", "pip", "install", "-e",
         str(ROOT / "packages" / "clscore")])


def install_ui_deps() -> None:
    ui_dir = ROOT / "apps" / "ui"
    if not (ui_dir / "package.json").exists():
        print("  (no UI package.json, skipping)")
        return
    print("\n=== Installing UI dependencies ===")
    run(["npm", "install"], cwd=str(ui_dir))
    print("\n=== Building UI ===")
    run(["npm", "run", "build"], cwd=str(ui_dir))


def create_offline_pack(out_dir: Path) -> None:
    """Download all wheels into a directory for offline install."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wheels_dir = out_dir / "wheels"
    wheels_dir.mkdir(exist_ok=True)

    # Apache-2.0 section 4(a)/(d): a distributed artifact carries the licence
    # and the NOTICE with it. The offline pack is a distribution channel.
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    print(f"\n=== Creating offline pack in {out_dir} ===")
    print("\n--- Downloading Python wheels ---")
    run([sys.executable, "-m", "pip", "download",
         "-r", str(REQUIREMENTS),
         "-d", str(wheels_dir),
         "--extra-index-url", TORCH_INDEX])

    (out_dir / "install_offline.py").write_text(
        '''#!/usr/bin/env python3
"""Install cls-studio from offline pack."""
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).parent
subprocess.call([sys.executable, "-m", "pip", "install",
    "--no-index", "--find-links", str(HERE / "wheels"),
    "-r", str(HERE.parent / "apps" / "api" / "requirements.txt")])
print("Done! Run: python -m uvicorn apps.api.app.main:app --port 8791")
print("Note: DINOv2 weights load via torch.hub on first use - for")
print("air-gapped machines, pre-populate ~/.cache/torch/hub/ manually.")
''',
        encoding="utf-8",
    )
    print(f"\nOffline pack ready: {out_dir}")
    print(f"  wheels: {len(list(wheels_dir.glob('*')))} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="cls-studio installer")
    parser.add_argument("--offline-pack", type=str, default="",
                        help="Create offline installation bundle at the given path")
    parser.add_argument("--skip-python", action="store_true",
                        help="Skip Python dependency installation")
    parser.add_argument("--skip-ui", action="store_true",
                        help="Skip UI build")
    args = parser.parse_args()

    print("cls-studio Installer")
    print(f"  Root: {ROOT}")
    print(f"  Python: {sys.version}")

    if args.offline_pack:
        create_offline_pack(Path(args.offline_pack))
        return

    if not args.skip_python:
        install_python_deps()

    if not args.skip_ui:
        install_ui_deps()

    print("\n=== Installation complete ===")
    print("Start the server:")
    print("  python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8791")
    print("  Then open: http://localhost:8791/ui/")


if __name__ == "__main__":
    main()
