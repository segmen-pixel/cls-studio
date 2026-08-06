#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Build a cross-platform installer for cls-studio.

Supports Windows (embedded Python + CUDA) and macOS (standalone Python + MPS).

Usage:
    python scripts/build_installer.py                         # Current OS, lean
    python scripts/build_installer.py --platform win64        # Windows build
    python scripts/build_installer.py --platform macos-arm64  # Apple Silicon
    python scripts/build_installer.py --platform macos-x86    # Intel Mac
    python scripts/build_installer.py --inno                  # Inno Setup .exe (Windows)
    python scripts/build_installer.py --dmg                   # .dmg (macOS)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Windows still enforces MAX_PATH (260) for most APIs unless LongPathsEnabled
# is set, and the torch wheel carries a vendored licence tree 173 characters
# deep on its own. Staging inside the repo puts the prefix at ~101, which
# overflows -- pip fails halfway through and leaves a bundle missing most of
# its dependencies. Set CLS_BUILD_ROOT to a short path (the build prints what
# it needs) rather than moving the repo.
BUILD_DIR = Path(os.environ.get("CLS_BUILD_ROOT") or (ROOT / "build" / "installer"))

# The longest relative path we are willing to ship. 120 leaves room for an
# install root of ~135 characters, which covers Program Files and a user's
# Desktop; the raw torch tree at 173 does not.
MAX_STAGED_RELPATH = 120
DIST_DIR = ROOT / "dist"

# Dev venv to source fallback packages/licenses from. Prefer the cu128 build
# (Turing/Blackwell) when present; otherwise the standard .venv-windows.
_DEV_VENV_CU128 = ROOT / ".venv-windows-cu128"
DEV_VENV = _DEV_VENV_CU128 if (_DEV_VENV_CU128 / "Scripts" / "python.exe").exists() else ROOT / ".venv-windows"

PY_VERSION = "3.11.9"
PY_BUILD_TAG = "20240726"
_PBS_BASE = f"https://github.com/indygreg/python-build-standalone/releases/download/{PY_BUILD_TAG}"

# Platform configs  -  using python-build-standalone (full portable Python, multiprocessing works)
PLATFORMS = {
    "win64": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-win64.tar.gz",
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "py_exe": "python/python.exe",
        "label": "win64",
    },
    "macos-arm64": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-aarch64-apple-darwin-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-macos-arm64.tar.gz",
        "torch_index": "",
        "py_exe": "python/bin/python3",
        "label": "macos-arm64",
    },
    "macos-x86": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-x86_64-apple-darwin-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-macos-x86.tar.gz",
        "torch_index": "",
        "py_exe": "python/bin/python3",
        "label": "macos-x86",
    },
}


# Files/dirs to strip from installer (save ~1GB+)
STRIP_SITE_PACKAGES = {
    "dirs_remove": [
        # Build-only / not needed at runtime
        "torch/include", "torch/share/cmake",
        # Unused
        "pythonwin", "cv2/data",
        # Package managers (not needed after install)
        "pip", "setuptools", "_distutils_hack",
    ],
    "files_remove": [
        # distutils-precedence.pth triggers _distutils_hack import error after strip
        "distutils-precedence.pth",
        # Build-only .lib files (Windows)
        "torch/lib/*.lib",
        # Keep ALL CUDA DLLs — stripping caused silent crashes in packaged builds.
        # NOTE: Do NOT strip cudnn_engines_precompiled — required for conv2d
        # Build tools
        "torch/bin/protoc.exe", "torch/bin/protoc",
    ],
    "glob_remove": [
        # Type stubs
        "**/*.pyi",
        # Python cache
        "**/__pycache__",
        # Test directories (only top-level in each package, not 'testing' submodules)
        "**/tests",
    ],
}

# App dirs/files to exclude when copying
APP_EXCLUDE = shutil.ignore_patterns(
    "node_modules", "__pycache__", "*.pyc", ".git", "build",
    "e2e", "e2e-test*", "debug_screenshots",
    "playwright.config.ts", "tsconfig.json", "vite.config.mjs",
    "package-lock.json", "Dockerfile", "nginx.conf",
)

BUNDLE_ID = "com.segmen-pixel.cls-studio"


# ── Version ──

def _app_version() -> str:
    """Read version from pyproject.toml (single source of truth)."""
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            return m.group(1)
    # Fallback to package.json
    try:
        import json
        return json.loads((ROOT / "apps" / "ui" / "package.json").read_text(encoding="utf-8")).get("version", "0.9.0")
    except Exception:
        return "0.9.0"


# ── Helpers ──

def _detect_platform() -> str:
    if sys.platform == "win32":
        return "win64"
    elif sys.platform == "darwin":
        return "macos-arm64" if platform.machine() == "arm64" else "macos-x86"
    else:
        print("Linux not yet supported")
        sys.exit(1)


def step(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run(cmd: list[str], check: bool = False, **kw) -> int:
    print(f"  $ {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
    rc = subprocess.call(cmd, **kw)
    if check and rc != 0:
        raise SystemExit(
            f"  ABORT: command failed with exit code {rc}\n"
            f"    {' '.join(cmd)}\n"
            "  Refusing to keep building on top of a failed step."
        )
    return rc


def _lockfile_pin(package: str) -> str:
    """The exact version the lockfile pins, so nothing here can drift from it.

    The torch pin used to be written out twice -- once here and once in
    apps/api/requirements.txt -- with a comment asking whoever bumped one to
    remember the other. When they drifted, pip resolved the conflict by
    failing, and because the build ignored that, a 4 GB bundle shipped without
    fastapi in it. Reading the pin removes the second copy.
    """
    req = ROOT / "apps" / "api" / "requirements.txt"
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(rf"^{re.escape(package)}==([^\s;]+)", line, re.IGNORECASE)
        if m:
            return m.group(1)
    raise SystemExit(f"  ABORT: {package} is not pinned in {req}")


# What has to import for the app to start at all. A bundle missing any of these
# is a 4 GB download that opens on a stack trace, which is why this runs during
# the build rather than being left to whoever installs it.
RUNTIME_IMPORTS = ["fastapi", "uvicorn", "starlette", "pydantic", "sqlmodel",
                   "torch", "cv2", "sklearn", "numpy", "PIL", "zarr"]


def _verify_runtime_deps(py_exe: Path, torch_pin: str) -> None:
    """Import every runtime dependency inside the staged interpreter."""
    step("2b/6  Verifying the staged interpreter can import the app")
    probe = (
        "import importlib, sys\n"
        f"mods = {RUNTIME_IMPORTS!r}\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        missing.append('%s (%s)' % (m, e.__class__.__name__))\n"
        "import torch\n"
        "print('TORCH=' + torch.__version__)\n"
        "print('MISSING=' + ','.join(missing))\n"
    )
    out = subprocess.run([str(py_exe), "-c", probe], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"  ABORT: the staged interpreter is unusable\n{out.stderr[-800:]}")
    info = dict(
        line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line
    )
    missing = [m for m in info.get("MISSING", "").split(",") if m]
    if missing:
        raise SystemExit(
            "  ABORT: the staged bundle cannot import: " + ", ".join(missing)
            + "\n  The dependency install did not complete; shipping this "
              "would be a download that fails on first launch."
        )
    got = info.get("TORCH", "")
    if not got.startswith(torch_pin):
        raise SystemExit(
            f"  ABORT: staged torch is {got}, lockfile pins {torch_pin}"
        )
    print(f"  all {len(RUNTIME_IMPORTS)} runtime imports OK, torch {got}")


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  (cached: {dest.name})")
        return
    print(f"  Downloading {url.split('/')[-1]}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


# ── Platform-specific Python setup ──

def _setup_python_portable(staging: Path, cache_dir: Path, plat: dict) -> Path:
    """Setup python-build-standalone (full portable Python). Returns python exe path."""
    import tarfile
    archive = cache_dir / plat["py_archive"]
    download(plat["py_url"], archive)
    step("  Extracting Python...")
    with tarfile.open(archive) as tf:
        tf.extractall(staging)
    py_exe = staging / plat["py_exe"]
    if not py_exe.exists():
        raise FileNotFoundError(f"Python exe not found: {py_exe}")
    run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip", "--no-warn-script-location"])
    return py_exe


# ── Windows launchers ──

def _create_launcher_windows(staging: Path, version: str) -> None:
    (staging / "start.bat").write_text(
        '@echo off\r\n'
        'title cls-studio\r\n'
        'echo ==============================\r\n'
        f'echo   cls-studio v{version}\r\n'
        'echo ==============================\r\n'
        'echo.\r\n'
        'set "PYTHONPATH=%~dp0;%~dp0packages;%~dp0python\\Lib\\site-packages"\r\n'
        'set "PATH=%~dp0python;%~dp0python\\Scripts;%PATH%"\r\n'
        'cd /d "%~dp0"\r\n'
        '"%~dp0python\\python.exe" "%~dp0start.py"\r\n'
        'pause\r\n', encoding="utf-8")
    # Copy the committed launcher assets from build/installer/launcher/ (that
    # path is whitelisted in .gitignore precisely so the installer stays
    # reproducible from a clean clone; everything else under build/ is scratch).
    launcher_dir = ROOT / "build" / "installer" / "launcher"
    src = launcher_dir / "start.py"
    if not src.exists():
        raise FileNotFoundError(
            f"Required launcher asset missing: {src}. It is committed to the "
            f"repository — restore it from git before building the installer."
        )
    shutil.copy2(src, staging / "start.py")
    print("  Copied start.py from build/installer/launcher/")
    # The icon is optional: without it Windows falls back to the generic
    # console icon, which is cosmetic, not a build failure.
    icon = launcher_dir / "cls-studio.ico"
    if icon.exists():
        shutil.copy2(icon, staging / "cls-studio.ico")
        print("  Copied cls-studio.ico from build/installer/launcher/")
    else:
        print("  No cls-studio.ico found - the launcher will use the default icon")
    (staging / "cls-studio.bat").write_text(
        '@echo off\r\n'
        f'title cls-studio v{version}\r\n'
        'start "" "http://localhost:8791/ui/"\r\n'
        'call "%~dp0start.bat"\r\n', encoding="utf-8")


# ── macOS launchers ──

def _create_launcher_mac(staging: Path, version: str) -> None:
    script = (
        '#!/bin/bash\n'
        f'echo "cls-studio v{version}"\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'cd "$DIR"\n'
        'export PATH="$DIR/python/bin:$PATH"\n'
        'export PYTHONPATH="$DIR:$DIR/packages:$PYTHONPATH"\n'
        'open "http://localhost:8791/ui/" &\n'
        '"$DIR/python/bin/python3" -m uvicorn apps.api.app.main:app '
        '--host 127.0.0.1 --port 8791\n'
    )
    launcher = staging / "cls-studio.command"
    launcher.write_text(script, encoding="utf-8")
    launcher.chmod(0o755)
    start = staging / "start.sh"
    start.write_text(script, encoding="utf-8")
    start.chmod(0o755)


# ── macOS .app bundle ──

def _create_app_bundle(staging: Path, version: str, plat_name: str) -> Path:
    """Create a macOS .app bundle that wraps the staging directory."""
    app_dir = BUILD_DIR / "cls-studio.app"
    if app_dir.exists():
        shutil.rmtree(app_dir)

    contents = app_dir / "Contents"
    macos_dir = contents / "MacOS"
    resources = contents / "Resources"
    macos_dir.mkdir(parents=True)
    resources.mkdir(parents=True)

    # --- Info.plist ---
    short_version = version.split("-")[0]  # e.g. "0.9.1" from "0.9.1-beta"
    info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>cls-studio</string>
    <key>CFBundleDisplayName</key>
    <string>cls-studio</string>
    <key>CFBundleIdentifier</key>
    <string>{BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>CFBundleShortVersionString</key>
    <string>{short_version}</string>
    <key>CFBundleExecutable</key>
    <string>cls-studio</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>SGST</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2026 cls-studio Project. Apache-2.0 License.</string>
</dict>
</plist>'''
    (contents / "Info.plist").write_text(info_plist, encoding="utf-8")

    # --- Executable (shell script launcher) ---
    launcher_script = f'''#!/bin/bash
# cls-studio v{version}  -  macOS App Launcher
# This script is the CFBundleExecutable inside the .app bundle.

# Resolve the Resources directory (where app files live)
RESOURCES_DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"
APP_DIR="$RESOURCES_DIR/app"

export PATH="$APP_DIR/python/bin:$PATH"
export PYTHONPATH="$APP_DIR:$APP_DIR/packages:${{PYTHONPATH:-}}"

# Create projects directory
PROJECTS_DIR="$HOME/Documents/ClsStudio/projects"
mkdir -p "$PROJECTS_DIR"
export CLS_PROJECTS_DIR="$PROJECTS_DIR"
export CLS_DB_PATH="$PROJECTS_DIR/app.db"

# Open browser after a short delay
(sleep 2 && open "http://localhost:8791/ui/") &

# Launch the server
exec "$APP_DIR/python/bin/python3" -m uvicorn \\
    apps.api.app.main:app \\
    --host 127.0.0.1 --port 8791
'''
    launcher = macos_dir / "cls-studio"
    launcher.write_text(launcher_script, encoding="utf-8")
    launcher.chmod(0o755)

    # --- Copy app files into Resources/app/ ---
    app_dest = resources / "app"
    print(f"  Copying staging → {app_dest.name}/ ...")
    shutil.copytree(staging, app_dest, dirs_exist_ok=True)

    # --- Icon placeholder ---
    icon_src = ROOT / "icon.icns"
    if icon_src.exists():
        shutil.copy2(icon_src, resources / "AppIcon.icns")
        print("  Copied AppIcon.icns")
    else:
        print("  WARNING: icon.icns not found in repo root  -  app will use default icon")

    print(f"  Created: {app_dir}")
    return app_dir


def _create_dmg(app_dir: Path, version: str, plat_label: str) -> Path:
    """Create a .dmg from the .app bundle with Applications symlink."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    dmg_name = f"cls-studio-v{version}-{plat_label}"
    dmg_path = DIST_DIR / f"{dmg_name}.dmg"

    # Remove old dmg
    if dmg_path.exists():
        dmg_path.unlink()

    # Create a temporary directory for the DMG contents
    dmg_staging = BUILD_DIR / "dmg_staging"
    if dmg_staging.exists():
        shutil.rmtree(dmg_staging)
    dmg_staging.mkdir(parents=True)

    # Copy .app into dmg staging
    shutil.copytree(app_dir, dmg_staging / "cls-studio.app", symlinks=True)

    # Create Applications symlink (drag-to-install UX)
    os.symlink("/Applications", str(dmg_staging / "Applications"))

    # Create a README
    (dmg_staging / "README.txt").write_text(
        f"cls-studio v{version}\n"
        f"{'=' * 40}\n\n"
        "Drag cls-studio.app to the Applications folder to install.\n\n"
        "After installation, launch cls-studio from your Applications folder\n"
        "or Spotlight (Cmd+Space, type 'cls-studio').\n\n"
        "Your project data is stored in ~/Documents/ClsStudio/projects/\n"
        "and will NOT be removed when you delete the app.\n\n"
        "License: Apache-2.0\n"
        "https://github.com/segmen-pixel/cls-studio\n",
        encoding="utf-8",
    )

    # Use hdiutil to create the DMG
    print(f"  Creating DMG: {dmg_path.name}")
    result = subprocess.run([
        "hdiutil", "create",
        "-volname", f"cls-studio v{version}",
        "-srcfolder", str(dmg_staging),
        "-ov",
        "-format", "UDZO",  # compressed
        str(dmg_path),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  hdiutil error: {result.stderr}")
        print("  Falling back to ZIP...")
        return _create_zip_fallback(app_dir, version, plat_label)

    # Cleanup staging
    shutil.rmtree(dmg_staging, ignore_errors=True)

    size_mb = dmg_path.stat().st_size / 1024 / 1024
    print(f"  DMG created: {dmg_path} ({size_mb:.0f} MB)")
    return dmg_path


def _create_zip_fallback(app_dir: Path, version: str, plat_label: str) -> Path:
    """Fallback: ZIP the .app bundle if hdiutil is unavailable."""
    zip_path = DIST_DIR / f"cls-studio-v{version}-{plat_label}.zip"
    print(f"  Creating ZIP: {zip_path.name}")
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in app_dir.rglob("*"):
            if fp.is_file():
                arcname = f"cls-studio.app/{fp.relative_to(app_dir)}"
                zf.write(fp, arcname)
                count += 1
                if count % 3000 == 0:
                    print(f"    {count} files...")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  ZIP created: {zip_path} ({size_mb:.0f} MB, {count} files)")
    return zip_path


# ── Windows Inno Setup ──

def _write_inno_script(staging: Path, version: str) -> Path:
    """Generate an Inno Setup script with full version display."""
    suffix = ""
    icon_line = ""
    icon_src = staging / "icon.ico"
    if not icon_src.exists():
        # Try repo root
        repo_icon = ROOT / "icon.ico"
        if repo_icon.exists():
            shutil.copy2(repo_icon, icon_src)
    if icon_src.exists():
        icon_line = f'SetupIconFile={staging}\\icon.ico'

    license_line = ""
    license_file = ROOT / "LICENSE"
    if license_file.exists():
        license_line = f'LicenseFile={license_file}'

    iss_content = f'''; cls-studio v{version}  -  Inno Setup Script (auto-generated by build_installer.py)

[Setup]
AppId={{{{04666EF4-6968-4C5E-814B-76A1B277B768}}}}
AppName=Cls-Studio
AppVersion={version}
AppVerName=cls-studio v{version}
AppPublisher=cls-studio Project
AppPublisherURL=https://github.com/segmen-pixel/cls-studio
AppSupportURL=https://github.com/segmen-pixel/cls-studio/issues
DefaultDirName={{localappdata}}\\Programs\\cls-studio
DefaultGroupName=Cls-Studio
DisableProgramGroupPage=yes
OutputDir={DIST_DIR}
OutputBaseFilename=cls-studio-v{version}{suffix}-win64-setup
{license_line}
{icon_line}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
ShowLanguageDialog=yes
PrivilegesRequired=lowest
ChangesEnvironment=yes
UninstallDisplayName=cls-studio v{version}
VersionInfoVersion={version.split("-")[0]}.0
VersionInfoDescription=Cls-Studio Visual Inspection Toolkit v{version}
VersionInfoProductName=Cls-Studio
VersionInfoProductVersion={version}

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\\Japanese.isl"
Name: "english";  MessagesFile: "compiler:Default.isl"

[CustomMessages]
japanese.LaunchApp=cls-studio v{version} を起動する
japanese.CreateDesktopIcon=デスクトップにショートカットを作成する(&D)
japanese.ProjectsDirInfo=プロジェクトデータは以下に保存されます:%n%n  {{userdocs}}\\cls-studio\\projects%n%nこのフォルダはアンインストール時に削除されません。
english.LaunchApp=Launch cls-studio v{version}
english.CreateDesktopIcon=Create a &desktop shortcut
english.ProjectsDirInfo=Project data will be stored in:%n%n  {{userdocs}}\\cls-studio\\projects%n%nThis folder will NOT be removed on uninstall.

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "{staging}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{{userdocs}}\\cls-studio";          Flags: uninsneveruninstall
Name: "{{userdocs}}\\cls-studio\\projects"; Flags: uninsneveruninstall

[Icons]
Name: "{{group}}\\cls-studio v{version}";                Filename: "{{app}}\\cls-studio.bat"; Comment: "cls-studio v{version}"
Name: "{{group}}\\cls-studio をアンインストール"; Filename: "{{uninstallexe}}"
Name: "{{userdesktop}}\\cls-studio";           Filename: "{{app}}\\cls-studio.bat"; Tasks: desktopicon; Comment: "cls-studio v{version}"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "CLS_PROJECTS_DIR"; ValueData: "{{userdocs}}\\cls-studio\\projects"; Flags: uninsdeletevalue

[Run]
Filename: "cmd.exe"; Parameters: "/c mkdir ""{{userdocs}}\\cls-studio\\projects"" 2>nul"; Flags: runhidden; StatusMsg: "Creating project folder..."
Filename: "{{app}}\\cls-studio.bat"; Description: "{{cm:LaunchApp}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{{app}}"

[Code]
procedure InitializeWizard();
var
  InfoPage: TOutputMsgWizardPage;
begin
  InfoPage := CreateOutputMsgPage(
    wpSelectDir,
    'プロジェクトデータ / Project Data',
    'cls-studio v{version}',
    ExpandConstant('{{cm:ProjectsDirInfo}}')
  );
end;
'''
    iss_path = BUILD_DIR / "cls-studio.iss"
    iss_path.parent.mkdir(parents=True, exist_ok=True)
    iss_path.write_text(iss_content, encoding="utf-8-sig")  # BOM required for Inno Setup
    return iss_path


# ── Redistribution guard ──

def _assert_no_gpl_binaries(staging: Path) -> None:
    """Refuse to package a bundle that would redistribute GPL binaries.

    cls-studio is Apache-2.0 and uses no video I/O at all (no VideoCapture /
    VideoWriter anywhere), but the opencv-python-headless wheel bundles FFmpeg
    regardless — and the two platforms bundle DIFFERENT builds:

      Windows  opencv_videoio_ffmpeg*_64.dll, OpenCV's own LGPL-2.1+ build.
               Verified 2026-07-31: no --enable-gpl / libx264 / libx265.
      macOS    a Homebrew FFmpeg built --enable-gpl --enable-libx264
               --enable-libx265, shipped as delocated dylibs together with
               libx264 / libx265 / libxvid / libopencore-amr. Those are
               GPL-2.0-or-later.

    Stripping them is not an option: cv2.abi3.so links libavcodec/libavformat/
    libavutil directly through @loader_path, so removing them breaks import.
    Shipping them means an Apache-2.0 artifact redistributing GPL binaries with
    no GPL compliance — so the build stops here instead. macOS users install
    from source (install-macos.sh), where pip fetches the wheel on their own
    machine and we redistribute nothing.
    """
    gpl_markers = ("libx264", "libx265", "libxvid", "libopencore-amr")
    hits = sorted(
        {p.name for p in staging.rglob("*.dylib") if any(m in p.name for m in gpl_markers)}
    )
    if hits:
        raise SystemExit(
            "refusing to package: GPL-licensed binaries found in the bundle — "
            + ", ".join(hits)
            + "\nThese arrive inside the opencv-python-headless macOS wheel "
            "(FFmpeg built with --enable-gpl) and cannot be stripped, because "
            "cv2 links FFmpeg directly. cls-studio is Apache-2.0 and ships no "
            "GPL compliance material, so this artifact must not be "
            "distributed. Use the from-source install path on macOS "
            "(install-macos.sh) until an FFmpeg-free OpenCV build is pinned."
        )


# ── Strip unnecessary files ──

def _strip_installer(staging: Path, plat_name: str) -> None:
    """Remove unnecessary files to reduce installer size."""
    step("Stripping unnecessary files")
    sp = staging / "python" / ("Lib" if plat_name == "win64" else "lib/python3.12") / "site-packages"
    if not sp.exists():
        for candidate in staging.glob("python/lib/python*/site-packages"):
            sp = candidate
            break

    if not sp.exists():
        print("  site-packages not found, skipping strip")
        return

    removed_bytes = 0

    for d in STRIP_SITE_PACKAGES["dirs_remove"]:
        target = sp / d
        if target.exists():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target)
            removed_bytes += size

    import glob as _glob
    for pattern in STRIP_SITE_PACKAGES["files_remove"]:
        for f in _glob.glob(str(sp / pattern)):
            p = Path(f)
            if p.exists():
                removed_bytes += p.stat().st_size
                p.unlink()

    for pattern in STRIP_SITE_PACKAGES["glob_remove"]:
        for p in sp.glob(pattern):
            if p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
                removed_bytes += size
            elif p.is_file():
                removed_bytes += p.stat().st_size
                p.unlink()

    # Strip ui dev files (keep only dist/)
    ui = staging / "apps" / "ui"
    for d in ["src", "public", "scripts", "debug_screenshots"]:
        t = ui / d
        if t.exists():
            size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
            shutil.rmtree(t)
            removed_bytes += size

    print(f"  Stripped {removed_bytes / 1024 / 1024:.0f} MB")


# ── Write version file ──

def _collapse_deep_license_trees(staging: Path) -> None:
    """Zip up licence trees that are too deep to extract safely on Windows.

    torch vendors its dependencies' licences as a directory tree
    (kineto/dynolog/prometheus-cpp/civetweb/...) that is 173 characters deep.
    Nothing reads those files at runtime, but they are the licence texts of
    everything torch bundles, so deleting them would drop an attribution we
    are obliged to ship. Zipping the tree in place keeps every byte and makes
    the depth someone else's problem -- theirs only if they open it.
    """
    step("Collapsing deep licence trees")
    collapsed = 0
    for info in sorted(staging.rglob("*.dist-info")):
        tree = info / "licenses" / "third_party"
        if not tree.is_dir():
            continue
        archive = info / "licenses" / "third_party_licenses.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(tree.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(tree).as_posix())
        shutil.rmtree(tree)
        collapsed += 1
        print(f"  {info.name}: licences/third_party -> {archive.name}")
    if not collapsed:
        print("  nothing deep enough to collapse")


def _assert_path_lengths(staging: Path) -> None:
    """Refuse to ship a tree that cannot be unzipped at a normal install path."""
    worst = []
    for f in staging.rglob("*"):
        rel = f.relative_to(staging).as_posix()
        if len(rel) > MAX_STAGED_RELPATH:
            worst.append((len(rel), rel))
    if worst:
        worst.sort(reverse=True)
        lines = "\n".join(f"    {n:>4}  {r}" for n, r in worst[:10])
        raise SystemExit(
            f"  ABORT: {len(worst)} staged path(s) longer than "
            f"{MAX_STAGED_RELPATH} characters:\n{lines}\n"
            "  Windows extraction fails once the install root pushes these "
            "past 260. Collapse or drop them before packaging."
        )
    print(f"  longest staged path is within {MAX_STAGED_RELPATH} characters")


def _write_version_file(staging: Path, version: str, plat_name: str) -> None:
    """Write a VERSION file into the staging dir for runtime version display."""
    (staging / "VERSION").write_text(version, encoding="utf-8")
    print(f"  VERSION file: {version}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_release_manifest(staging: Path, version: str, plat_name: str) -> None:
    """Write release_manifest.json — path, size, SHA-256 of every staged file.

    This is the artifact-level ground truth that the SBOM and license audits
    diff against ("is every shipped file accounted for?"), and it is what
    scripts/release/collect_lgpl_sources.py reads to decide which copyleft
    sources have to accompany the binary. Generated after staging is final,
    and it ships inside the package itself.
    """
    entries = []
    for fp in sorted(staging.rglob("*")):
        if fp.is_file():
            entries.append({
                "path": fp.relative_to(staging).as_posix(),
                "size": fp.stat().st_size,
                "sha256": _sha256(fp),
            })
    manifest = {
        "name": "cls-studio",
        "version": version,
        "platform": plat_name,
        "file_count": len(entries),
        "files": entries,
    }
    (staging / "release_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"  release_manifest.json: {len(entries)} files hashed")


# ── Main build ──

def build(plat_name: str, inno: bool = False, dmg: bool = False) -> None:
    global DIST_DIR
    plat = PLATFORMS[plat_name]
    version = _app_version()
    is_win = plat_name == "win64"
    is_mac = plat_name.startswith("macos")
    # Organize output by version so past releases are preserved
    DIST_DIR = ROOT / "dist" / f"v{version}"
    print(f"Building cls-studio v{version} for {plat_name}")

    staging = BUILD_DIR / "cls-studio"
    if staging.exists():
        try:
            shutil.rmtree(staging)
        except (PermissionError, OSError) as e:
            # Windows: DLLs may be locked by a previous process.
            # Use a timestamped staging directory instead.
            import time
            alt = BUILD_DIR / f"cls-studio-{int(time.time())}"
            print(f"  WARNING: Cannot clean staging dir ({e.__class__.__name__}). Using {alt.name}")
            staging = alt
    staging.mkdir(parents=True)
    cache_dir = BUILD_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Python (python-build-standalone)
    step(f"1/7  Python ({plat_name})")
    py_exe = _setup_python_portable(staging, cache_dir, plat)

    # 2. Dependencies
    # Pinning policy (CONTRIBUTING.md, "Pinning git+URL dependencies"):
    # every install below must reference an exact version or commit SHA so
    # release builds are reproducible and cannot silently ingest upstream
    # license changes. torch mirrors the lockfile
    # (apps/api/requirements.txt); keep the pins in sync
    # with the lockfile.
    # Bumping a pin is a dependency upgrade: re-confirm the upstream LICENSE
    # and re-run smoke tests.
    step("2/6  Python dependencies")
    torch_pin = _lockfile_pin("torch")
    pip_cmd = [str(py_exe), "-m", "pip", "install", "--no-warn-script-location"]
    torch_req = f"torch=={torch_pin}"
    if plat["torch_index"]:
        run(pip_cmd + [torch_req, "--index-url", plat["torch_index"]], check=True)
    else:
        run(pip_cmd + [torch_req], check=True)
    run(pip_cmd + ["-r", str(ROOT / "apps" / "api" / "requirements.txt")], check=True)
    _verify_runtime_deps(py_exe, torch_pin)

    # 3. App code
    step("3/6  App code")
    for d in ["apps", "packages"]:
        src = ROOT / d
        if src.exists():
            shutil.copytree(src, staging / d, dirs_exist_ok=True, ignore=APP_EXCLUDE)
    for f in ["LICENSE", "NOTICE", "README.md", "THIRD_PARTY_NOTICES.md"]:
        src = ROOT / f
        if src.exists():
            dst = staging / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    # Third-party license attributions (model weights, etc.) — Apache-2.0 §4(a)
    licenses_src = ROOT / "licenses"
    if licenses_src.exists():
        shutil.copytree(licenses_src, staging / "licenses", dirs_exist_ok=True)

    # Propagate license bundles from key wheels we ship inside the installer.
    # PyTorch's wheel LICENSE/NOTICE already aggregate NVIDIA cuDNN/cuBLAS/NCCL
    # and other third-party notices required by their respective redistribution
    # terms; we copy them out so end users can find the obligations text.
    bundled_licenses_dst = staging / "licenses" / "third_party" / "wheels"
    bundled_licenses_dst.mkdir(parents=True, exist_ok=True)
    _wheel_license_sources = [
        ("torch",       ["LICENSE", "NOTICE"], "PYTORCH"),
        ("opencv_python_headless", ["LICENSE", "LICENSE-3RD-PARTY.txt"], "OPENCV"),
        ("Pillow",      ["LICENSE"],             "PILLOW"),
    ]
    # Both layouts: Windows embeds python/Lib/site-packages, POSIX
    # python/lib/python3.X/site-packages. Hardcoding the Windows one made every
    # macOS build ship an EMPTY licenses/third_party/wheels/ while NOTICE said
    # the texts were bundled — a silent no-op, not a visible failure.
    _license_bases = [
        staging / "python" / "Lib" / "site-packages",
        *sorted(staging.glob("python/lib/python*/site-packages")),
        DEV_VENV / "Lib" / "site-packages",
        *sorted(DEV_VENV.glob("lib/python*/site-packages")),
    ]
    _missing_licenses: list[str] = []
    for pkg, files, label in _wheel_license_sources:
        copied = 0
        for base in _license_bases:
            dist_info = sorted(base.glob(f"{pkg}-*.dist-info"))
            if not dist_info:
                continue
            for fname in files:
                # Newer wheels put licence texts under dist-info/licenses/.
                for src in (dist_info[0] / fname, dist_info[0] / "licenses" / fname):
                    if src.exists():
                        safe = fname.replace(" ", "_").replace("/", "_")
                        shutil.copy2(src, bundled_licenses_dst / f"{label}-{safe}")
                        copied += 1
                        break
            if copied:
                break
        if not copied:
            _missing_licenses.append(pkg)
    if _missing_licenses:
        # Loud, not silent: NOTICE tells users these texts are in the bundle, so
        # an artifact without them is a distribution defect, not a warning.
        raise SystemExit(
            "license propagation found no licence files for: "
            + ", ".join(_missing_licenses)
            + " — the bundle would ship without the texts NOTICE promises"
        )

    # 4. UI
    step("4/6  UI")
    staging_ui_dist = staging / "apps" / "ui" / "dist"
    if not staging_ui_dist.exists():
        ui_dist = ROOT / "apps" / "ui" / "dist"
        if not ui_dist.exists():
            raise FileNotFoundError(
                f"UI dist/ missing: {ui_dist}. "
                f"Run 'npm run build' in apps/ui/ before building the installer."
            )
        shutil.copytree(ui_dist, staging_ui_dist)
        print("  Copied pre-built UI dist/")
    else:
        print("  UI dist/ already in staging")


    # 5b. DINOv2 teacher model (weights only — Apache-2.0).
    #
    # We deliberately do NOT bundle the upstream torch-hub source tree
    # (``facebookresearch_dinov2_main/``). Recent versions of that repository
    # mix Apache-2.0 with non-commercial license fragments (CC-BY-NC-4.0 /
    # FAIR Noncommercial under LICENSE_CELL_DINO_CODE and
    # LICENSE_XRAY_DINO_MODEL), which cannot be redistributed alongside an
    # Apache-2.0 OSS build. The checkpoint itself stays Apache-2.0 and is
    # safe to ship; the model-definition code is fetched at runtime via
    # ``torch.hub.load('facebookresearch/dinov2', ...)`` on first use.
    if True:
        step("5/6  DINOv2 checkpoint (weights only)")
        dinov2_dir = staging / "models" / "dinov2"
        dinov2_dir.mkdir(parents=True, exist_ok=True)
        dinov2_ckpt = dinov2_dir / "dinov2_vitb14_pretrain.pth"
        if not dinov2_ckpt.exists():
            local_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
            if local_ckpt.exists():
                print("  dinov2_vitb14_pretrain.pth (local cache copy)")
                shutil.copy2(local_ckpt, dinov2_ckpt)
            else:
                url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
                print("  dinov2_vitb14_pretrain.pth (downloading ~330MB)...")
                urllib.request.urlretrieve(url, str(dinov2_ckpt))
        else:
            print("  dinov2_vitb14_pretrain.pth already in staging")
        # NOTE: facebookresearch_dinov2_main/ source tree is intentionally NOT
        # bundled here. End users with internet access pick it up via
        # torch.hub on first call to load_dinov2_teacher(); air-gapped users
        # must fetch the Apache-2.0-only files themselves and place them
        # under ~/.cache/torch/hub/facebookresearch_dinov2_main/.

    # 6. Strip + version file
    _strip_installer(staging, plat_name)
    _collapse_deep_license_trees(staging)
    _assert_path_lengths(staging)
    _assert_no_gpl_binaries(staging)
    _write_version_file(staging, version, plat_name)
    _write_release_manifest(staging, version, plat_name)

    # 7. Launcher + package
    step("6/6  Launcher")
    if is_win:
        _create_launcher_windows(staging, version)
    else:
        _create_launcher_mac(staging, version)

    # Copy icon if available
    for icon_name in ["icon.ico", "icon.icns"]:
        icon_src = ROOT / icon_name
        if icon_src.exists():
            shutil.copy2(icon_src, staging / icon_name)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""

    # ── Windows: Inno Setup .exe ──
    if inno and is_win:
        step("6/6  Inno Setup installer")
        iss = _write_inno_script(staging, version)
        iscc = (
            shutil.which("iscc")
            or next((p for p in [
                r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
                r"C:\Program Files\Inno Setup 6\ISCC.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"),
            ] if Path(p).exists()), "ISCC.exe")
        )
        if Path(iscc).exists():
            rc = run([iscc, str(iss)])
            exe_path = DIST_DIR / f"cls-studio-v{version}{suffix}-win64-setup.exe"
            if rc == 0 and exe_path.exists():
                # Move with build number (DIST_DIR is already versioned)
                ver_dir = DIST_DIR
                ver_dir.mkdir(parents=True, exist_ok=True)
                build_num = 1
                while (ver_dir / f"cls-studio-v{version}{suffix}-win64-setup-b{build_num}.exe").exists():
                    build_num += 1
                final_path = ver_dir / f"cls-studio-v{version}{suffix}-win64-setup-b{build_num}.exe"
                shutil.move(str(exe_path), str(final_path))
                size_mb = final_path.stat().st_size / 1024 / 1024
                print(f"\n  Installer: {final_path} ({size_mb:.0f} MB)")
                return
            else:
                print(f"\n  ERROR: Inno Setup failed (exit code {rc})")
                sys.exit(1)
        else:
            print("  Inno Setup not found, falling back to ZIP")

    # ── macOS: .app bundle + .dmg ──
    if is_mac and dmg:
        step("6/6  macOS .app + .dmg")
        app_dir = _create_app_bundle(staging, version, plat_name)
        output = _create_dmg(app_dir, version, plat["label"])
        print(f"\n  Installer: {output}")
        return

    if is_mac:
        step("6/6  macOS .app + ZIP")
        app_dir = _create_app_bundle(staging, version, plat_name)
        output = _create_zip_fallback(app_dir, version, plat["label"])
        print(f"\n  Package: {output}")
        return

    # ── Fallback: ZIP ──
    step("6/6  Creating ZIP")
    basename = f"cls-studio-v{version}{suffix}-{plat['label']}"
    zip_path = DIST_DIR / f"{basename}.zip"
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fp in staging.rglob("*"):
            if fp.is_file():
                zf.write(fp, f"cls-studio/{fp.relative_to(staging)}")
                count += 1
                if count % 3000 == 0:
                    print(f"  {count} files...")
    size_gb = zip_path.stat().st_size / 1024 / 1024 / 1024
    print(f"\nPackage: {zip_path}")
    print(f"  {count} files, {size_gb:.2f} GB")
    print("\n=== Done! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build cls-studio installer")
    parser.add_argument("--platform", choices=list(PLATFORMS.keys()),
                        default=None, help="Target platform (default: auto-detect)")
    parser.add_argument("--inno", action="store_true",
                        help="Create Inno Setup .exe installer (Windows only)")
    parser.add_argument("--dmg", action="store_true",
                        help="Create .dmg installer (macOS only)")
    args = parser.parse_args()
    plat = args.platform or _detect_platform()
    build(plat, inno=args.inno, dmg=args.dmg)
