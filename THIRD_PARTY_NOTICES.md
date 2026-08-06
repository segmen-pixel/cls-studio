# Third-Party Notices

Cls-Studio is licensed under Apache License 2.0 (see `LICENSE`). It bundles or
depends on the following third-party components. Each component is governed by
its own license; the terms below are summaries — refer to each project's source
for the authoritative text.

**Scope:** This file lists components that are *redistributed* with Cls-Studio,
either bundled in the Windows installer or fetched at install time on macOS.
Build- and development-only tools (pytest, ruff, eslint, type stub packages,
etc.) that are never shipped to end users are not listed here.

For the upstream NOTICE attributions required by Apache License 2.0 §4(d),
see `NOTICE` at the repository root.

Last updated: 2026-07-29 for **v0.1.0**. This file describes the release it
ships with: apart from the one exception called out inline (`torch`
re-installed from the PyTorch CUDA index by the Windows installer), every
entry below is pinned in this tag's lockfiles
(`apps/api/requirements.txt` and its CUDA / OpenVINO variants), and the license
recorded for each entry is the license the package publishes for that exact
pinned version.

Optional extras that no installer pulls in are **not** listed, because nothing
here redistributes them: `clscore[timm]` (timm, Apache-2.0) is installed by
the user only if they choose to run a BG-aware backbone.

---

## Bundled at runtime (included in installer)

### Embedded Python runtime

- **python-build-standalone** — MPL-2.0 (build scripts); the produced Python
  distribution aggregates multiple licenses — CPython (PSF-2.0) plus the
  licenses of bundled libraries (OpenSSL, libffi, etc.) as recorded in the
  distribution's own `PYTHON.json` / license metadata, which ships alongside
  the runtime in the installer.
  - https://github.com/astral-sh/python-build-standalone

### Python: core server

| Package | License | Source |
|---|---|---|
| fastapi | MIT | https://github.com/fastapi/fastapi |
| starlette | BSD-3-Clause | https://github.com/encode/starlette |
| uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |
| uvloop | MIT | https://github.com/MagicStack/uvloop |
| httptools | MIT | https://github.com/MagicStack/httptools |
| watchfiles | MIT | https://github.com/samuelcolvin/watchfiles |
| websockets | BSD-3-Clause | https://github.com/python-websockets/websockets |
| h11 | MIT | https://github.com/python-hyper/h11 |
| click | BSD-3-Clause | https://github.com/pallets/click |
| colorama | BSD-3-Clause | https://github.com/tartley/colorama |
| pydantic | MIT | https://github.com/pydantic/pydantic |
| pydantic-core | MIT | https://github.com/pydantic/pydantic-core |
| annotated-types | MIT | https://github.com/annotated-types/annotated-types |
| annotated-doc | MIT | https://github.com/tiangolo/annotated-doc |
| typing-inspection | MIT | https://github.com/pydantic/typing-inspection |
| typing-extensions | PSF-2.0 | https://github.com/python/typing_extensions |
| sqlmodel | MIT | https://github.com/fastapi/sqlmodel |
| sqlalchemy | MIT | https://github.com/sqlalchemy/sqlalchemy |
| greenlet | MIT AND PSF-2.0 | https://github.com/python-greenlet/greenlet |
| python-multipart | Apache-2.0 | https://github.com/Kludex/python-multipart |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| httpcore | BSD-3-Clause | https://github.com/encode/httpcore |
| anyio | MIT | https://github.com/agronholm/anyio |
| idna | BSD-3-Clause | https://github.com/kjd/idna |
| certifi | MPL-2.0 | https://github.com/certifi/python-certifi |
| jinja2 | BSD-3-Clause | https://github.com/pallets/jinja |
| markupsafe | BSD-3-Clause | https://github.com/pallets/markupsafe |
| python-dotenv | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| pyyaml | MIT | https://github.com/yaml/pyyaml |
| setuptools | MIT | https://github.com/pypa/setuptools |

### Python: ML / inference

| Package | License | Source |
|---|---|---|
| torch (PyTorch) | BSD-3-Clause | https://github.com/pytorch/pytorch |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://github.com/numpy/numpy |
| scipy | BSD-3-Clause | https://github.com/scipy/scipy |
| scikit-learn | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| joblib | BSD-3-Clause | https://github.com/joblib/joblib |
| threadpoolctl | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| pillow | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |
| opencv-python-headless | Apache-2.0 | https://github.com/opencv/opencv-python |
| sympy | BSD-3-Clause | https://github.com/sympy/sympy |
| mpmath | BSD-3-Clause | https://github.com/mpmath/mpmath |
| networkx | BSD-3-Clause | https://github.com/networkx/networkx |
| filelock | MIT | https://github.com/tox-dev/filelock |
| fsspec | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| triton (Linux x86-64 only) | MIT | https://github.com/triton-lang/triton |


### Python: OpenVINO extra (opt-in, `--with-openvino`)

Installed only when the Windows installer is run with `--with-openvino`, which
pip-installs `apps/api/requirements-openvino.txt` from PyPI on the user's own
machine. None of it is bundled, and a default install contains none of it.

| Package | License | Source |
|---|---|---|
| openvino | Apache-2.0 | https://github.com/openvinotoolkit/openvino |
| openvino-telemetry | Apache-2.0 | https://github.com/openvinotoolkit/telemetry |
| nncf | Apache-2.0 | https://github.com/openvinotoolkit/nncf |
| safetensors | Apache-2.0 | https://github.com/huggingface/safetensors |
| ninja | Apache-2.0 AND BSD-3-Clause | https://github.com/scikit-build/ninja-python-distributions |
| packaging | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| pygments | BSD-2-Clause | https://github.com/pygments/pygments |
| rich | MIT | https://github.com/Textualize/rich |
| markdown-it-py | MIT | https://github.com/executablebooks/markdown-it-py |
| mdurl | MIT | https://github.com/executablebooks/mdurl |
| narwhals | MIT | https://github.com/narwhals-dev/narwhals |
| pydot | MIT | https://github.com/pydot/pydot |
| pyparsing | MIT | https://github.com/pyparsing/pyparsing |
| tabulate | MIT | https://github.com/astanin/python-tabulate |

`joblib`, `networkx`, `numpy`, `scikit-learn`, `scipy` and `threadpoolctl` are
also pinned in that lockfile; they are already listed above.

> **Telemetry.** `openvino-telemetry` arrives transitively via `openvino` and
> `nncf`. Cls-Studio contains no telemetry code and never initialises that
> library — whether it reports anything depends entirely on the OpenVINO
> opt-in state on the machine, which OpenVINO manages itself. A default
> Cls-Studio install (without `--with-openvino`) does not contain the package
> at all.

### Python: data / storage

| Package | License | Source |
|---|---|---|
| zarr | MIT | https://github.com/zarr-developers/zarr-python |
| numcodecs | MIT | https://github.com/zarr-developers/numcodecs |
| asciitree | MIT | https://github.com/mbr/asciitree |
| fasteners | Apache-2.0 | https://github.com/harlowja/fasteners |
| deprecated | MIT | https://github.com/laurent-laporte-pro/deprecated |
| wrapt | BSD-2-Clause | https://github.com/GrahamDumpleton/wrapt |

### Model weights: DINOv2 (feature extractor)

- **DINOv2** — Apache-2.0 — Meta Platforms, Inc.
  - https://github.com/facebookresearch/dinov2
  - The pretrained weight file `dinov2_vitb14_pretrain.pth` is the **only**
    model weight redistributed with Cls-Studio. The installer copies it from
    the local torch hub cache, or downloads it from
    `https://dl.fbaipublicfiles.com/dinov2/` at build time
    (`scripts/build_installer.py`).

> ⚠️ **Source tree note.** Only the pretrained weight file is redistributed.
> The DINOv2 torch-hub source tree (`facebookresearch_dinov2_main/`) is **not**
> bundled, because recent revisions mix Apache-2.0 with non-commercial
> fragments that cannot be re-shipped under Apache-2.0. The model-definition
> Python files are fetched at runtime by `torch.hub.load` on the user's own
> machine, which keeps them outside Cls-Studio's redistribution surface.

---

## Frontend (`apps/ui`)

Only the packages whose code is compiled into the shipped UI bundle are listed;
the build toolchain (vite, typescript, eslint, playwright, type stubs) is
development-only and out of scope per the note above.

| Package | License | Source |
|---|---|---|
| react | MIT | https://github.com/facebook/react |
| react-dom | MIT | https://github.com/facebook/react |
| scheduler | MIT | https://github.com/facebook/react |
| Feather Icons (SVG path data) | MIT | https://github.com/feathericons/feather |

`loose-envify` and `js-tokens` appear in the lockfile as react's own
dependencies but are build-time transforms only — no code of theirs reaches
the bundle.

---

## Fonts, icons, and media

- Cls-Studio icon (`build/installer/launcher/cls-studio.ico`,
  `apps/ui/public/favicon.ico`): © Cls-Studio contributors, distributed under
  Apache-2.0 (part of this project).
- Any web fonts loaded at runtime are served by the user's browser from their
  standard sources; no fonts are redistributed by this project.

---

## NVIDIA libraries via PyTorch CUDA wheel

The CUDA builds ship the PyTorch CUDA wheel, which transitively bundles the
NVIDIA libraries pinned as `nvidia-*-cu12` in `apps/api/requirements.txt`,
under their respective redistributable licenses. As of this tag that is:

- cuBLAS (`nvidia-cublas-cu12`)
- CUPTI (`nvidia-cuda-cupti-cu12`)
- NVRTC (`nvidia-cuda-nvrtc-cu12`)
- CUDA Runtime (`nvidia-cuda-runtime-cu12`)
- cuDNN (`nvidia-cudnn-cu12`)
- cuFFT (`nvidia-cufft-cu12`)
- cuRAND (`nvidia-curand-cu12`)
- cuSOLVER (`nvidia-cusolver-cu12`)
- cuSPARSE (`nvidia-cusparse-cu12`)
- cuSPARSELt (`nvidia-cusparselt-cu12`) — carries its own NVIDIA license terms,
  distinct from the other libraries here
- NCCL (`nvidia-nccl-cu12`)
- nvJitLink (`nvidia-nvjitlink-cu12`)
- NVTX (`nvidia-nvtx-cu12`)

The lockfile is authoritative: if that pin set changes, so does this list.
Cls-Studio does not download or package these libraries from NVIDIA directly; it
inherits them from the PyTorch wheel that PyTorch's maintainers redistribute
under the permissions granted by NVIDIA.

On Linux these arrive as the separate `nvidia-*-cu12` wheels above; on Windows
the equivalent DLLs are bundled inside the `torch` wheel itself
(`torch/lib/`).

CUDA, cuDNN, cuBLAS, NCCL, NVRTC, NVTX, NVIDIA, the NVIDIA logo, and related
marks are trademarks of NVIDIA Corporation. PyTorch is a trademark of The Linux
Foundation.

---

## Transitive LGPL components (Windows installer)

The Windows installer ships pre-built binaries that include the following
LGPL-licensed component as a dynamically-linked library. It can be replaced by
the user with a compatible build, as required by the LGPL. Full license texts
are bundled under `licenses/third_party/lgpl/` (see the README in that
directory for the obligation→file mapping and the upstream-source URL).

| Component | License | Upstream source |
|---|---|---|
| FFmpeg (cv2-bundled) | LGPL-2.1+ | LGPL build shipped inside `opencv-python-headless` as `opencv_videoio_ffmpeg*_64.dll`; upstream https://github.com/FFmpeg/FFmpeg |

**Windows only.** The macOS `opencv-python-headless` wheel bundles a
*different* FFmpeg — one built `--enable-gpl` with x264/x265 — so no macOS
bundle is distributed; macOS installs from source and pip fetches that wheel
on the user's own machine. The build script refuses to package a bundle
containing those binaries.

---

## Trademarks

All trademarks, service marks, trade names, product names, and logos appearing
in this software are the property of their respective owners:

- PyTorch — trademark of The Linux Foundation.
- NVIDIA, CUDA, cuDNN — trademarks of NVIDIA Corporation.
- Microsoft, Windows — trademarks of Microsoft Corporation.
- OpenVINO, Intel — trademarks of Intel Corporation.
- DINOv2 — research model from Meta Platforms, Inc.

The use of these names is for identification and informational purposes only
and does not imply endorsement by their respective owners.

Corrections or omissions: please open an issue at
https://github.com/segmen-pixel/cls-studio/issues.
