# Contributing to Cls-Studio

Thank you for your interest in contributing! This guide covers everything you
need to get started — from environment setup to submitting your first PR.

---

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style](#code-style)
- [Architecture Decisions](#architecture-decisions)
- [License](#license)

---

## Reporting Bugs

Open a [GitHub Issue](../../issues/new?template=bug_report.md) with:

- A clear title and description
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, Node version, GPU model)

Check existing issues first to avoid duplicates.

## Suggesting Features

Open a [Feature Request](../../issues/new?template=feature_request.md) describing
the use case, proposed solution, and any alternatives you considered.

---

## Development Setup

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ (3.11 recommended) | `from __future__ import annotations` required |
| Node.js | 18+ | For the React UI |
| npm | 9+ | |
| Git | 2.30+ | |
| NVIDIA GPU | Optional | CUDA 13.0, driver >= 580 (or CUDA 12.4 on older GPUs) for GPU training; CPU-only works fine for UI/API dev |
| Apple Silicon | Optional | MPS acceleration on macOS (M1/M2/M3/M4) |

### Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/segmen-pixel/cls-studio.git
cd cls-studio

# 2. Python environment
# Windows:
python -m venv .venv-windows
.venv-windows\Scripts\activate
# macOS:
python3 -m venv .venv-macos
source .venv-macos/bin/activate
# Linux:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies (pinned via lockfile)
# `requirements.txt` is a lockfile generated from `requirements.in` by
# `uv pip compile`. See "Dependency Policy" below before adding/upgrading
# packages.
pip install -r apps/api/requirements.txt

# (Optional) Editable install of the training core (nested package layout)
# pip install -e packages/clscore

# (Optional, contributors) Install dev/test dependencies
# pip install -r apps/api/requirements-dev.txt

# (Optional) GPU support — overrides the CPU torch wheel from the lockfile.
# This is dev-only; the lockfile still records the canonical CPU pin.
# Windows/Linux — install PyTorch with CUDA (cu124 for Maxwell/Pascal/Volta):
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
# macOS — default PyPI wheels include MPS support:
# pip install torch

# 4. Install UI dependencies and build
cd apps/ui
npm install
npm run build
cd ../..

# 5. Start the API server
python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8791

# 6. Open http://localhost:8791/ui/ in your browser
```

### Windows Shortcut

```batch
scripts\windows\install_windows.bat cuda
scripts\windows\start_local_windows.bat
```

### macOS Shortcut

```bash
bash scripts/macos/install_macos.sh
bash scripts/macos/start_local_macos.sh
```

### Verify Your Setup

```bash
# Run the test suite (venv must be activated)
bash scripts/test.sh          # Linux/macOS/WSL
scripts\test.bat              # Windows (cmd)
```

### Pre-commit Hook (recommended)

The repo ships a pre-commit hook that blocks staged changes which would
re-introduce non-commercial-licensed vendor references. Enable it once per
clone:

```bash
git config core.hooksPath scripts/dev-hooks
```

(If you work from a superproject that contains this repo as a `cls-studio/`
subdirectory, use `git config core.hooksPath cls-studio/scripts/dev-hooks`
instead — see the header of `scripts/dev-hooks/pre-commit`.)

---

## Project Structure

```
cls-studio/
├── apps/
│   ├── api/        # FastAPI backend (Python)
│   │   ├── app/
│   │   │   ├── main.py     # Entry point, lazy-loading startup
│   │   │   ├── core/       # Config, DB, utilities
│   │   │   └── routers/    # API endpoints
│   │   └── tests/          # API unit tests (pytest)
│   ├── ui/         # React frontend (TypeScript)
│   │   ├── src/
│   │   │   └── components/ # Teach (Develop), Inspect (Operator), Projects, Settings
│   │   └── e2e/            # Playwright E2E tests
├── packages/
│   └── clscore/           # Memory-bank core (installable: pip install -e packages/clscore)
│       └── clscore/
│           ├── bank.py     # 3-tier memory bank
│           ├── scoring.py  # Distance scoring
│           └── compress.py # int8 + IVF compression
├── scripts/
│   ├── test.sh / test.bat  # Unified test runners
│   ├── windows/            # Windows setup/start scripts
│   └── macos/              # macOS setup/start scripts
├── docs/                   # Documentation (EN + ja/)
├── tests/                  # cross-cutting unit tests (pytest)
├── pyproject.toml          # Python project config
└── README.md
```

### Which directory should I edit?

| I want to... | Edit |
|--------------|------|
| Fix a backend API endpoint | `apps/api/app/routers/` |
| Change how patch features are extracted | `packages/clscore/clscore/feature_extractor.py` |
| Change memory-bank storage or teach | `packages/clscore/clscore/bank.py` |
| Change scoring or bank compression | `packages/clscore/clscore/scoring.py`, `compress.py` |
| Modify the Develop (teach / evaluate) UI | `apps/ui/src/components/Develop.tsx` |
| Modify the Operator (inspection) UI | `apps/ui/src/components/Operator.tsx` |
| Add/fix an E2E test | `apps/ui/e2e/` |
| Add a Python unit test | `apps/api/tests/` or `packages/clscore/tests/` |

---

## Running Tests

### Full Test Suite

```bash
# Activate venv first!
bash scripts/test.sh
```

This runs (in order):
1. **TypeScript type check** — `tsc --noEmit`
2. **ESLint** — UI linting
3. **Ruff** — Python linting (`packages/`, `apps/api/app/`, `scripts/`)
4. **Python import verification** — ensures core modules load correctly
5. **Pytest** — unit tests (`tests/`, `apps/api/tests/`)
6. **UI build** — `vite build` (catches compile errors)
7. **E2E tests** — Playwright (requires API running on port 8791)

### Running Tests Individually

```bash
# Python unit tests only
pytest tests/ -v
pytest apps/api/tests/ -v

# TypeScript check only
cd apps/ui && npx tsc --noEmit

# E2E tests (API must be running)
cd apps/ui && npx playwright test

# Single E2E test file
npx playwright test e2e/specs/smoke.spec.ts
```

> **E2E notes:**
> - Run `npx playwright test` **without** a `--reporter` flag — the CLI flag
>   would override the reporter list in `playwright.config.ts` and silently
>   drop the skip-budget gate (`e2e/skip-budget-reporter.ts`).
> - The suite seeds its own fixture projects (`zz-e2e-seed-1/2`) on the
>   running API via `e2e/global-setup.ts`; no manual data setup is needed.

### UI Development Cycle

After editing frontend code, you **must** rebuild before testing on port 8791:

```bash
cd apps/ui
npm run build
# Then check http://localhost:8791/ui/
```

> **Note:** Vite dev server (port 5173) is not used in production.
> Always test against the built version on port 8791.

---

## Submitting Pull Requests

1. **Fork** the repository and clone your fork.
2. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/short-description
   ```
3. **Make your changes** in small, focused commits.
4. **Run the test suite** — all checks must pass.
5. **Push** your branch and open a Pull Request against `main`.
6. Fill in the PR template. Link any related issues.

### Branch Naming

| Type     | Prefix      | Example                  |
|----------|-------------|--------------------------|
| Bug fix  | `fix/`      | `fix/mask-save-race`     |
| Feature  | `feat/`     | `feat/bank-compression`  |
| Refactor | `refactor/` | `refactor/train-loop`    |
| Docs     | `docs/`     | `docs/deployment-guide`  |

### Commit Messages

Use a short imperative summary (50 chars or less), then a blank line and
optional details:

```
Fix mask save race condition on slow disks

The beforeunload handler now waits for the keepalive fetch to complete
before allowing the page to unload.
```

### What Makes a Good PR

- **Focused**: one logical change per PR
- **Tested**: include relevant test updates
- **Documented**: update docs if behavior changes
- **Small**: prefer multiple small PRs over one large one

---

## Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/). Enforced by [Ruff](https://docs.astral.sh/ruff/).
- `from __future__ import annotations` in all modules.
- `logging` instead of `print()` for server output.
- `torch.load(..., weights_only=True)` for security.
- `encoding="utf-8"` when opening files (Windows compatibility).
- GroupNorm only — never use BatchNorm (single-image inference requirement).

### TypeScript / React

- Follow existing conventions in `apps/ui/src/`.
- No unused imports or variables (ESLint enforced).
- Keep components focused; extract large sections into separate files.

### CSS

- Use CSS custom properties (`--sidebar-w`, `--accent`, etc.).
- `overflow: hidden` + `text-overflow: ellipsis` for any text that might overflow.
- `box-shadow: inset 0 0 0 2px` for selection borders (not border/outline).
- No `!important` unless overriding third-party styles.

---

## Dependency Policy

We bundle a lot of ML wheels and ship under Apache-2.0, so a careless
dependency add can quietly violate the project's license guarantee. The
flow below makes that violation mechanically detectable.

### File layout

| File | Role |
|------|------|
| `apps/api/requirements.in`     | Human-edited dependency source (loose ranges OK). |
| `apps/api/requirements.txt`    | **Lockfile** — fully pinned, auto-generated from `.in`. |
| `apps/api/requirements-dev.in` / `.txt` | Same pair for dev/test deps. |
| `apps/api/requirements-openvino.in` / `.txt` | Same pair for the optional OpenVINO export deps (`--with-openvino`). |
| `apps/api/requirements-cu130.in` / `.txt` | Same pair for the parallel Blackwell (CUDA 13.0) lockfile. |

`requirements.txt` is **never edited by hand**. CI verifies it matches
`requirements.in` (`.github/workflows/lockfile-drift.yml`).

### Adding or upgrading a Python dependency

1. **Edit `requirements.in`** (the relevant one — trainer / serving / dev).
   Keep ranges loose for utilities, pin to `==X.Y.*` for ML/server core.
2. **Confirm the license is OSS-compatible** at *all three* sources:
   - GitHub repo root `LICENSE`
   - Any sub-tree `LICENSE_*` / `LICENSE.<component>` files
   - HuggingFace model card `license:` field (if it's a model)

   Acceptable: `Apache-2.0` / `MIT` / `BSD-2-Clause` / `BSD-3-Clause` /
   `MPL-2.0` / `LGPL` (linked, not modified) / `Python-2.0`.

   **Blocked**: any non-commercial license — `NVIDIA Source Code License-NC`,
   `CC-BY-NC*`, `Research-only` — and any GPL flavour for runtime deps.
   AGPL-3.0 is also blocked because of its redistribution clause.

   `.github/workflows/license-check.yml` greps source files for known NC
   vendor strings. `.github/workflows/sbom.yml` re-checks the
   machine-readable license expressions in the SBOM.
3. **Recompile the lockfile** with [uv](https://docs.astral.sh/uv/).
   Use exactly the command (and working directory) documented in each
   `requirements*.in` header — uv embeds the command line into the lockfile,
   so any deviation shows up as drift in CI
   (`.github/workflows/lockfile-drift.yml`):

   ```bash
   # Install uv once: pip install uv  (or: pipx install uv)

   # cls-studio API (run inside apps/api)
   (cd apps/api && uv pip compile requirements.in \
     -o requirements.txt --python-version 3.11 --universal)

   # Dev deps (repo root; constrained by the trainer lockfile)
   uv pip compile apps/api/requirements-dev.in \
     -o apps/api/requirements-dev.txt \
     --python-version 3.11 --no-emit-index-url --universal \
     --constraint apps/api/requirements.txt

   # OpenVINO export deps (run inside apps/api)
   (cd apps/api && uv pip compile requirements-openvino.in \
     -o requirements-openvino.txt --python-version 3.11 --universal)
   ```

4. **Run the test suite** to confirm the new resolution actually works:

   ```bash
   bash scripts/test.sh
   ```

5. **Commit `.in` and `.txt` together**, with a license-confirmation trail
   in the commit body — one line per package added or bumped:

   ```
   deps(trainer): add foo for X feature

   LICENSE: foo 1.2.3 Apache-2.0 confirmed at https://github.com/example/foo/blob/main/LICENSE
   ```

   This trail is what makes a half-yearly re-audit (`git log --grep
   "LICENSE:"`) tractable.

### Pinning git+URL dependencies

If a dependency ever has to come from a git URL instead of PyPI, pin it to
a specific commit SHA (not a branch ref). Bumping a SHA is a dependency
upgrade — re-confirm the upstream `LICENSE` has not changed AND re-run the
scoring smoke tests before committing the new SHA. (The tree currently has
no git+URL dependencies.)

### Vulnerability response

`pip-audit` runs in CI against every lockfile (OSV vulnerability
database). If a transitive dep gets a CVE, bump it in `requirements.in`
and recompile per the steps above — don't pin a patched transitive in the
lockfile by hand, or drift detection will block the next PR.

### SBOM

Every release tag (`v*`) triggers `.github/workflows/sbom.yml`, which
attaches CycloneDX 1.6 + SPDX 2.3 SBOMs to the GitHub Release. The
workflow also re-verifies that no NC license expression slipped through
transitive deps before the SBOM goes public.

### GPU / CUDA matrix

`requirements.txt` (the default lockfile) pins `torch==2.6.0` / CUDA
12.4 wheels, with the `arch_list` covering Pascal sm_60 through Hopper
sm_90. This is the production-validated setup for Ampere (3080Ti /
3090 / A6000) and Hopper (H100) GPUs.

`requirements-cu130.txt` is a **parallel lockfile** that bumps the torch
family to `2.13.0` (installed from the CUDA 13.0 wheel index) to add
Blackwell (sm_120 / RTX 5090 / B200) support. All non-torch dependencies
are identical between the two lockfiles.

| Lockfile | torch | CUDA | arch_list covers | Status |
|---|---|---|---|---|
| `requirements.txt`        | 2.6.0  | 12.4 | sm_60 – sm_90  | Production (Ampere / Hopper) |
| `requirements-cu130.txt`  | 2.13.0 | 13.0 | sm_75 – sm_120 | Validated for Blackwell |

Newer PyTorch wheels are backwards-compatible with older supported GPUs,
so the cu130 variant also runs on 3080Ti / 3090. Choose the lockfile
that matches the hardware you target; the rest of the workflow
(test suite, drift CI, SBOM) operates identically on both.

When upgrading the torch family in either lockfile, follow the same
license-confirmation trail as for any other dependency change. Both
lockfiles record canonical PyPI versions (no `+cu130` local-version
suffix); the CUDA-specific wheels are selected at install time via the
PyTorch `--index-url` / `--extra-index-url`, so CI can install CPU
wheels from the same lockfiles.

---

## Getting Help

- **Questions**: [GitHub Discussions](../../discussions)
- **Bugs**: [GitHub Issues](../../issues)
- **Security**: See [SECURITY.md](SECURITY.md)

---

## License & Developer Certificate of Origin

By contributing, you agree that your contributions will be licensed under the
same terms as the project — [Apache License 2.0](LICENSE).

We use the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) to confirm that contributors have the right to submit their changes.
Every commit on a pull request must be signed off by adding a `Signed-off-by`
line to the commit message:

```
Fix mask save race on slow disks

Signed-off-by: Your Name <your.email@example.com>
```

The easiest way to add this is the `-s` flag:

```bash
git commit -s -m "Fix mask save race on slow disks"
```

The name and email **must** match the identity in `git config user.name` /
`user.email`; pull requests with unsigned commits will be asked to amend.

By signing off, you certify the four points listed at
https://developercertificate.org/ — in short, that you wrote the code (or have
the right to submit it) and that it is provided under the project's license.
