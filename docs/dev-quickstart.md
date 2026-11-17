# Developer Quickstart

## Layout

```
apps/api/        FastAPI backend
  app/routers/     projects, bank, staging, score, images, inspections,
                   captures, hardware, system, root
  app/core/        state, scoring glue, schemas, settings, security
  tests/           API tests (pytest, TestClient, temp data dir)
packages/clscore/ memory-bank core — no FastAPI dependency
  clscore/        bank.py (3-tier bank), scoring.py, compress.py,
                   feature_extractor.py, incident.py, projection.py, fsio.py
  tests/           core tests (pytest, CPU-only)
apps/ui/         React + TypeScript + Vite frontend
  src/components/  Develop (Teach/Check), Operator (Inspect), Projects,
                   Settings
  e2e/             Playwright suite (see below)
scripts/         user-facing CLIs (installers, launcher,
                   reduce_bank.py bank-shrink maintenance)
```

Design boundary: `clscore` knows nothing about HTTP or the UI; scoring
functions are pure so results are reproducible in isolation. The API layer
owns state, locking, persistence, and settings.

## Run from source

Backend (Python 3.10+):
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r apps/api/requirements.txt -r apps/api/requirements-dev.txt
uvicorn apps.api.app.main:app --port 8791
```

Frontend (Node 18+):
```bash
cd apps/ui
npm install
npm run dev        # Vite dev server, proxies /api to :8791
```

Or build once and let the API serve it: `npm run build` → open
`http://localhost:8791/ui/`.

## Tests

```bash
# Python — API + core (fast, CPU-only, temp data dir; never touches real data)
python -m pytest apps/api/tests packages/clscore/tests -q

# TypeScript
cd apps/ui && npx tsc --noEmit && npx eslint src --ext .ts,.tsx

# Everything (skips missing tools gracefully)
bash scripts/test.sh          # Windows: scripts\test.bat
```

### E2E (Playwright)

```bash
cd apps/ui
npx playwright install chromium   # once
npm run test:e2e
```

- Runs against a live server at `http://localhost:8791` serving the built
  UI — start one first.
- The suite is **additive-only**: it creates uniquely-named `zz-e2e-*`
  projects, works inside them, and deletes them afterwards. It never
  touches other projects, but prefer a scratch `CLS_PROJECTS_DIR`
  instance for CI-like runs.
- The manual release pass lives in [e2e-checklist.md](e2e-checklist.md).

## Conventions

- Apache-2.0; every source file carries an SPDX header.
- Conventional-commit style messages (`feat:` / `fix:` / `docs:` /
  `chore:`).
- Bank mutations must be atomic on disk (`clscore/fsio.py` helpers) and
  must invalidate the cached GPU tensors (`state.mark_dirty()`).
- Anything that changes scoring semantics must be reflected in the
  eval-cache fingerprint (`routers/bank.py`) — stale caches presented as
  fresh are treated as data corruption.
- UI: new views copy the interaction grammar of existing tabs (zoom / pan /
  keys — see the User Guide table); color choices must stay
  colorblind-safe (pair color with shape/label; avoid blue-purple pairs).
- pytest must stay green and CPU-only; GPU-dependent behaviour is covered
  by live smokes, not unit tests.

## CI

GitHub Actions (`.github/workflows/ci.yml`): TypeScript, ESLint, UI build,
Ruff, pytest, pip-audit, and lockfile drift checks. Reproduce locally with
`scripts/test.sh` plus `python -m piptools compile` for lockfiles.
