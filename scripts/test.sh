#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
# ---------------------------------------------------------------
# Unified test runner for cls-studio
# Runs available checks and skips missing components gracefully.
# Usage: bash scripts/test.sh
# ---------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PASS=0
FAIL=0
SKIP=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  [SKIP] $1"; SKIP=$((SKIP + 1)); }

# Find python executable (needed early for Ruff check)
PYTHON=""
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif [ -x "$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe" ]; then
    PYTHON="$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe"
elif [ -x "$REPO_ROOT/.venv-windows/Scripts/python.exe" ]; then
    PYTHON="$REPO_ROOT/.venv-windows/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
fi

echo "========================================"
echo " cls-studio Test Runner"
echo "========================================"
echo ""

# ------------------------------------------------------------------
# 1. TypeScript type check
# ------------------------------------------------------------------
echo "--- TypeScript ---"
UI_DIR="$REPO_ROOT/apps/ui"

if [ -f "$UI_DIR/tsconfig.json" ] && [ -d "$UI_DIR/node_modules" ]; then
    if (cd "$UI_DIR" && npx tsc --noEmit 2>&1); then
        pass "TypeScript type check"
    else
        fail "TypeScript type check"
    fi
elif [ -f "$UI_DIR/tsconfig.json" ]; then
    skip "TypeScript type check (run 'npm install' in apps/ui first)"
else
    skip "TypeScript type check (no tsconfig.json found)"
fi
echo ""

# ------------------------------------------------------------------
# 2. ESLint (only if config exists in the project, not node_modules)
# ------------------------------------------------------------------
echo "--- ESLint ---"
ESLINT_CONFIG=""
for f in "$UI_DIR/.eslintrc" "$UI_DIR/.eslintrc.js" "$UI_DIR/.eslintrc.json" \
         "$UI_DIR/.eslintrc.yml" "$UI_DIR/.eslintrc.yaml" \
         "$UI_DIR/eslint.config.js" "$UI_DIR/eslint.config.mjs" \
         "$UI_DIR/eslint.config.cjs"; do
    if [ -f "$f" ]; then
        ESLINT_CONFIG="$f"
        break
    fi
done

if [ -n "$ESLINT_CONFIG" ] && [ -d "$UI_DIR/node_modules" ]; then
    # Delegate to the `lint` script in apps/ui/package.json so this runner and
    # the ESLint job in .github/workflows/ci.yml share one threshold.
    if (cd "$UI_DIR" && npm run lint 2>&1); then
        pass "ESLint"
    else
        fail "ESLint"
    fi
else
    skip "ESLint (no eslint config in apps/ui)"
fi
echo ""

# ------------------------------------------------------------------
# 2b. Ruff (Python linter)
# ------------------------------------------------------------------
echo "--- Ruff ---"
if [ -n "$PYTHON" ] && "$PYTHON" -m ruff version &>/dev/null; then
    # Same target as the Ruff job in .github/workflows/ci.yml — settings come
    # from [tool.ruff] in pyproject.toml.
    if (cd "$REPO_ROOT" && "$PYTHON" -m ruff check . 2>&1); then
        pass "Ruff lint"
    else
        fail "Ruff lint"
    fi
else
    skip "Ruff (not installed — pip install ruff)"
fi
echo ""

# ------------------------------------------------------------------
# 3. Python import check — verify key modules can be imported
# ------------------------------------------------------------------
echo "--- Python imports ---"

if [ -n "$PYTHON" ]; then
    IMPORT_OK=true

    # Check FastAPI app module
    if "$PYTHON" -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'packages'))
os.chdir('$REPO_ROOT')
from apps.api.app.core.config import APP_VERSION
print(f'  config.APP_VERSION = {APP_VERSION}')
" 2>&1; then
        pass "Python import: config"
    else
        fail "Python import: config"
        IMPORT_OK=false
    fi

    # Check clscore package. The package root is packages/clscore (the
    # importable package itself is packages/clscore/clscore), so that is
    # what goes on sys.path when clscore is not pip-installed.
    if "$PYTHON" -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'packages', 'clscore'))
from clscore.bank import Bank
from clscore.scoring import score_stored_features
print('  clscore.bank / clscore.scoring import OK')
" 2>&1; then
        pass "Python import: clscore"
    else
        fail "Python import: clscore"
        IMPORT_OK=false
    fi
else
    skip "Python import checks (no python found)"
fi
echo ""

# ------------------------------------------------------------------
# 3b. Pytest unit tests
# ------------------------------------------------------------------
echo "--- Pytest ---"

if [ -n "$PYTHON" ]; then
    # Check if pytest is available in the python environment
    if "$PYTHON" -m pytest --version &>/dev/null; then
        TEST_DIRS=""
        # Collect test directories that exist
        if [ -d "$REPO_ROOT/apps/api/tests" ]; then
            TEST_DIRS="$TEST_DIRS $REPO_ROOT/apps/api/tests"
        fi
        if [ -d "$REPO_ROOT/tests" ]; then
            TEST_DIRS="$TEST_DIRS $REPO_ROOT/tests"
        fi
        if [ -d "$REPO_ROOT/packages/clscore/tests" ]; then
            TEST_DIRS="$TEST_DIRS $REPO_ROOT/packages/clscore/tests"
        fi

        if [ -n "$TEST_DIRS" ]; then
            echo "  Running: $PYTHON -m pytest -x --tb=short $TEST_DIRS"
            if (cd "$REPO_ROOT" && "$PYTHON" -m pytest -x --tb=short $TEST_DIRS 2>&1); then
                pass "Pytest unit tests"
            else
                fail "Pytest unit tests"
            fi
        else
            skip "Pytest (no test directories found)"
        fi
    else
        echo "  [WARN] pytest is not installed. Install test deps with:"
        echo "         pip install -r apps/api/requirements-dev.txt"
        skip "Pytest (pytest not installed)"
    fi
else
    skip "Pytest (no python found)"
fi
echo ""

# ------------------------------------------------------------------
# 4. UI build check (verify production build succeeds)
# ------------------------------------------------------------------
echo "--- UI build check ---"
if [ -f "$UI_DIR/package.json" ] && [ -d "$UI_DIR/node_modules" ]; then
    if (cd "$UI_DIR" && npx vite build 2>&1); then
        pass "UI production build (vite build)"
    else
        fail "UI production build (vite build)"
    fi
else
    skip "UI build check (run 'npm install' in apps/ui first)"
fi
echo ""

# ------------------------------------------------------------------
# 5. E2E tests (only if API is running)
# ------------------------------------------------------------------
echo "--- E2E tests ---"
# Health endpoint lives under the versioned API prefix (the hardware router
# is mounted at /api/v1 — see apps/api/app/router_registry.py).
API_URL="http://localhost:8791/api/v1/health"

if curl -sf "$API_URL" >/dev/null 2>&1; then
    if [ -d "$UI_DIR/e2e" ] && [ -d "$UI_DIR/node_modules" ]; then
        echo "  API is running. Launching Playwright E2E tests..."
        if (cd "$UI_DIR" && npx playwright test 2>&1); then
            pass "E2E tests (Playwright)"
        else
            fail "E2E tests (Playwright)"
        fi
    else
        skip "E2E tests (e2e directory or node_modules missing)"
    fi
else
    skip "E2E tests (API not running on localhost:8791)"
fi
echo ""

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
