@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 The Cls-Studio Contributors
setlocal EnableExtensions EnableDelayedExpansion
REM ---------------------------------------------------------------
REM Unified test runner for cls-studio (Windows)
REM Runs available checks and skips missing components gracefully.
REM Usage: scripts\test.bat
REM ---------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

set PASS=0
set FAIL=0
set SKIP=0

echo ========================================
echo  cls-studio Test Runner
echo ========================================
echo.

REM ------------------------------------------------------------------
REM 1. TypeScript type check
REM ------------------------------------------------------------------
echo --- TypeScript ---
set "UI_DIR=%REPO_ROOT%\apps\ui"

if not exist "%UI_DIR%\tsconfig.json" (
    echo   [SKIP] TypeScript type check ^(no tsconfig.json found^)
    set /a SKIP+=1
    goto :eslint
)
if not exist "%UI_DIR%\node_modules" (
    echo   [SKIP] TypeScript type check ^(run 'npm install' in apps\ui first^)
    set /a SKIP+=1
    goto :eslint
)

pushd "%UI_DIR%"
call npx tsc --noEmit >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] TypeScript type check
    set /a PASS+=1
) else (
    echo   [FAIL] TypeScript type check
    call npx tsc --noEmit 2>&1
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 2. ESLint (only if config exists in the project)
REM ------------------------------------------------------------------
:eslint
echo --- ESLint ---
set "HAS_ESLINT="
for %%F in (.eslintrc .eslintrc.js .eslintrc.json .eslintrc.yml .eslintrc.yaml eslint.config.js eslint.config.mjs eslint.config.cjs) do (
    if exist "%UI_DIR%\%%F" set "HAS_ESLINT=1"
)

if not defined HAS_ESLINT (
    echo   [SKIP] ESLint ^(no eslint config in apps\ui^)
    set /a SKIP+=1
    goto :pyimport
)
if not exist "%UI_DIR%\node_modules" (
    echo   [SKIP] ESLint ^(node_modules missing^)
    set /a SKIP+=1
    goto :pyimport
)

REM Delegate to the `lint` script in apps\ui\package.json so this runner and
REM the ESLint job in .github\workflows\ci.yml share one threshold.
pushd "%UI_DIR%"
call npm run lint >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] ESLint
    set /a PASS+=1
) else (
    echo   [FAIL] ESLint
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 2b. Ruff (Python linter)
REM ------------------------------------------------------------------
:ruff
echo --- Ruff ---

set "PYTHON="
if exist "%REPO_ROOT%\.venv-windows-cu130\Scripts\python.exe" (
  set "PYTHON=%REPO_ROOT%\.venv-windows-cu130\Scripts\python.exe"
) else if exist "%REPO_ROOT%\.venv-windows-cu128\Scripts\python.exe" (
    set "PYTHON=%REPO_ROOT%\.venv-windows-cu128\Scripts\python.exe"
) else if exist "%REPO_ROOT%\.venv-windows\Scripts\python.exe" (
    set "PYTHON=%REPO_ROOT%\.venv-windows\Scripts\python.exe"
) else if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON=python"
    )
)

if not defined PYTHON (
    echo   [SKIP] Ruff ^(no python found^)
    set /a SKIP+=1
    goto :pyimport
)

"%PYTHON%" -m ruff version >nul 2>nul
if %errorlevel% neq 0 (
    echo   [SKIP] Ruff ^(not installed — pip install ruff^)
    set /a SKIP+=1
    goto :pyimport
)

REM Same target as the Ruff job in .github\workflows\ci.yml — settings come
REM from [tool.ruff] in pyproject.toml.
pushd "%REPO_ROOT%"
"%PYTHON%" -m ruff check . >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Ruff lint
    set /a PASS+=1
) else (
    echo   [FAIL] Ruff lint
    "%PYTHON%" -m ruff check . 2>&1
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 3. Python import check
REM ------------------------------------------------------------------
:pyimport
echo --- Python imports ---

if not defined PYTHON (
    echo   [SKIP] Python import checks ^(no python found^)
    set /a SKIP+=1
    goto :uibuild
)

pushd "%REPO_ROOT%"

"%PYTHON%" -c "import sys,os;sys.path.insert(0,os.path.join(r'%REPO_ROOT%','packages'));from apps.api.app.core.config import APP_VERSION;print(f'  config.APP_VERSION = {APP_VERSION}')" 2>nul
if %errorlevel% equ 0 (
    echo   [PASS] Python import: config
    set /a PASS+=1
) else (
    echo   [FAIL] Python import: config
    set /a FAIL+=1
)

REM The importable package is packages\clscore\clscore, so packages\clscore
REM is what goes on sys.path when clscore is not pip-installed.
"%PYTHON%" -c "import sys,os;sys.path.insert(0,os.path.join(r'%REPO_ROOT%','packages','clscore'));from clscore.bank import Bank;from clscore.scoring import score_stored_features;print('  clscore import OK')" 2>nul
if %errorlevel% equ 0 (
    echo   [PASS] Python import: clscore
    set /a PASS+=1
) else (
    echo   [FAIL] Python import: clscore
    set /a FAIL+=1
)

popd
echo.

REM ------------------------------------------------------------------
REM 3b. Pytest unit tests
REM ------------------------------------------------------------------
:pytest
echo --- Pytest ---

if not defined PYTHON (
    echo   [SKIP] Pytest ^(no python found^)
    set /a SKIP+=1
    goto :uibuild
)

"%PYTHON%" -m pytest --version >nul 2>nul
if %errorlevel% neq 0 (
    echo   [WARN] pytest is not installed. Install test deps with:
    echo          pip install -r apps\api\requirements-dev.txt
    echo   [SKIP] Pytest ^(pytest not installed^)
    set /a SKIP+=1
    goto :uibuild
)

set "TEST_DIRS="
if exist "%REPO_ROOT%\apps\api\tests" set "TEST_DIRS=%TEST_DIRS% %REPO_ROOT%\apps\api\tests"
if exist "%REPO_ROOT%\tests" set "TEST_DIRS=%TEST_DIRS% %REPO_ROOT%\tests"
if exist "%REPO_ROOT%\packages\clscore\tests" set "TEST_DIRS=%TEST_DIRS% %REPO_ROOT%\packages\clscore\tests"

if not defined TEST_DIRS (
    echo   [SKIP] Pytest ^(no test directories found^)
    set /a SKIP+=1
    goto :uibuild
)

pushd "%REPO_ROOT%"
echo   Running: "%PYTHON%" -m pytest -x --tb=short%TEST_DIRS%
"%PYTHON%" -m pytest -x --tb=short%TEST_DIRS%
if %errorlevel% equ 0 (
    echo   [PASS] Pytest unit tests
    set /a PASS+=1
) else (
    echo   [FAIL] Pytest unit tests
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 4. UI build check
REM ------------------------------------------------------------------
:uibuild
echo --- UI build check ---
if not exist "%UI_DIR%\package.json" (
    echo   [SKIP] UI build check ^(no package.json^)
    set /a SKIP+=1
    goto :e2e
)
if not exist "%UI_DIR%\node_modules" (
    echo   [SKIP] UI build check ^(run 'npm install' in apps\ui first^)
    set /a SKIP+=1
    goto :e2e
)

pushd "%UI_DIR%"
call npx vite build >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] UI production build ^(vite build^)
    set /a PASS+=1
) else (
    echo   [FAIL] UI production build ^(vite build^)
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 5. E2E tests (only if API is running)
REM ------------------------------------------------------------------
:e2e
echo --- E2E tests ---

curl -sf http://localhost:8791/api/v1/health >nul 2>nul
if %errorlevel% neq 0 (
    echo   [SKIP] E2E tests ^(API not running on localhost:8791^)
    set /a SKIP+=1
    goto :summary
)

if not exist "%UI_DIR%\e2e" (
    echo   [SKIP] E2E tests ^(e2e directory missing^)
    set /a SKIP+=1
    goto :summary
)

pushd "%UI_DIR%"
echo   API is running. Launching Playwright E2E tests...
call npx playwright test 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] E2E tests ^(Playwright^)
    set /a PASS+=1
) else (
    echo   [FAIL] E2E tests ^(Playwright^)
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM Summary
REM ------------------------------------------------------------------
:summary
echo.
echo ========================================
echo  Results: %PASS% passed, %FAIL% failed, %SKIP% skipped
echo ========================================

if %FAIL% gtr 0 exit /b 1
exit /b 0
