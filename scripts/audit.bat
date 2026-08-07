@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 The Cls-Studio Contributors
REM cls-studio — Dependency Security Audit
REM Runs npm audit (UI) and pip-audit (API)

setlocal

echo ============================================
echo  cls-studio Dependency Audit
echo ============================================
echo.

REM --- UI (npm) ---
echo [1/2] npm audit (apps/ui)
echo --------------------------------------------
pushd "%~dp0..\apps\ui"
if exist package-lock.json (
    call npm audit --audit-level=moderate 2>&1
) else (
    echo SKIP: package-lock.json not found
)
popd
echo.

REM --- API (pip) ---
echo [2/2] pip-audit (Python dependencies)
echo --------------------------------------------
where pip-audit >nul 2>&1
if %ERRORLEVEL% equ 0 (
    pip-audit 2>&1
) else (
    echo SKIP: pip-audit not installed. Install with: pip install pip-audit
)

echo.
echo ============================================
echo  Audit complete
echo ============================================
