@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 The Cls-Studio Contributors
setlocal EnableExtensions

REM Resolve repo root from this script's location (scripts\windows\)
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

cd /d "%REPO_ROOT%"

REM ---- Select venv: CLS_VENV override; else prefer the cu128 build if present ----
if not defined CLS_VENV if exist ".venv-windows-cu128\Scripts\python.exe" set "CLS_VENV=.venv-windows-cu128"
if not defined CLS_VENV set "CLS_VENV=.venv-windows"

REM ---- Check venv exists and works --------------------------
if not exist "%CLS_VENV%\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at: %REPO_ROOT%\%CLS_VENV%
  echo.
  echo   Run the installer first:
  echo     scripts\windows\install_windows.bat
  echo.
  pause
  exit /b 1
)

"%CLS_VENV%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Virtual environment Python is broken.
  echo   Delete .venv-windows and rerun install_windows.bat
  pause
  exit /b 1
)

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

REM ---- Host binding (default: localhost only; LAN opt-in via GUI Settings) ----
if not defined CLS_HOST (
  REM Outer double-quote wrap stops cmd from stripping inner quotes.
  for /f "delims=" %%H in ('""%CLS_VENV%\Scripts\python.exe" "%REPO_ROOT%\scripts\windows\_resolve_host.py" 2^>nul"') do set "CLS_HOST=%%H"
  if not defined CLS_HOST set "CLS_HOST=127.0.0.1"
)

REM ---- Shared secret when binding to the LAN (0.0.0.0) ------
REM   A non-loopback bind without CLS_API_TOKEN is refused at startup, so
REM   without this the GUI's "Allow access from LAN" toggle would simply stop
REM   the server from starting. Mint one on first LAN start, persist it in
REM   runtime_settings.json, and show it. An explicit CLS_API_TOKEN always wins.
if "%CLS_HOST%"=="0.0.0.0" if not defined CLS_API_TOKEN (
  for /f "delims=" %%T in ('""%CLS_VENV%\Scripts\python.exe" "%REPO_ROOT%\scripts\_lan_token.py""') do set "CLS_API_TOKEN=%%T"
  if not defined CLS_API_TOKEN (
    echo [ERROR] Could not create the LAN access token. The server refuses to
    echo         serve the LAN unauthenticated, so it will not start.
    exit /b 1
  )
  echo.
  echo  LAN access token: %CLS_API_TOKEN%
  echo.
)

echo [INFO] Starting trainer API on port 8791 (host=%CLS_HOST%)...
echo        Docs: http://localhost:8791/docs
echo        UI:   http://localhost:8791/ui/
echo.
%CLS_VENV%\Scripts\python.exe -m uvicorn apps.api.app.main:app --host %CLS_HOST% --port 8791
