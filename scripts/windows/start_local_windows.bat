@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 The Cls-Studio Contributors
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  cls-studio  --  Start Local Services (Windows)
REM ============================================================

REM ---- Handle --help ----------------------------------------
for %%A in (%*) do (
  if /I "%%~A"=="--help" goto :show_help
  if /I "%%~A"=="-h"     goto :show_help
  if /I "%%~A"=="/?"     goto :show_help
)

REM ---- Locate repo root ------------------------------------
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_ABS=%%~fI"
call :find_repo_root "%SCRIPT_ABS%"
if not defined REPO_ROOT call :find_repo_root "%CD%"
if not defined REPO_ROOT (
  echo [ERROR] Could not find repository root.
  echo         Ensure this file is under ^<repo^>\scripts\windows\
  echo         Or run from the repo root directory.
  exit /b 1
)

cd /d "%REPO_ROOT%"

REM ============================================================
REM  Pre-flight checks
REM ============================================================
echo.
echo ============================================================
echo  cls-studio  --  Starting Services
echo ============================================================
echo  Repo: %REPO_ROOT%
echo.

REM ---- Select venv: CLS_VENV override; else prefer the cu128 build if present ----
if not defined CLS_VENV if exist ".venv-windows-cu128\Scripts\python.exe" set "CLS_VENV=.venv-windows-cu128"
if not defined CLS_VENV set "CLS_VENV=.venv-windows"
set "PYTHON_EXE=%REPO_ROOT%\%CLS_VENV%\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Virtual environment not found.
  echo         Expected: %PYTHON_EXE%
  echo.
  echo         Run the installer first:
  echo           scripts\windows\install_windows.bat
  echo.
  exit /b 1
)

REM ---- Force venv Python for all child processes ----------------
REM   Prevents uvicorn workers from picking up a system/Store Python.
set "VIRTUAL_ENV=%REPO_ROOT%\%CLS_VENV%"
set "PATH=%REPO_ROOT%\%CLS_VENV%\Scripts;%PATH%"

REM ---- Verify venv Python works -----------------------------
"%PYTHON_EXE%" -c "import sys; sys.exit(0)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Virtual environment Python is broken.
  echo         Path: %PYTHON_EXE%
  echo.
  echo         Fix: Delete .venv-windows and rerun the installer:
  echo           rmdir /s /q .venv-windows
  echo           scripts\windows\install_windows.bat
  echo.
  exit /b 1
)

REM ---- Check port conflicts ---------------------------------
set "PORT_CONFLICT=0"
for %%P in (8791) do (
  for /f "tokens=5" %%I in ('netstat -ano 2^>nul ^| findstr /C:":%%P" ^| findstr /I /C:"LISTENING"') do (
    if "!PORT_CONFLICT!"=="0" echo [WARN] Port conflicts detected:
    echo         Port %%P is already in use ^(PID %%I^)
    set "PORT_CONFLICT=1"
  )
)
if "%PORT_CONFLICT%"=="1" (
  echo.
  echo [WARN] Existing services may be running. They will be replaced.
  echo.
)

REM ---- Set environment variables ----------------------------
REM   User data lives OUTSIDE the repo (the server hard-refuses an in-repo
REM   projects dir). Default to the user's Documents; CLS_PROJECTS_DIR
REM   still wins if the caller set it somewhere else outside the repo.
if not defined CLS_PROJECTS_DIR set "CLS_PROJECTS_DIR=%USERPROFILE%\Documents\ClsStudio\projects"
if not defined CLS_DB_PATH set "CLS_DB_PATH=%CLS_PROJECTS_DIR%\app.db"
set "CLS_MODELS_DIR=%REPO_ROOT%\models"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONDONTWRITEBYTECODE=1"

REM ---- Host binding (default: localhost only; opt-in to LAN via GUI Settings) -
REM   CLS_HOST env var always wins. If unset, ask _resolve_host.py to peek at
REM   runtime_settings.json (the GUI persists `lan_access` there).
if not defined CLS_HOST (
  REM Extra pair of double quotes around the whole command keeps cmd's for /f
  REM from stripping the inner quotes around %PYTHON_EXE% / the script path.
  for /f "delims=" %%H in ('""%PYTHON_EXE%" "%REPO_ROOT%\scripts\windows\_resolve_host.py" 2^>nul"') do set "CLS_HOST=%%H"
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

REM ---- Firewall setup when binding to the LAN (0.0.0.0) ------
REM   Windows Firewall blocks unsolicited inbound by default, and a stray
REM   "Block python.exe" rule silently drops every LAN SYN even after
REM   uvicorn binds 0.0.0.0. On first LAN startup we self-elevate once
REM   (via _setup_firewall.ps1), disable any conflicting Block rule on
REM   the venv base python, and add idempotent Allow rules for
REM   8791 (Private profile). Subsequent starts find the rules
REM   and skip the UAC prompt.
if "%CLS_HOST%"=="0.0.0.0" (
  set "FW_OK="
  for /f "delims=" %%R in ('powershell -NoProfile -Command "if (Get-NetFirewallRule -DisplayName 'cls-studio LAN api 8791' -ErrorAction SilentlyContinue) { 'ok' }" 2^>nul') do set "FW_OK=%%R"
  if not defined FW_OK (
    echo.
    echo [INFO] First-time LAN setup -- requesting admin to configure firewall...
    REM Resolve the venv's base python.exe (the binary that actually owns the socket).
    set "BASE_PY="
    for /f "tokens=2 delims==" %%K in ('findstr /B /C:"executable" "%REPO_ROOT%\%CLS_VENV%\pyvenv.cfg" 2^>nul') do (
      for /f "tokens=* delims= " %%T in ("%%K") do set "BASE_PY=%%T"
    )
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -Wait -Verb RunAs -FilePath powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','%REPO_ROOT%\scripts\windows\_setup_firewall.ps1','-BasePython','!BASE_PY!')"
    if errorlevel 1 (
      echo [WARN] Firewall configuration was cancelled or failed.
      echo        Other PCs on the LAN will be blocked until you allow
      echo        inbound TCP 8791 (Private^) manually via wf.msc.
    ) else (
      echo [INFO] Firewall configured. LAN access enabled.
    )
    echo.
  )
)

REM ---- Ensure directories exist -----------------------------
if not exist "%CLS_PROJECTS_DIR%" mkdir "%CLS_PROJECTS_DIR%"
set "LOG_DIR=%REPO_ROOT%\logs\windows"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] start_local_windows.bat REPO_ROOT=%REPO_ROOT%>>"%LOG_DIR%\start_local.log"

REM ============================================================
REM  Start services
REM ============================================================

echo [INFO] Starting cls-studio API (port 8791, host=%CLS_HOST%)
start "cls-studio API" /B cmd /c "%CLS_VENV%\Scripts\python.exe -m uvicorn apps.api.app.main:app --host %CLS_HOST% --port 8791 >> logs\windows\api.log 2>&1"

REM ---- UI: served by the API at /ui/ on the same port (8791). ----
REM   Single port, single UI. If a build is missing (or npm is present so we
REM   can refresh it), build it now; then the API static mount serves it.
REM   For hot-reload development run `npm --prefix apps\ui run dev` yourself.
where npm >nul 2>nul
if errorlevel 1 goto :ui_no_npm
echo [INFO] Building the UI (served by the API at /ui/)...
cmd /c "npm --prefix apps\ui run build >> logs\windows\ui_build.log 2>&1"
if errorlevel 1 echo [WARN] UI build failed -- see logs\windows\ui_build.log
goto :after_npm
:ui_no_npm
if exist "%REPO_ROOT%\apps\ui\dist\index.html" (
  echo [INFO] npm not found; serving the existing UI build via the API.
) else (
  echo [WARN] npm not found and no UI build exists -- the UI at /ui/ will be empty.
  echo        Install Node.js, then: npm --prefix apps\ui install ^&^& npm --prefix apps\ui run build
)
:after_npm

echo.
echo ============================================================
echo  Services started successfully
echo ============================================================
echo.
set "BROWSE_HOST=%CLS_HOST%"
if "%CLS_HOST%"=="0.0.0.0" set "BROWSE_HOST=localhost"
echo  cls-studio UI  : http://%BROWSE_HOST%:8791/ui/
echo  cls-studio API : http://%BROWSE_HOST%:8791/docs
if "%CLS_HOST%"=="0.0.0.0" echo  [LAN] other PCs: same URL with this PC IP instead of localhost
echo  Logs        : %LOG_DIR%
echo.
echo  To stop all: scripts\windows\stop_local_windows.bat
echo.

REM ---- Wait for API ready, then open browser -------------------
echo [INFO] Waiting for API to be ready...
set "READY=0"
for /L %%N in (1,1,30) do (
  if "!READY!"=="0" (
    "%PYTHON_EXE%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8791/startup-status', timeout=2)" >nul 2>nul
    if not errorlevel 1 (
      set "READY=1"
      echo [INFO] API is ready. Opening browser...
    ) else (
      ping -n 2 127.0.0.1 >nul
    )
  )
)
if "!READY!"=="0" (
  echo [WARN] API did not respond within 60s. Opening browser anyway...
)
start "" "http://%BROWSE_HOST%:8791/ui/"

exit /b 0

:show_help
echo.
echo  cls-studio -- Start Local Services
echo.
echo  Usage:
echo    start_local_windows.bat [--help]
echo.
echo  Environment variables:
echo    CLS_HOST=0.0.0.0           Bind to all interfaces (default: 127.0.0.1)
echo    CLS_VENV=.venv-windows     venv to use (default: cu128 build if present)
echo.
echo  This script starts:
echo    - cls-studio API on port 8791
echo    - UI served by the API at http://localhost:8791/ui/
echo.
echo  Prerequisites:
echo    Run scripts\windows\install_windows.bat first.
echo.
exit /b 0

:find_repo_root
set "CANDIDATE=%~f1"
:find_repo_loop
if exist "%CANDIDATE%\apps\api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%"
  goto :eof
)
if exist "%CANDIDATE%\cls-studio\apps\api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\cls-studio"
  goto :eof
)
if exist "%CANDIDATE%\seg-sutie\apps\api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\seg-sutie"
  goto :eof
)
if exist "%CANDIDATE%\windows\cls-studio\apps\api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\windows\cls-studio"
  goto :eof
)
if exist "%CANDIDATE%\windows\seg-sutie\apps\api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\windows\seg-sutie"
  goto :eof
)
for %%P in ("%CANDIDATE%\..") do set "PARENT=%%~fP"
if /I "%PARENT%"=="%CANDIDATE%" goto :eof
set "CANDIDATE=%PARENT%"
goto :find_repo_loop
