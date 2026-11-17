@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 The Cls-Studio Contributors
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  cls-studio  --  Windows Install Script
REM  Usage:  install_windows.bat [cpu|cuda|cuda124]
REM                              [--with-openvino]
REM                              [--skip-ui] [--help]
REM ============================================================

REM ---- Handle --help early (before repo root detection) ------
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
  echo.
  echo  [ERROR] Could not find repository root.
  echo          This script expects to be located at:
  echo            ^<repo^>\scripts\windows\install_windows.bat
  echo          Or run it from the repository root directory.
  echo.
  goto :fail
)
cd /d "%REPO_ROOT%"

REM ---- Log setup --------------------------------------------
set "LOG_DIR=%REPO_ROOT%\logs\windows"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\install_windows.log"
echo ============================================================ > "%LOG_FILE%"
echo [%date% %time%] install_windows.bat started >> "%LOG_FILE%"
echo Repo root: %REPO_ROOT% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM ---- Parse arguments --------------------------------------
REM Auto-detect NVIDIA GPU: default to cuda if nvidia-smi is available
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  set "TORCH_FLAVOR=cpu"
) else (
  set "TORCH_FLAVOR=cuda"
)
REM CUDA build: cu130 (default; Turing/RTX 20xx and newer, incl. Blackwell;
REM             CUDA 13.x needs an NVIDIA driver >= 580)
REM             or cu124 (older GPUs: Maxwell/Pascal/Volta) via the "cuda124" arg.
set "TORCH_CUDA_INDEX=cu130"
set "WITH_OPENVINO=0"
set "SKIP_UI=0"


:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="cpu" (
  set "TORCH_FLAVOR=cpu"
) else if /I "%~1"=="cuda" (
  set "TORCH_FLAVOR=cuda"
  set "TORCH_CUDA_INDEX=cu130"
) else if /I "%~1"=="cuda124" (
  set "TORCH_FLAVOR=cuda"
  set "TORCH_CUDA_INDEX=cu124"
) else if /I "%~1"=="--with-openvino" (
  set "WITH_OPENVINO=1"
) else if /I "%~1"=="--skip-ui" (
  set "SKIP_UI=1"
) else (
  echo [WARN] Unknown option: %~1
  echo        Run with --help for usage information.
)
shift
goto parse_args

:args_done

REM ---- Validate repo root -----------------------------------
if not exist "apps\api\app\main.py" (
  echo [ERROR] Repository structure validation failed.
  echo         Expected file not found: apps\api\app\main.py
  echo         Detected root: %REPO_ROOT%
  goto :fail
)

REM ============================================================
REM  STEP 1: Prerequisites check
REM ============================================================
echo.
echo ============================================================
echo  cls-studio Windows Installer
echo ============================================================
echo  Repo root : %REPO_ROOT%
echo  Mode      : %TORCH_FLAVOR%
echo  Log file  : %LOG_FILE%
echo ============================================================
echo.
echo [STEP 1/6] Checking prerequisites...
echo.

set "PREREQ_OK=1"

REM ---- Python detection -------------------------------------
set "PY_BOOTSTRAP="
set "PY_VERSION="
call :resolve_python
if errorlevel 1 goto :python_not_found
goto :python_found

:python_not_found
set "PREREQ_OK=0"
echo   Python ...... NOT FOUND
echo.
echo   [ERROR] Python 3.10 or later is required but was not found.
echo.
echo   How to install Python:
echo     1. Download from https://www.python.org/downloads/windows/
echo        (Recommended: Python 3.11.x)
echo     2. IMPORTANT: Check "Add Python to PATH" during installation
echo     3. After installing, CLOSE this terminal and open a new one
echo     4. Verify: python --version
echo.
echo   Alternatively, if you have winget:
echo     winget install Python.Python.3.11
echo.
goto :fail

:python_found
echo   Python ...... OK  (!PY_VERSION!)

REM ---- Check Python version is 3.10+ -----------------------
call :check_python_version
if errorlevel 1 (
  echo.
  echo   [ERROR] Python %PY_VERSION% is outside the supported range.
  echo           cls-studio needs Python 3.10, 3.11, 3.12 or 3.13.
  echo.
  echo   A newer Python is not "better" here: the pinned PyTorch build has no
  echo   wheel for it, so the install would fail partway through instead.
  echo.
  echo   Install a supported version from:
  echo     https://www.python.org/downloads/windows/
  echo   or with winget:
  echo     winget install Python.Python.3.13
  echo.
  echo   Both can coexist with the Python you already have -- the installer
  echo   picks the supported one via the "py" launcher.
  echo.
  goto :fail
)

REM ---- Node/npm detection -----------------------------------
set "HAS_NPM=1"
where npm >nul 2>nul
if errorlevel 1 goto :npm_not_found
goto :npm_found

:npm_not_found
if "%SKIP_UI%"=="1" (
  set "HAS_NPM=0"
  echo   npm ......... SKIPPED ^(--skip-ui^)
  goto :npm_done
)
echo   npm ......... NOT FOUND ^(attempting auto-install...^)
call :install_nodejs
where npm >nul 2>nul
if errorlevel 1 goto :npm_auto_failed
for /f "tokens=*" %%V in ('npm --version 2^>nul') do echo   npm ......... OK  ^(v%%V - just installed^)
goto :npm_done

:npm_auto_failed
set "HAS_NPM=0"
echo   npm ......... NOT FOUND ^(auto-install failed^)
echo.
echo   [WARN] Node.js/npm is needed for the cls-studio UI.
echo          The API will work without it, but the UI will not be built.
echo.
echo   How to install Node.js:
echo     1. Download Node.js 22 LTS from https://nodejs.org/
echo     2. Run the installer (includes npm)
echo     3. Close and reopen this terminal
echo     4. Verify: npm --version
echo.
goto :npm_done

:npm_found
for /f "tokens=*" %%V in ('npm --version 2^>nul') do echo   npm ......... OK  ^(v%%V^)

:npm_done


REM ---- curl detection (needed for checkpoint downloads) -----
set "HAS_CURL=1"
where curl >nul 2>nul
if errorlevel 1 goto :curl_not_found
echo   curl ........ OK
goto :curl_done

:curl_not_found
set "HAS_CURL=0"
echo   curl ........ NOT FOUND
echo.
echo   [WARN] curl is required to download model checkpoints.
echo          curl is included in Windows 10 1803+ by default.
echo          If missing, install from: https://curl.se/windows/
echo.

:curl_done

REM ---- CUDA check (only when cuda mode) --------------------
if /I not "%TORCH_FLAVOR%"=="cuda" goto :cuda_skip
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :cuda_not_found
REM Quote the --format value: inside for /f the command string is re-parsed by
REM cmd, which treats a bare comma as an argument separator, so nvidia-smi was
REM handed 'noheader' as its own option. It rejects it and prints the complaint
REM on STDOUT -- which this loop then captured and echoed as the GPU name, so
REM the line read 'CUDA ........ OK (ERROR: Option noheader is not recognized)'.
REM Detection still 'passed', because it only ever checked that the loop ran.
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name "--format=csv,noheader" 2^>nul') do echo   CUDA ........ OK  ^(%%G^)
goto :cuda_done

:cuda_not_found
echo   CUDA ........ NOT FOUND (nvidia-smi not in PATH)
echo.
echo   [WARN] You specified 'cuda' but nvidia-smi was not found.
echo          This usually means:
echo            - NVIDIA drivers are not installed
echo            - Or NVIDIA tools are not in PATH
echo          PyTorch CUDA wheels will be installed, but GPU
echo          acceleration may not work at runtime.
echo.
echo          If you don't have an NVIDIA GPU, use: install_windows.bat cpu
echo.
goto :cuda_done

:cuda_skip
echo   CUDA ........ SKIPPED (cpu mode; use 'cuda' arg for GPU)

:cuda_done

echo.

REM ============================================================
REM  STEP 2: Virtual environment
REM ============================================================
echo [STEP 2/6] Setting up Python virtual environment...

set "VENV_DIR=%REPO_ROOT%\.venv-windows"

REM ---- Handle existing venv ---------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" goto :venv_check_partial
REM Verify the existing venv is functional
"%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>nul
if errorlevel 1 (
  echo   [WARN] Existing venv appears broken. Recreating...
  echo   [WARN] Removing broken venv: %VENV_DIR% >> "%LOG_FILE%"
  rmdir /s /q "%VENV_DIR%" 2>nul
  goto :create_venv
)
echo   [INFO] Using existing venv: %VENV_DIR%
for /f "tokens=*" %%V in ('"%VENV_DIR%\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2^>nul') do (
  echo   [INFO] Venv Python version: %%V
)
goto :venv_ready

:venv_check_partial
if exist "%VENV_DIR%" (
  echo   [WARN] Incomplete venv found ^(no python.exe^). Recreating...
  rmdir /s /q "%VENV_DIR%" 2>nul
)

:create_venv
echo   [INFO] Creating virtualenv: %VENV_DIR%
%PY_BOOTSTRAP% -m venv "%VENV_DIR%"
REM Check the artefact, not the exit code -- see :try_py_version for why a
REM negative exit code from py.exe walks straight past "if errorlevel 1".
if not exist "%VENV_DIR%\Scripts\python.exe" goto :venv_create_failed
echo   [INFO] Virtualenv created successfully.
goto :venv_ready

:venv_create_failed
echo.
echo   [ERROR] Failed to create virtual environment.
echo.
echo   Common causes:
echo     - Python was installed without the 'venv' module
echo       Fix: Reinstall Python and ensure "pip" and "tcl/tk" are checked
echo     - Antivirus blocking file creation
echo       Fix: Add %VENV_DIR% to your antivirus exclusions
echo     - Path too long (Windows 260 char limit)
echo       Fix: Move the repo to a shorter path (e.g., C:\cls-studio)
echo.
goto :fail

:venv_ready
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] venv python not found: %PYTHON_EXE%
  goto :fail
)

REM ============================================================
REM  STEP 3: Python dependencies
REM ============================================================
echo.
echo [STEP 3/6] Installing Python dependencies...
echo   (This may take several minutes on first install)
echo.

echo   [INFO] Upgrading pip/setuptools/wheel...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [ERROR] Failed to upgrade pip/setuptools/wheel.
  echo          Check your internet connection and try again.
  echo          See log: %LOG_FILE%
  goto :fail
)

echo   [INFO] Installing trainer API dependencies (requirements.txt)...
"%PYTHON_EXE%" -m pip install -r apps\api\requirements.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :pip_trainer_failed
goto :pip_trainer_ok

:pip_trainer_failed
echo.
echo   [ERROR] Failed to install trainer API dependencies.
echo.
echo   Common causes:
echo     - No internet connection
echo     - Firewall/proxy blocking pip
echo     - Visual C++ Build Tools missing (needed by some packages)
echo       Fix: Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo     - Incompatible Python version
echo.
echo   Details in log: %LOG_FILE%
goto :fail

:pip_trainer_ok

REM ============================================================
REM  STEP 4: PyTorch (CPU or CUDA)
REM ============================================================
echo.
echo [STEP 4/6] Configuring PyTorch (%TORCH_FLAVOR%)...

if /I not "%TORCH_FLAVOR%"=="cuda" goto :torch_cpu
echo   [INFO] Installing CUDA-enabled PyTorch wheels (%TORCH_CUDA_INDEX%)...
echo          This download is ~2.5 GB and may take a while.
REM Pinned, and torchvision deliberately absent: it was dropped from the
REM requirements in 5d9b339 because its own torch dependency silently
REM overrode the pin, and nothing in cls-studio imports it. Installing it
REM here would reintroduce exactly that bug, outside every lockfile and
REM outside the CI licence gate.
REM TORCH_NODEPS: on the cu130 path --force-reinstall must not re-resolve
REM torch's dependencies. Left to itself it pulls them from the PyTorch
REM index and overwrites what requirements.txt pinned two steps earlier.
REM Measured 2026-08-18 on a clean install: setuptools went 83.0.0 -> 78.1.0
REM (PYSEC-2026-3447 is fixed exactly at 83.0.0, so the finished install
REM shipped a flagged setuptools), and filelock, fsspec and typing-extensions
REM all drifted off their pins. cu130 installs the same torch VERSION that
REM requirements.txt pins -- only the wheel build differs -- so every runtime
REM dependency is already present and correct from STEP 3.
REM The cu124 path keeps resolving deps: it installs an older torch whose
REM requirements genuinely differ from the ones compiled for 2.13.0.
REM Single-line ifs on purpose -- see :try_py_version.
set "TORCH_PIN=torch==2.6.0"
set "TORCH_NODEPS="
if /I "%TORCH_CUDA_INDEX%"=="cu130" set "TORCH_PIN=torch==2.13.0"
if /I "%TORCH_CUDA_INDEX%"=="cu130" set "TORCH_NODEPS=--no-deps"
"%PYTHON_EXE%" -m pip install --force-reinstall %TORCH_NODEPS% --index-url https://download.pytorch.org/whl/%TORCH_CUDA_INDEX% %TORCH_PIN% >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :torch_cuda_failed
echo   [INFO] Verifying CUDA availability in PyTorch...
"%PYTHON_EXE%" -c "import torch; avail=torch.cuda.is_available(); print(f'  CUDA available: {avail}')"
goto :torch_done

:torch_cuda_failed
echo.
echo   [ERROR] CUDA PyTorch wheel installation failed.
echo.
echo   Possible fixes:
echo     - Check internet connection (large download ~2.5 GB)
echo     - Try CPU mode instead: install_windows.bat cpu
echo     - Check disk space (needs ~5 GB free)
echo.
echo   Details in log: %LOG_FILE%
goto :fail

:torch_cpu
echo   [INFO] Using CPU PyTorch (installed from requirements.txt).
echo          To enable GPU, rerun with: install_windows.bat cuda

:torch_done

REM ============================================================
REM  STEP 5: OpenVINO IR export (optional, Intel edge deployment)
REM ============================================================
echo.
if not "%WITH_OPENVINO%"=="1" goto :openvino_skip
echo [STEP 5/6] Installing OpenVINO + NNCF (Intel edge export)...
echo            ~300 MB download; required only for IR-format export.
"%PYTHON_EXE%" -m pip install -r apps\api\requirements-openvino.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] OpenVINO install failed. IR export will return HTTP 501.
  echo   [WARN] OpenVINO install failed >> "%LOG_FILE%"
) else (
  echo   [INFO] OpenVINO + NNCF installed successfully.
)
goto :openvino_done

:openvino_skip
echo [STEP 5/6] OpenVINO... SKIPPED (use --with-openvino to install)

:openvino_done


REM ============================================================
REM  STEP 6: cls-studio UI (Node.js / React)
REM ============================================================
echo.
if "%SKIP_UI%"=="1" (
  echo [STEP 6/6] cls-studio UI... SKIPPED ^(--skip-ui^)
  goto :after_ui
)
if not "%HAS_NPM%"=="1" (
  echo [STEP 6/6] cls-studio UI... SKIPPED ^(npm not available^)
  echo          Install Node.js 22 LTS from https://nodejs.org/ and rerun.
  goto :after_ui
)
echo [STEP 6/6] Building cls-studio UI...

echo   [INFO] Installing npm dependencies...
cd /d "%REPO_ROOT%\apps\ui"
call npm ci >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] 'npm ci' failed. Trying 'npm install' instead...
  call npm install >> "%LOG_FILE%" 2>&1
  if errorlevel 1 goto :npm_install_failed
)
goto :npm_install_ok

:npm_install_failed
echo.
echo   [ERROR] npm install failed.
echo.
echo   Common causes and fixes:
echo     - node-gyp errors: Install Visual C++ Build Tools
echo       https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo     - EACCES / permission errors: Run terminal as Administrator
echo     - Network errors: Check proxy settings
echo       npm config set proxy http://your-proxy:port
echo     - Corrupted cache: npm cache clean --force
echo     - node_modules conflict: delete apps\ui\node_modules
echo       and rerun this script
echo.
echo   Details in log: %LOG_FILE%
echo.
cd /d "%REPO_ROOT%"
REM Don't fail the whole install for UI issues
goto :after_ui

:npm_install_ok
echo   [INFO] Building UI (npm run build)...
call npm run build >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] UI build failed. The UI can still run via Vite dev server.
  echo   [WARN] To build manually: cd apps\ui ^&^& npm run build
  echo   [WARN] UI build failed >> "%LOG_FILE%"
) else (
  echo   [INFO] UI built successfully.
)
cd /d "%REPO_ROOT%"

:after_ui

cd /d "%REPO_ROOT%"
if not exist "logs\windows" mkdir "logs\windows"

REM ============================================================
REM  Summary
REM ============================================================
echo.
echo ============================================================
echo  Installation Complete
echo ============================================================
echo.
echo  Next steps:
echo    Start:  scripts\windows\start_local_windows.bat
echo    Stop:   scripts\windows\stop_local_windows.bat
echo    Status: scripts\windows\status_windows.bat
echo.
echo  Endpoints (after starting):
echo    cls-studio API : http://localhost:8791/docs
echo    cls-studio UI  : http://localhost:8791/ui/
echo.
echo  Full log: %LOG_FILE%
echo ============================================================
echo.

echo [%date% %time%] install_windows.bat completed successfully >> "%LOG_FILE%"
exit /b 0

REM ============================================================
REM  Subroutines
REM ============================================================

:show_help
echo.
echo  cls-studio Windows Installer
echo.
echo  Usage:
echo    install_windows.bat [OPTIONS]
echo.
echo  Options:
echo    cpu                   Install CPU-only PyTorch
echo    cuda                  CUDA PyTorch, cu130 build (default for NVIDIA GPUs^).
echo                          Turing/RTX 20xx and newer, incl. Blackwell (RTX 50xx, RTX PRO 6000^).
echo    cuda124               CUDA PyTorch, cu124 build for older GPUs
echo                          (Maxwell/Pascal/Volta: GTX 10xx, Tesla V100^).
echo                          (Default: auto-detect — cuda/cu130 if NVIDIA GPU found^)
echo    --with-openvino       Also install OpenVINO + NNCF (~300 MB^)
echo                          Enables Intel edge export (.xml/.bin, FP32/FP16/INT8^)
echo    --skip-ui             Skip Node.js/npm UI build
echo    --help, -h            Show this help message
echo.
echo  Examples:
echo    install_windows.bat                     CPU mode, full install
echo    install_windows.bat cuda                GPU mode with CUDA
echo    install_windows.bat --skip-ui           Skip UI build (API only)
echo.
echo  Prerequisites:
echo    Required:  Python 3.10+ (with pip and venv)
echo    Optional:  Node.js 22 LTS (for cls-studio UI)
echo    Optional:  NVIDIA GPU + drivers (for CUDA mode)
echo.
echo  If you encounter issues, check the log at:
echo    logs\windows\install_windows.log
echo.
exit /b 0

:fail
echo.
echo ============================================================
echo  [FAILED] Setup did not complete successfully.
echo ============================================================
if defined LOG_FILE (
  echo  Check the log for details:
  echo    %LOG_FILE%
)
echo.
echo  Common troubleshooting steps:
echo    1. Ensure Python 3.10+ is installed and in PATH
echo    2. Ensure you have internet access
echo    3. Try running as Administrator if permission errors occur
echo    4. If the venv is corrupted, delete .venv-windows and retry
echo    5. Run with --help for usage information
echo.
pause
exit /b 1

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

:install_nodejs
where winget >nul 2>nul
if errorlevel 1 (
  echo   [INFO] winget is unavailable. Cannot auto-install Node.js.
  goto :eof
)
echo   [INFO] Attempting to install Node.js 22 LTS via winget...
winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [INFO] Automatic Node.js install did not succeed.
  goto :eof
)
REM Refresh PATH to pick up newly installed Node.js
set "PROG_NODE=%ProgramFiles%\nodejs"
if exist "%PROG_NODE%\npm.cmd" (
  set "PATH=%PROG_NODE%;%PATH%"
  echo   [INFO] Node.js installed: %PROG_NODE%
)
goto :eof

:resolve_python
REM Try Python Launcher first (supports version selection)
REM NOTE: We use explicit sequential calls instead of a for-loop
REM       because batch errorlevel is sticky inside parenthesized blocks.
set "PY_BOOTSTRAP="
set "PY_VERSION="

where py >nul 2>nul
if errorlevel 1 goto :resolve_python_no_py

REM Try py -3.13
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.13
REM Try py -3.12
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.12
REM Try py -3.11
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.11
REM Try py -3.10
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.10

REM Try py -3 (default Python 3), but only if it is in the supported range.
REM Without that guard a machine whose default is newer than the torch pin
REM supports gets accepted here and fails during dependency install.
if defined PY_BOOTSTRAP goto :resolve_python_done_py
REM stdout-gated for the same reason as :try_py_version.
set "PY_PROBE="
for /f "tokens=*" %%O in ('py -3 -c "import sys; print(1 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 0)" 2^>nul') do set "PY_PROBE=%%O"
if not "%PY_PROBE%"=="1" goto :resolve_python_no_py
set "PY_IS_STORE="
py -3 -c "import sys; sys.exit(1 if 'windowsapps' in (sys.base_prefix + sys.executable).lower() else 0)" >nul 2>nul
if errorlevel 1 set "PY_IS_STORE=1"
if defined PY_IS_STORE echo   [INFO] Skipping the default Python 3: Microsoft Store build (WindowsApps execution alias).
if defined PY_IS_STORE goto :resolve_python_no_py
set "PY_BOOTSTRAP=py -3"
for /f "tokens=*" %%O in ('py -3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"') do set "PY_VERSION=%%O"

:resolve_python_done_py
if defined PY_BOOTSTRAP exit /b 0

:resolve_python_no_py
REM Try 'python' command
where python >nul 2>nul
if errorlevel 1 goto :resolve_python_no_python
REM stdout-gated for the same reason as :try_py_version.
set "PY_PROBE="
for /f "tokens=*" %%O in ('python -c "import sys; print(sys.version_info.major)" 2^>nul') do set "PY_PROBE=%%O"
if not defined PY_PROBE goto :resolve_python_no_python
set "PY_IS_STORE="
python -c "import sys; sys.exit(1 if 'windowsapps' in (sys.base_prefix + sys.executable).lower() else 0)" >nul 2>nul
if errorlevel 1 set "PY_IS_STORE=1"
if defined PY_IS_STORE echo   [INFO] Skipping 'python': Microsoft Store build (WindowsApps execution alias).
if defined PY_IS_STORE goto :resolve_python_no_python
set "PY_BOOTSTRAP=python"
for /f "tokens=*" %%O in ('python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"') do set "PY_VERSION=%%O"
if defined PY_BOOTSTRAP exit /b 0

:resolve_python_no_python
REM Try 'python3' command
where python3 >nul 2>nul
if errorlevel 1 goto :resolve_python_auto_install
REM stdout-gated for the same reason as :try_py_version.
set "PY_PROBE="
for /f "tokens=*" %%O in ('python3 -c "import sys; print(sys.version_info.major)" 2^>nul') do set "PY_PROBE=%%O"
if not defined PY_PROBE goto :resolve_python_auto_install
set "PY_IS_STORE="
python3 -c "import sys; sys.exit(1 if 'windowsapps' in (sys.base_prefix + sys.executable).lower() else 0)" >nul 2>nul
if errorlevel 1 set "PY_IS_STORE=1"
if defined PY_IS_STORE echo   [INFO] Skipping 'python3': Microsoft Store build (WindowsApps execution alias).
if defined PY_IS_STORE goto :resolve_python_auto_install
set "PY_BOOTSTRAP=python3"
for /f "tokens=*" %%O in ('python3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"') do set "PY_VERSION=%%O"
if defined PY_BOOTSTRAP exit /b 0

:resolve_python_auto_install
REM Try auto-install via winget as last resort
where winget >nul 2>nul
if errorlevel 1 exit /b 1

echo   [INFO] Attempting to install Python 3.11 via winget...
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [INFO] Automatic Python install did not succeed.
  exit /b 1
)

REM Check standard install location
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  set "PY_BOOTSTRAP=%LocalAppData%\Programs\Python\Python311\python.exe"
  set "PY_VERSION=3.11 (just installed)"
  exit /b 0
)

REM Try py launcher after install
where py >nul 2>nul
if errorlevel 1 goto :resolve_python_post_install_python
REM stdout-gated for the same reason as :try_py_version -- this is the very
REM same py.exe, so the same negative exit code applies straight after a
REM winget install that did not register a 3.11 runtime with the launcher.
set "PY_PROBE="
for /f "tokens=*" %%O in ('py -3.11 -c "import sys; print(1)" 2^>nul') do set "PY_PROBE=%%O"
if not defined PY_PROBE goto :resolve_python_post_install_python
set "PY_BOOTSTRAP=py -3.11"
set "PY_VERSION=3.11 (just installed)"
exit /b 0

:resolve_python_post_install_python
where python >nul 2>nul
if errorlevel 1 goto :resolve_python_need_restart
REM stdout-gated for the same reason as :try_py_version.
set "PY_PROBE="
for /f "tokens=*" %%O in ('python -c "import sys; print(1)" 2^>nul') do set "PY_PROBE=%%O"
if not defined PY_PROBE goto :resolve_python_need_restart
set "PY_BOOTSTRAP=python"
set "PY_VERSION=3.11 (just installed - may need terminal restart)"
exit /b 0

:resolve_python_need_restart
echo.
echo   [INFO] Python was installed but is not yet in PATH.
echo          Please CLOSE this terminal, open a new one, and rerun this script.
echo.
exit /b 1

:try_py_version
REM %1 = version like 3.12
REM Do not gate on errorlevel here. The Python Install Manager build of
REM py.exe exits 0xA0000006 -- which cmd reads as -1610612730 -- when the
REM requested runtime is not installed, and "if errorlevel N" is a signed
REM ">= N" test, so every negative code slips straight through. Measured on
REM 2026-08-18 on a machine carrying only 3.11: the gate MISSED, so
REM PY_BOOTSTRAP became "py -3.13", the prerequisite line printed
REM "Python ...... OK ()" with an empty version, and STEP 2 went on to
REM report "Virtualenv created successfully" for a venv it never created.
REM Gate on stdout instead: a usable interpreter prints, a missing one
REM does not. Same reasoning as the nvidia-smi fix in the CUDA check.
set "PY_PROBE="
for /f "tokens=*" %%O in ('py -%1 -c "import sys; print(sys.version_info.major)" 2^>nul') do set "PY_PROBE=%%O"
if not defined PY_PROBE exit /b 1
REM Reject the Microsoft Store build and fall through to the next candidate.
REM Its python.exe is a WindowsApps execution alias that resolves only inside
REM an interactive desktop session, so a venv built on it cannot be launched
REM from a scheduled task, a service or an SSH session -- the process dies
REM with "Unable to create process". Measured on a Surface Book 3 on
REM 2026-08-18, where py -3.13 was the Store build and this script duly
REM picked it -- three lines below a comment about not letting uvicorn pick
REM up a Store Python. If every candidate is a Store build the winget branch
REM installs a real one, which is the outcome we want.
REM
REM Written as single-line ifs on purpose: a multi-line "if ( ... )" block
REM is what cmd mis-parses when this file has LF-only line endings, which is
REM exactly how it arrives in a release source zip (git exports the blob,
REM and .gitattributes eol=crlf only applies on checkout).
set "PY_IS_STORE="
py -%1 -c "import sys; sys.exit(1 if 'windowsapps' in (sys.base_prefix + sys.executable).lower() else 0)" >nul 2>nul
if errorlevel 1 set "PY_IS_STORE=1"
if defined PY_IS_STORE echo   [INFO] Skipping Python %1: Microsoft Store build (WindowsApps execution alias).
if defined PY_IS_STORE exit /b 1
set "PY_BOOTSTRAP=py -%1"
for /f "tokens=*" %%O in ('py -%1 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"') do set "PY_VERSION=%%O"
exit /b 0

:check_python_version
REM Verify Python is within the supported range (3.10 - 3.13).
REM
REM The upper bound matters as much as the lower one: requirements.txt pins a
REM torch version, and torch only publishes wheels for a bounded set of
REM interpreters (as of torch 2.6, up to cp313). On a newer Python pip cannot
REM resolve the pin and the install dies minutes in, with an error that looks
REM like a download failure. Keep this in step with the torch pin.
if not defined PY_BOOTSTRAP exit /b 1
REM stdout-gated for the same reason as :try_py_version.
set "PY_PROBE="
for /f "tokens=*" %%O in ('%PY_BOOTSTRAP% -c "import sys; print(1 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 0)" 2^>nul') do set "PY_PROBE=%%O"
if not "%PY_PROBE%"=="1" exit /b 1
exit /b 0
