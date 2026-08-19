@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 The Cls-Studio Contributors
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_ABS=%%~fI"
call :find_repo_root "%SCRIPT_ABS%"
if not defined REPO_ROOT call :find_repo_root "%CD%"
if not defined REPO_ROOT (
  echo [ERROR] Could not find repository root.
  exit /b 1
)
cd /d "%REPO_ROOT%"

set "LOG_DIR=%REPO_ROOT%\logs\windows"

echo [INFO] Repo: %REPO_ROOT%
echo.
echo [INFO] Listening ports:
for %%P in (8791 5173) do (
  set "FOUND=0"
  for /f "tokens=5" %%I in ('netstat -ano ^| findstr /C:":%%P" ^| findstr /I /C:"LISTENING"') do (
    if "!FOUND!"=="0" (
      echo   port %%P : LISTENING (PID %%I)
      set "FOUND=1"
    )
  )
  if "!FOUND!"=="0" (
    echo   port %%P : NOT LISTENING
  )
)

echo.
echo [INFO] HTTP checks:
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$targets = @(" ^
  "  @{name='trainer_version'; url='http://127.0.0.1:8791/version'}," ^
  "  @{name='ui'; url='http://127.0.0.1:8791/ui/'}," ^
  ");" ^
  "foreach ($t in $targets) {" ^
  "  try {" ^
  "    $r = Invoke-WebRequest -UseBasicParsing -Uri $t.url -TimeoutSec 3;" ^
  "    Write-Host ('  ' + $t.name + ' : OK ' + $r.StatusCode)" ^
  "  } catch {" ^
  "    Write-Host ('  ' + $t.name + ' : FAIL ' + $_.Exception.Message)" ^
  "  }" ^
  "}"

echo.
if exist "%LOG_DIR%\trainer.log" (
  echo [INFO] trainer.log (tail 20)
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOG_DIR%\trainer.log' -Tail 20"
) else (
  echo [WARN] trainer.log not found: %LOG_DIR%\trainer.log
)

echo.
if exist "%LOG_DIR%\serving.log" (
  echo [INFO] serving.log (tail 20)
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOG_DIR%\serving.log' -Tail 20"
) else (
  echo [WARN] serving.log not found: %LOG_DIR%\serving.log
)

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
