@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 The Cls-Studio Contributors
REM Convenience wrapper so the launcher is visible right after unzip.
REM All options are forwarded - see scripts\windows\stop_local_windows.bat
call "%~dp0scripts\windows\stop_local_windows.bat" %*
