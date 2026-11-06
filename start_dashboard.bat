@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.conda-training\python.exe" (
    "%~dp0.conda-training\python.exe" "%~dp0src\launcher.py"
) else (
    py -3.10 "%~dp0src\launcher.py"
)
if errorlevel 1 (
    echo.
    echo Dashboard startup failed. See the message above.
    pause
)
