@echo off
setlocal
cd /d "%~dp0"
py -3.10 "%~dp0src\launcher.py"
if errorlevel 1 (
    echo.
    echo Dashboard startup failed. See the message above.
    pause
)
