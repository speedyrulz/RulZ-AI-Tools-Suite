@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] No venv found. Run install.bat first.
    pause
    exit /b 1
)

set PORT=7870
if not "%~1"=="" set PORT=%~1

echo Starting RulzTrainer WebUI on http://127.0.0.1:%PORT%  (close this window to stop)
echo.
"venv\Scripts\python.exe" webui.py --port %PORT% --open
if errorlevel 1 (
    echo.
    echo [ERROR] The WebUI exited with an error. See the messages above.
    pause
)
