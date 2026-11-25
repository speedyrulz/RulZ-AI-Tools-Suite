@echo off
rem Starts RulZ ComfyUI Launcher from source, without a console window.
cd /d "%~dp0"
where pythonw >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer, then run:  python -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" pythonw "main.pyw" %*
