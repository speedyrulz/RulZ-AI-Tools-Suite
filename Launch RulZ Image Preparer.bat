@echo off
rem Starts RulZ Image Preparer: the built .exe if present, otherwise from source.
setlocal
cd /d "%~dp0"

set "EXE=%~dp0dist\RulZ Image Preparer\RulZ Image Preparer.exe"
if exist "%EXE%" (
    start "" "%EXE%"
    exit /b 0
)

if not exist ".venv\Scripts\pythonw.exe" (
    echo Setting up RulZ Image Preparer for the first time...
    python -m venv .venv || goto :nopython
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :failed
)
start "" ".venv\Scripts\pythonw.exe" run.py
exit /b 0

:nopython
echo Python 3.10 or newer is required to run from source. Install it from https://www.python.org/
pause
exit /b 1

:failed
echo Installing dependencies failed. See the messages above.
pause
exit /b 1
