@echo off
REM RulZ Huggingface Manager - Windows launcher (double-click me)
setlocal
cd /d "%~dp0"

set "VENV=.venv"
set "PY="

if exist "%VENV%\Scripts\python.exe" goto :have_venv

where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python 3.10+ was not found. Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^).
    pause
    exit /b 1
)

echo Creating virtual environment in %VENV% ...
%PY% -m venv "%VENV%" || (echo Failed to create the virtual environment. & pause & exit /b 1)

:have_venv
"%VENV%\Scripts\python.exe" -c "import nicegui, huggingface_hub" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies ...
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
    "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || (echo Dependency install failed. & pause & exit /b 1)
)

"%VENV%\Scripts\python.exe" -m rulz_hf_manager %*
if errorlevel 1 pause
endlocal
