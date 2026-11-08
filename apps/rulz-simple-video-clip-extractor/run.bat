@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe python -m venv .venv
rem Install the packages on the first run and again whenever requirements.txt changes.
fc /b requirements.txt .venv\requirements.installed >nul 2>&1 || (
  .venv\Scripts\python -m pip install -r requirements.txt && copy /y requirements.txt .venv\requirements.installed >nul
)
start "" .venv\Scripts\pythonw.exe -m simplecut.app %*
