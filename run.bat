@echo off
cd /d "%~dp0"
if not exist .venv (
  echo First run: creating virtual environment...
  python -m venv .venv || goto :err
  .venv\Scripts\python -m pip install -r requirements.txt || goto :err
)
start "" .venv\Scripts\pythonw.exe main.py
exit /b 0
:err
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
