@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo  RulzTrainer installer
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Install Python 3.10 - 3.12 from python.org
    echo         and tick "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Using Python %PYVER%

if exist "venv\Scripts\python.exe" (
    echo venv already exists, reusing it.
) else (
    echo Creating virtual environment in .\venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set PY=venv\Scripts\python.exe
set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
if not "%~1"=="" set "TORCH_INDEX=%~1"

echo.
echo Upgrading pip ...
"%PY%" -m pip install --upgrade pip wheel
if errorlevel 1 goto :fail

echo.
echo Installing PyTorch (CUDA) from %TORCH_INDEX% ...
echo (Pass a different index URL as the first argument, e.g. install.bat https://download.pytorch.org/whl/cu128)
"%PY%" -m pip install --upgrade "torch>=2.9" torchvision --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail

echo.
echo Installing requirements ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Checking CUDA ...
"%PY%" -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

echo.
echo ============================================
echo  Install complete. Run run_webui.bat to start.
echo ============================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. See the messages above.
pause
exit /b 1
