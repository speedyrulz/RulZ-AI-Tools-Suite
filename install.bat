@echo off
setlocal
title RulZ AI Tools Suite - Install
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js was not found. Install Node.js 22 or newer from https://nodejs.org and run this again.
  goto :fail
)

echo [1/3] Installing dependencies...
call npm install --no-audit --no-fund
if errorlevel 1 goto :fail

echo [2/3] Downloading the Electron runtime...
call node node_modules\electron\install.js
if errorlevel 1 goto :fail

echo [3/3] Building the launcher...
call npm run build
if errorlevel 1 goto :fail

echo.
echo Done. Start the launcher with run.bat
pause
exit /b 0

:fail
echo.
echo Install failed. See the messages above.
pause
exit /b 1
