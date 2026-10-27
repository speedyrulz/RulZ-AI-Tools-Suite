@echo off
setlocal
title RulZ Coding Agent - Install
cd /d "%~dp0"

echo ============================================
echo   RulZ Coding Agent - installer
echo ============================================
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js was not found. Install Node.js 22 or newer from https://nodejs.org and run this again.
  goto :fail
)
for /f "delims=" %%v in ('node --version') do echo Node.js %%v found.

echo.
echo [1/3] Installing dependencies...
call npm install --no-audit --no-fund
if errorlevel 1 (
  echo [ERROR] npm install failed.
  goto :fail
)

echo.
echo [2/3] Downloading the Electron runtime...
call node node_modules\electron\install.js
if errorlevel 1 (
  echo [ERROR] Could not download Electron.
  goto :fail
)

echo.
echo [3/3] Building the app...
call npm run build
if errorlevel 1 (
  echo [ERROR] Build failed.
  goto :fail
)

echo.
where git >nul 2>nul
if errorlevel 1 (
  echo [NOTE] Git for Windows was not found. "Undo last task" works best with it.
  echo        Install it from https://git-scm.com/download/win
) else (
  echo Git found: "Undo last task" will use it.
)

echo.
echo Done. Start the app with run.bat
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
