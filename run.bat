@echo off
setlocal
cd /d "%~dp0"

rem Usage: run.bat        start the app
rem        run.bat dev    start with hot reload (development)
rem        run.bat build  rebuild, then start

if not exist "node_modules\electron\dist\electron.exe" (
  echo RulZ Coding Agent is not installed yet. Run install.bat first.
  pause
  exit /b 1
)

if /i "%~1"=="dev" (
  call npm run dev
  exit /b %errorlevel%
)

if /i "%~1"=="build" goto :build
if not exist "out\main\index.js" goto :build
goto :start

:build
echo Building RulZ Coding Agent...
call npm run build
if errorlevel 1 (
  echo [ERROR] Build failed.
  pause
  exit /b 1
)

:start
start "" "node_modules\electron\dist\electron.exe" "%~dp0."
exit /b 0
