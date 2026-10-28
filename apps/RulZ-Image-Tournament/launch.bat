@echo off
rem Starts RulZ Image Tournament from this folder, building it first if needed.
rem Images or folders dropped onto this file open in the app.
setlocal
set "EXE=%~dp0bin\RulZ Image Tournament.exe"

if not exist "%EXE%" (
    echo Building RulZ Image Tournament...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1"
    if errorlevel 1 (
        echo Build failed.
        pause
        exit /b 1
    )
)

start "" "%EXE%" %*
