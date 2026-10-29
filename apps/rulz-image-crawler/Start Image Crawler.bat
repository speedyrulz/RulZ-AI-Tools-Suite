@echo off
rem Double-click to start the model, the app, and open it in your browser.
title RulZ Image Crawler
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_app.ps1" %*
if errorlevel 1 pause
