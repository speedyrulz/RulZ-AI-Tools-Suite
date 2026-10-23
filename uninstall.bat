@echo off
rem Removes what install.bat added. Your images and Losers folders are never touched.
setlocal
set "APP=RulZ Image Tournament"

rem The RULZ_* locations can be preset to match a custom install (used for testing).
if not defined RULZ_INSTALL_DIR set "RULZ_INSTALL_DIR=%LOCALAPPDATA%\Programs\%APP%"
if not defined RULZ_START_MENU call :known_folder Programs RULZ_START_MENU
if not defined RULZ_DESKTOP call :known_folder Desktop RULZ_DESKTOP
if not defined RULZ_SENDTO call :known_folder SendTo RULZ_SENDTO
if not defined RULZ_DATA_DIR set "RULZ_DATA_DIR=%APPDATA%\%APP%"

del "%RULZ_START_MENU%\%APP%.lnk" 2>nul
del "%RULZ_DESKTOP%\%APP%.lnk" 2>nul
del "%RULZ_SENDTO%\%APP%.lnk" 2>nul

rem Delete only the files install.bat copied, then the folder if it's empty.
del "%RULZ_INSTALL_DIR%\%APP%.exe" 2>nul
del "%RULZ_INSTALL_DIR%\README.md" 2>nul
rmdir "%RULZ_INSTALL_DIR%" 2>nul
if exist "%RULZ_INSTALL_DIR%\%APP%.exe" (
    echo Couldn't remove the app. If %APP% is open, close it and run uninstall.bat again.
    pause
    exit /b 1
)
echo Removed %APP% and its shortcuts.

if exist "%RULZ_DATA_DIR%" (
    echo.
    set "WIPE="
    set /p "WIPE=Also delete saved settings and tournament progress? [y/N] "
    call :maybe_wipe
)
echo.
pause
exit /b 0

:maybe_wipe
if /i "%WIPE%"=="y" (
    del "%RULZ_DATA_DIR%\settings.ini" "%RULZ_DATA_DIR%\tournament.txt" "%RULZ_DATA_DIR%\tournament.txt.tmp" 2>nul
    rmdir "%RULZ_DATA_DIR%" 2>nul
    echo Settings deleted.
)
exit /b 0

:known_folder
for /f "usebackq delims=" %%d in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('%1')"`) do set "%2=%%d"
exit /b 0
