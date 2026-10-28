@echo off
rem Installs RulZ Image Tournament for the current user (no admin rights needed).
rem Builds the app, copies it to %LOCALAPPDATA%\Programs\RulZ Image Tournament, and adds
rem Start Menu, Desktop and "Send to" shortcuts. Run it again to update after changing the code.
setlocal
set "APP=RulZ Image Tournament"
set "SRC=%~dp0"

rem The RULZ_* locations can be preset to install somewhere else (used for testing).
if not defined RULZ_INSTALL_DIR set "RULZ_INSTALL_DIR=%LOCALAPPDATA%\Programs\%APP%"
if not defined RULZ_START_MENU call :known_folder Programs RULZ_START_MENU
if not defined RULZ_DESKTOP call :known_folder Desktop RULZ_DESKTOP
if not defined RULZ_SENDTO call :known_folder SendTo RULZ_SENDTO

echo Building %APP%...
powershell -NoProfile -ExecutionPolicy Bypass -File "%SRC%build.ps1"
if errorlevel 1 goto :failed

echo Installing to %RULZ_INSTALL_DIR%
if not exist "%RULZ_INSTALL_DIR%" mkdir "%RULZ_INSTALL_DIR%"
copy /y "%SRC%bin\%APP%.exe" "%RULZ_INSTALL_DIR%\" >nul
if errorlevel 1 (
    echo Couldn't copy the app. If %APP% is open, close it and run install.bat again.
    goto :failed
)
copy /y "%SRC%README.md" "%RULZ_INSTALL_DIR%\" >nul

call :shortcut "%RULZ_START_MENU%\%APP%.lnk" || goto :failed
call :shortcut "%RULZ_DESKTOP%\%APP%.lnk" || goto :failed
call :shortcut "%RULZ_SENDTO%\%APP%.lnk" || goto :failed

echo.
echo Installed. Start it from the Start Menu or the Desktop shortcut.
echo You can also right-click images or a folder and choose Send to ^> %APP%.
echo To remove it, run uninstall.bat.
echo.
pause
exit /b 0

:failed
echo.
echo Install failed.
pause
exit /b 1

rem Sets variable %2 to the path of Windows known folder %1 (handles OneDrive-redirected Desktops).
:known_folder
for /f "usebackq delims=" %%d in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('%1')"`) do set "%2=%%d"
exit /b 0

:shortcut
set "RULZ_LNK=%~1"
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:RULZ_LNK); $s.TargetPath = Join-Path $env:RULZ_INSTALL_DIR ($env:APP + '.exe'); $s.WorkingDirectory = $env:RULZ_INSTALL_DIR; $s.Description = 'Pick your favorite image, two at a time'; $s.Save()"
if errorlevel 1 (
    echo Couldn't create shortcut %RULZ_LNK%
    exit /b 1
)
echo Added shortcut %RULZ_LNK%
exit /b 0
