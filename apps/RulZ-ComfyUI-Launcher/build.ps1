# Builds dist\RulZ ComfyUI Launcher\RulZ ComfyUI Launcher.exe
# Usage:  powershell -ExecutionPolicy Bypass -File .\build.ps1 [-Test]
# -Test runs the test suite first (always on in GitHub Actions, where a failure stops the release).
param([switch]$Test)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    python -m venv .venv
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
.\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt pyinstaller
if ($LASTEXITCODE) { exit $LASTEXITCODE }

if ($Test -or $env:GITHUB_ACTIONS -eq "true") {
    .\.venv\Scripts\python.exe -m pip install --quiet pytest
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean "RulZ ComfyUI Launcher.spec"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Write-Host "`nBuilt: $PSScriptRoot\dist\RulZ ComfyUI Launcher\RulZ ComfyUI Launcher.exe. Keep the whole folder together."
