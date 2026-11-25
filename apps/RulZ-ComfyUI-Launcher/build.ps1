# Builds dist\RulZ ComfyUI Launcher\RulZ ComfyUI Launcher.exe
# Needs: python -m pip install -r requirements-dev.txt
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

python -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }

python -m PyInstaller --noconfirm 'RulZ ComfyUI Launcher.spec'
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed.' }

Write-Host 'Built dist\RulZ ComfyUI Launcher\RulZ ComfyUI Launcher.exe. Keep the whole folder together.'
