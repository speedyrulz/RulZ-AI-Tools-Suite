# Builds a throw-away copy of RulZ Image Grid outside the repo, compiles capture_image_grid.cs against it and runs
# it, for the screenshots. The repo is only read: build.ps1, src\ and assets\ are copied to -Build and built there
# with the copy of build.ps1, so no bin\ folder appears in the repo. RULZ_IMAGEGRID_DATA points the app's
# settings.ini at -Data (a folder that must not exist yet), so the real %APPDATA%\RulZ Image Grid is left alone.
#
#   powershell -ExecutionPolicy Bypass -File capture_image_grid.ps1 -Build R:\grid-build -Data R:\grid-data `
#       -Images R:\Grid-demo -Out R:\grid-out\image-grid [-Readme R:\grid-out-readme]
param(
    [string]$Repo = "",  # default: apps\RulZ-Image-Grid in this repository
    [Parameter(Mandatory)] [string]$Build,
    [Parameter(Mandatory)] [string]$Data,
    [Parameter(Mandatory)] [string]$Images,
    [Parameter(Mandatory)] [string]$Out,
    [string]$Readme = ""
)
$ErrorActionPreference = 'Stop'
if (-not $Repo) { $Repo = Join-Path $PSScriptRoot "..\..\apps\RulZ-Image-Grid" }
$Repo = (Resolve-Path $Repo).Path
if (-not (Select-String -Quiet -SimpleMatch "RULZ_IMAGEGRID_DATA" "$Repo\src\Theme.cs")) {
    throw "Theme.cs doesn't read RULZ_IMAGEGRID_DATA: the capture would use the real %APPDATA% folder"
}
if (Test-Path $Build) { throw "$Build already exists: give a new folder" }
if (Test-Path $Data) { throw "$Data already exists: give a new folder" }

New-Item -ItemType Directory -Force $Build, $Data | Out-Null
Copy-Item "$Repo\build.ps1" $Build
Copy-Item "$Repo\src" "$Build\src" -Recurse
Copy-Item "$Repo\assets" "$Build\assets" -Recurse
& "$Build\build.ps1"
$bin = "$Build\bin"

# The same Roslyn csc that build.ps1 found.
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
$csc = Join-Path $vs 'MSBuild\Current\Bin\Roslyn\csc.exe'
$fw = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319"
Copy-Item (Join-Path $PSScriptRoot 'capture_image_grid.cs') $Build
& $csc -nologo -target:exe -platform:anycpu -langversion:latest "-out:$bin\capture_image_grid.exe" `
    "-r:$bin\RulZ Image Grid.exe" "-r:$fw\System.dll" "-r:$fw\System.Core.dll" "-r:$fw\System.Xaml.dll" `
    "-r:$fw\WPF\WindowsBase.dll" "-r:$fw\WPF\PresentationCore.dll" "-r:$fw\WPF\PresentationFramework.dll" `
    "$Build\capture_image_grid.cs"
if ($LASTEXITCODE -ne 0) { throw "Harness build failed ($LASTEXITCODE)" }

$env:RULZ_IMAGEGRID_DATA = $Data
$harnessArgs = @('--images', $Images, '--out', $Out)
if ($Readme) { $harnessArgs += @('--readme', $Readme) }
& "$bin\capture_image_grid.exe" @harnessArgs
if ($LASTEXITCODE -ne 0) { throw "Capture failed ($LASTEXITCODE)" }
