# Builds "RulZ Image Grid.exe" into .\bin using the Roslyn C# compiler that ships with
# Visual Studio / Build Tools, targeting the .NET Framework 4.8 built into Windows 10/11.
# Pass -Test to also build and run the tests in tests\ first.
param([switch]$Test)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$csc = $null
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -products * -property installationPath
    if ($vs) {
        $c = Join-Path $vs 'MSBuild\Current\Bin\Roslyn\csc.exe'
        if (Test-Path $c) { $csc = $c }
    }
}
if (-not $csc) {
    $c = Get-ChildItem "$env:ProgramFiles\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio" `
        -Recurse -Filter csc.exe -ErrorAction SilentlyContinue | Where-Object FullName -like '*Roslyn*' | Select-Object -First 1
    if ($c) { $csc = $c.FullName }
}
if (-not $csc) { throw 'Roslyn csc.exe not found. Install Visual Studio or "Build Tools for Visual Studio".' }

$fw  = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319"
$wpf = "$fw\WPF"
$out = Join-Path $root 'bin'
New-Item -ItemType Directory -Force $out | Out-Null

$refs = @(
    "$fw\System.dll", "$fw\System.Core.dll", "$fw\System.Xaml.dll",
    "$wpf\WindowsBase.dll", "$wpf\PresentationCore.dll", "$wpf\PresentationFramework.dll"
) | ForEach-Object { "-r:$_" }
$sources = Get-ChildItem (Join-Path $root 'src') -Filter *.cs | ForEach-Object { $_.FullName }
$exe = Join-Path $out 'RulZ Image Grid.exe'

if ($Test) {
    $testExe = Join-Path $out 'ImageGridTests.exe'
    & $csc -nologo -target:exe -platform:anycpu -optimize+ -langversion:latest -main:ImageGridTests `
        "-out:$testExe" $refs $sources (Join-Path $root 'tests\ImageGridTests.cs')
    if ($LASTEXITCODE -ne 0) { throw "Test build failed ($LASTEXITCODE)" }
    & $testExe
    if ($LASTEXITCODE -ne 0) { throw "Tests failed ($LASTEXITCODE)" }
}

$assets = Join-Path $root 'assets'
$resources = @(
    "-resource:$(Join-Path $root 'src\Theme.xaml'),RulZ.ImageGrid.Theme.xaml",
    "-resource:$assets\logo.png,RulZ.ImageGrid.logo.png",
    "-resource:$assets\logo-light.png,RulZ.ImageGrid.logo-light.png",
    "-resource:$assets\app.ico,RulZ.ImageGrid.app.ico"
)

& $csc -nologo -target:winexe -platform:anycpu -optimize+ -langversion:latest `
    "-out:$exe" "-win32manifest:$(Join-Path $root 'src\app.manifest')" "-win32icon:$assets\app.ico" `
    $resources $refs $sources
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }
Write-Host "Built $exe"
