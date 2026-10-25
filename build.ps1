# Builds "bin\RulZ Image Tournament.exe" with the C# compiler that ships with Windows (.NET Framework 4.8).
# No SDK or Visual Studio needed. Pass -Test to also build and run the bracket logic tests.
param([switch]$Test)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$fx = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319"
$csc = "$fx\csc.exe"
$wpf = "$fx\WPF"
New-Item -ItemType Directory -Force bin | Out-Null

if ($Test) {
    & $csc /nologo /out:bin\TournamentTests.exe /r:System.Core.dll src\Tournament.cs tests\TournamentTests.cs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & .\bin\TournamentTests.exe
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $csc /nologo /target:winexe /platform:anycpu /optimize+ `
    "/out:bin\RulZ Image Tournament.exe" `
    /win32icon:assets\app.ico /win32manifest:app.manifest `
    "/resource:assets\logo.png,RulZ.ImageTournament.logo.png" `
    "/resource:assets\logo-light.png,RulZ.ImageTournament.logo-light.png" `
    /r:System.dll /r:System.Core.dll /r:System.Drawing.dll /r:System.Windows.Forms.dll /r:Microsoft.VisualBasic.dll `
    "/r:$wpf\PresentationCore.dll" "/r:$wpf\WindowsBase.dll" /r:System.Xaml.dll `
    src\*.cs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built bin\RulZ Image Tournament.exe"
