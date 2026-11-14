# Builds a throw-away copy of RulZ Image Tournament outside the repo, for the screenshots. The app keeps settings.ini
# and tournament.txt (the resumable tournament) in %APPDATA%\RulZ Image Tournament and rewrites both on exit;
# since 1.2.0 it reads them from <RULZ_TOURNAMENT_DATA>\RulZ Image Tournament instead when that variable is set, so
# capture_image_tournament.py points it at a workspace folder. The repo is only read: sources are copied to -Out
# and compiled there exactly like the repo's build.ps1, so no bin\ folder appears in the repo.
param(
    [string]$Repo = "",  # default: apps\RulZ-Image-Tournament in this repository
    [Parameter(Mandatory)] [string]$Out
)
$ErrorActionPreference = 'Stop'
if (-not $Repo) { $Repo = Join-Path $PSScriptRoot "..\..\apps\RulZ-Image-Tournament" }
$Repo = (Resolve-Path $Repo).Path
if (-not (Select-String -Quiet -SimpleMatch "RULZ_TOURNAMENT_DATA" "$Repo\src\Support.cs")) {
    throw "Support.cs doesn't read RULZ_TOURNAMENT_DATA: the capture would use the real %APPDATA% folder"
}
New-Item -ItemType Directory -Force "$Out\src", "$Out\assets" | Out-Null
Copy-Item "$Repo\src\*.cs" "$Out\src\"
Copy-Item "$Repo\app.manifest" $Out
Copy-Item "$Repo\assets\app.ico", "$Repo\assets\logo.png", "$Repo\assets\logo-light.png" "$Out\assets\"

$fx = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319"
Push-Location $Out
try {
    & "$fx\csc.exe" /nologo /target:winexe /platform:anycpu /optimize+ "/out:$Out\RulZ Image Tournament.exe" `
        /win32icon:assets\app.ico /win32manifest:app.manifest `
        "/resource:assets\logo.png,RulZ.ImageTournament.logo.png" `
        "/resource:assets\logo-light.png,RulZ.ImageTournament.logo-light.png" `
        /r:System.dll /r:System.Core.dll /r:System.Drawing.dll /r:System.Windows.Forms.dll `
        "/r:$fx\WPF\PresentationCore.dll" "/r:$fx\WPF\WindowsBase.dll" /r:System.Xaml.dll src\*.cs
    if ($LASTEXITCODE -ne 0) { throw "csc failed" }
} finally { Pop-Location }
Write-Host "Built $Out\RulZ Image Tournament.exe"
