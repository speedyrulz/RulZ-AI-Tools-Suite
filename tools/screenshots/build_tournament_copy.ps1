# Builds a throw-away copy of RulZ Image Tournament whose settings/progress folder can be redirected with
# RULZ_TOURNAMENT_DATA, so screenshots never touch %APPDATA%\RulZ Image Tournament (the real app rewrites
# settings.ini and tournament.txt there on exit, and declining its resume prompt deletes the saved tournament).
# The repo is only read: sources are copied to -Out and patched there, then compiled like the repo's build.ps1.
param(
    [string]$Repo = "C:\coding\RulZ Image Tournament",
    [Parameter(Mandatory)] [string]$Out
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force "$Out\src", "$Out\assets" | Out-Null
Copy-Item "$Repo\src\*.cs" "$Out\src\"
Copy-Item "$Repo\app.manifest" $Out
Copy-Item "$Repo\assets\app.ico", "$Repo\assets\logo.png" "$Out\assets\"

$support = "$Out\src\Support.cs"
$text = [IO.File]::ReadAllText($support)
$old = 'Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "RulZ Image Tournament");'
$new = '(Environment.GetEnvironmentVariable("RULZ_TOURNAMENT_DATA") ?? Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData)), "RulZ Image Tournament");'
if (-not $text.Contains($old)) { throw "Support.cs changed: patch target not found" }
[IO.File]::WriteAllText($support, $text.Replace($old, $new), (New-Object Text.UTF8Encoding $true))

$fx = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319"
Push-Location $Out
try {
    & "$fx\csc.exe" /nologo /target:winexe /platform:anycpu /optimize+ "/out:$Out\RulZ Image Tournament.exe" `
        /win32icon:assets\app.ico /win32manifest:app.manifest "/resource:assets\logo.png,RulZ.ImageTournament.logo.png" `
        /r:System.dll /r:System.Core.dll /r:System.Drawing.dll /r:System.Windows.Forms.dll /r:Microsoft.VisualBasic.dll `
        "/r:$fx\WPF\PresentationCore.dll" "/r:$fx\WPF\WindowsBase.dll" /r:System.Xaml.dll src\*.cs
    if ($LASTEXITCODE -ne 0) { throw "csc failed" }
} finally { Pop-Location }
Write-Host "Built $Out\RulZ Image Tournament.exe"
