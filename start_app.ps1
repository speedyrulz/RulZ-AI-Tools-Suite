# One-click launcher: starts llama-server (unless it's already running), waits for the model to load,
# starts the crawler app, and opens it in the browser. Closing this window or pressing Ctrl+C stops
# everything it started.
#
#   powershell -ExecutionPolicy Bypass -File start_app.ps1
#   powershell -ExecutionPolicy Bypass -File start_app.ps1 -Gpu 1 -NoBrowser
#
# By default the model is loaded once per GPU (ports 8080, 8081, ...) and the app spreads images over them.

param(
    [int]$Port        = 8000,
    [int]$LlamaPort   = 8080,
    [string]$Gpu      = "all",   # "all", or nvidia-smi indexes like "0" or "0,1"; one model per GPU
    [int]$TimeoutSec  = 180,     # how long to wait for the model to load
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

# Windows job object with KILL_ON_JOB_CLOSE: every process assigned to it (and their children) dies
# when this launcher exits, however it exits - Ctrl+C, closing the window, or a crash.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class LauncherJob {
    [StructLayout(LayoutKind.Sequential)]
    struct BASIC { public long a, b; public uint LimitFlags; public UIntPtr c, d; public uint e; public UIntPtr f; public uint g, h; }
    [StructLayout(LayoutKind.Sequential)]
    struct IO { public ulong a, b, c, d, e, f; }
    [StructLayout(LayoutKind.Sequential)]
    struct EXT { public BASIC Basic; public IO Io; public UIntPtr p, q, r, s; }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr a, string n);
    [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr j, int c, ref EXT i, uint l);
    [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
    static IntPtr job = IntPtr.Zero;
    public static void Add(IntPtr process) {
        if (job == IntPtr.Zero) {
            job = CreateJobObject(IntPtr.Zero, null);
            var info = new EXT();
            info.Basic.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            SetInformationJobObject(job, 9, ref info, (uint)Marshal.SizeOf(typeof(EXT)));
        }
        AssignProcessToJobObject(job, process);
    }
}
"@

function Test-Url([string]$url) {
    try { return (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { return $false }
}

function Wait-Url([string]$url, [int]$seconds, [System.Diagnostics.Process]$proc, [string]$what, [string]$log) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Url $url) { return }
        if ($proc -and $proc.HasExited) {
            Write-Host "`n$what stopped unexpectedly. Last lines of ${log}:" -ForegroundColor Red
            if (Test-Path $log) { Get-Content $log -Tail 15 | ForEach-Object { Write-Host "  $_" } }
            throw "$what failed to start"
        }
        Start-Sleep -Milliseconds 500
        Write-Host "." -NoNewline
    }
    throw "$what did not become ready within $seconds s (see $log)"
}

# "all" = every NVIDIA card with room for the model (~13 GB), else a list like "0" or "0,1".
function Get-ModelGpus([string]$spec) {
    if ($spec -ne "all") {
        return @($spec -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return @("0") }
    $gpus = @(& nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>$null | ForEach-Object {
        $idx, $mem = $_ -split ","
        if ([int]$mem.Trim() -ge 14000) { $idx.Trim() }
    })
    if ($gpus.Count -eq 0) { return @("0") }
    return $gpus
}

$started = @()
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python is not on PATH." }

    # ---- 1. model servers: one per GPU, on consecutive ports ---------------------------------
    $gpus = Get-ModelGpus $Gpu
    $servers = @()
    for ($k = 0; $k -lt $gpus.Count; $k++) {
        $g = $gpus[$k]
        $gpuPort = $LlamaPort + $k
        $url = "http://127.0.0.1:$gpuPort"
        $log = Join-Path $logDir "llama-server-gpu$g.log"
        if (Test-Url "$url/health") {
            Write-Host "Model server already running on port $gpuPort - reusing it."
            $servers += [pscustomobject]@{ Gpu = $g; Url = $url; Proc = $null; Log = $log }
            continue
        }
        Write-Host "Starting the model on GPU $g (port $gpuPort, log: $log)"
        $proc = Start-Process powershell -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                            "`"$root\image_detection\scripts\start_llama_server.ps1`"", "-Gpu", $g, "-Port", $gpuPort)
        [LauncherJob]::Add($proc.Handle)
        $started += $proc
        $servers += [pscustomobject]@{ Gpu = $g; Url = $url; Proc = $proc; Log = $log }
    }
    Write-Host "Waiting for the model to load" -NoNewline
    $ready = @()
    foreach ($s in $servers) {
        try {
            Wait-Url "$($s.Url)/health" $TimeoutSec $s.Proc "The model server on GPU $($s.Gpu)" "$($s.Log).err"
            $ready += $s
        } catch {
            if ($servers.Count -eq 1) { throw }
            # e.g. the card is busy with another program: carry on with the others
            Write-Host "`nGPU $($s.Gpu) is not available ($($_.Exception.Message)); continuing without it." -ForegroundColor Yellow
            if ($s.Proc -and -not $s.Proc.HasExited) { Stop-Process -Id $s.Proc.Id -Force -ErrorAction SilentlyContinue }
        }
    }
    if ($ready.Count -eq 0) { throw "No model server could be started." }
    Write-Host " ready on GPU $(($ready | ForEach-Object { $_.Gpu }) -join ' and ')."

    # ---- 2. crawler app --------------------------------------------------------------------
    $appUrl = "http://127.0.0.1:$Port/"
    if (Test-Url "${appUrl}api/health") {
        Write-Host "The app is already running on port $Port."
        if (-not $NoBrowser) { Start-Process $appUrl }
        return
    }
    $appLog = Join-Path $logDir "crawler-app.log"
    Write-Host "Starting the app (log: $appLog)" -NoNewline
    $env:LLAMA_URLS = ($ready | ForEach-Object { $_.Url }) -join ","
    $app = Start-Process python -PassThru -WindowStyle Hidden -WorkingDirectory $root `
        -RedirectStandardOutput "$appLog.out" -RedirectStandardError $appLog `
        -ArgumentList @("-m", "crawler_app", "--port", $Port)
    [LauncherJob]::Add($app.Handle)
    $started += $app
    Wait-Url "${appUrl}api/health" 60 $app "The app" $appLog
    Write-Host " ready."

    if (-not $NoBrowser) { Start-Process $appUrl }
    Write-Host ""
    Write-Host "RulZ Image Crawler is running at $appUrl" -ForegroundColor Green
    Write-Host "Matched images are saved under $root\output"
    Write-Host "Close this window or press Ctrl+C to stop everything."
    Wait-Process -Id $app.Id
    Write-Host "The app stopped. See $appLog"
}
finally {
    # Normal exit / Ctrl+C. (If the window is closed instead, the job object does the same.)
    foreach ($p in $started) {
        if (-not $p.HasExited) {
            Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
