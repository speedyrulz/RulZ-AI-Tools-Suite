# Starts llama-server with the launch flags from the build spec.
#
# By default it uses the CUDA 12 llama-server bundled with LM Studio (newest version found).
# Point $env:LLAMA_SERVER_EXE at an official llama.cpp build to use that instead.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_llama_server.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start_llama_server.ps1 -Gpu 1 -Ctx 8192

param(
    [string]$ModelDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "LLM\Qwen3-VL-8B-Instruct-abliterated.Q8_0"),
    [string]$Model    = "Qwen3-VL-8B-Instruct-abliterated.Q8_0.gguf",
    [string]$MMProj   = "Qwen3-VL-8B-Instruct-abliterated.mmproj-f16.gguf",
    [string]$Gpu      = "0",        # index as shown by nvidia-smi
    [int]$Ctx         = 16384,
    [int]$Parallel    = 2,
    [int]$Port        = 8080
)

$ErrorActionPreference = "Stop"

$exe = $env:LLAMA_SERVER_EXE
if (-not $exe) {
    $backends = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends"
    $dir = Get-ChildItem $backends -Directory -Filter "llama.cpp-win-x86_64-nvidia-cuda12-avx2-*" -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "llama-server.exe") } |
        Sort-Object { [version]($_.Name -replace '.*-', '') } -Descending |
        Select-Object -First 1
    if (-not $dir) { throw "No llama-server.exe found. Set LLAMA_SERVER_EXE to a llama.cpp CUDA build." }
    $exe = Join-Path $dir.FullName "llama-server.exe"
    # LM Studio keeps the CUDA runtime DLLs in a separate vendor folder.
    $vendor = Join-Path $backends "vendor\win-llama-cuda12-vendor-v2"
    $env:PATH = "$($dir.FullName);$vendor;$env:PATH"
}

# Make CUDA device numbering match nvidia-smi.
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = $Gpu

Write-Host "llama-server: $exe"
& $exe `
    -m (Join-Path $ModelDir $Model) `
    --mmproj (Join-Path $ModelDir $MMProj) `
    -ngl 99 -fa on --jinja `
    -c $Ctx --parallel $Parallel `
    --host 127.0.0.1 --port $Port
