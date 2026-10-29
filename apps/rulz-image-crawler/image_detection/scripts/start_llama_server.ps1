# Starts llama-server with the launch flags from the build spec.
#
# It uses, in order: $env:LLAMA_SERVER_EXE, the llama.cpp folder next to image_detection (the RulZ AI Tools
# Suite launcher downloads one there), or the CUDA 12 llama-server bundled with LM Studio (newest version found).
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

$modelPath = Join-Path $ModelDir $Model
if (-not (Test-Path $modelPath)) {
    throw "The model isn't downloaded yet ($modelPath). Run: python image_detection\scripts\download_model.py"
}

$exe = $env:LLAMA_SERVER_EXE
# The llama.cpp build the RulZ AI Tools Suite launcher puts in the app folder.
$bundled = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) "llama.cpp\llama-server.exe"
if (-not $exe -and (Test-Path $bundled)) { $exe = $bundled }
if (-not $exe) {
    $backends = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends"
    $dir = Get-ChildItem $backends -Directory -Filter "llama.cpp-win-x86_64-nvidia-cuda12-avx2-*" -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "llama-server.exe") } |
        Sort-Object { [version]($_.Name -replace '.*-', '') } -Descending |
        Select-Object -First 1
    if (-not $dir) { throw "No llama-server.exe found. Put a llama.cpp CUDA build in the llama.cpp folder or set LLAMA_SERVER_EXE." }
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
