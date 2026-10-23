# Builds dist\RulZ Image Organizer\RulZ Image Organizer.exe
# Usage:  powershell -ExecutionPolicy Bypass -File .\build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    python -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt pyinstaller

$models = "models"
New-Item -ItemType Directory -Force $models | Out-Null
$base = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
if (-not (Test-Path "$models\face_detection_yunet_2023mar.onnx")) {
    Invoke-WebRequest "$base/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "$models\face_detection_yunet_2023mar.onnx"
}
if (-not (Test-Path "$models\face_recognition_sface_2021dec.onnx")) {
    Invoke-WebRequest "$base/face_recognition_sface/face_recognition_sface_2021dec.onnx" -OutFile "$models\face_recognition_sface_2021dec.onnx"
}

if (-not (Test-Path "$models\emotion-ferplus-8.onnx")) {
    Invoke-WebRequest "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" -OutFile "$models\emotion-ferplus-8.onnx"
}

.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean RulZImageOrganizer.spec
Write-Host "`nBuilt: $PSScriptRoot\dist\RulZ Image Organizer\RulZ Image Organizer.exe"
