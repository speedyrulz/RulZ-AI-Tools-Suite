# Starts the tag-matching service on 127.0.0.1:8090 (llama-server must already be running).
# All knobs are environment variables; see README.md.
param([int]$Port = 8090)

# Run from the folder that contains the image_detection package so `python -m image_detection` finds it.
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
python -m image_detection --host 127.0.0.1 --port $Port
