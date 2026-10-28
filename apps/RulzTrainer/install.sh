#!/usr/bin/env bash
# RulzTrainer installer for Linux (desktop or server). Creates ./venv, installs CUDA PyTorch and the requirements.
#   ./install.sh                                        # cu128 wheels (RTX 30/40/50 series, driver >= 570)
#   ./install.sh https://download.pytorch.org/whl/cu130 # another PyTorch index
#   ./install.sh system                                 # keep the torch already installed in the active Python
#                                                       # (e.g. vast.ai / cloud images with /venv/main)
set -eo pipefail
cd "$(dirname "$0")"
TORCH_INDEX="${1:-https://download.pytorch.org/whl/cu128}"

if [ "$TORCH_INDEX" = "system" ]; then
    PY="$(command -v python3 || command -v python)"
    echo "Using the current Python: $PY ($("$PY" --version 2>&1))"
else
    if [ ! -x venv/bin/python ]; then
        echo "Creating virtual environment in ./venv ..."
        python3 -m venv venv
    fi
    PY=venv/bin/python
    "$PY" -m pip install --upgrade pip wheel
    echo "Installing PyTorch (CUDA) from $TORCH_INDEX ..."
    "$PY" -m pip install --upgrade "torch>=2.9" torchvision --index-url "$TORCH_INDEX"
fi

echo "Installing requirements ..."
if ! "$PY" -m pip install -r requirements.txt; then
    echo "[WARN] Full install failed (usually the optional face/body packages: insightface needs a C++ compiler)."
    echo "       Installing the core requirements only; face/body similarity scoring will be unavailable."
    grep -v -E "^\s*(#|$)|insightface|onnxruntime|mediapipe" requirements.txt > /tmp/rulz-core-req.txt
    "$PY" -m pip install -r /tmp/rulz-core-req.txt
fi

echo "Checking CUDA ..."
"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
echo "Install complete. Start the UI with ./run_webui.sh [port] [host]"
