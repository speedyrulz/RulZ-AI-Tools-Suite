#!/bin/bash
# supervisor wrapper for the RulzTrainer WebUI on vast.ai base images (installed by provision.sh / the Dockerfile).
# Binds to localhost:17870; the Instance Portal's Caddy proxies external port 7870 to it with authentication.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"      # /etc/environment (template vars such as HF_TOKEN) and ${WORKSPACE}/.env
. "${utils}/exit_portal.sh" "RulzTrainer"   # only start when the app is listed in /etc/portal.yaml

. /venv/main/bin/activate
cd "${WORKSPACE:-/workspace}/RulzTrainer"
# compiled kernels on the workspace volume, so a restarted instance does not compile from scratch again
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${WORKSPACE:-/workspace}/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE:-/workspace}/.cache/triton}"
exec python webui.py --host 127.0.0.1 --port "${RULZ_PORT:-17870}"
