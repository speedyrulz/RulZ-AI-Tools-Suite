#!/usr/bin/env bash
# Create (or print) the vast.ai template for RulzTrainer with the vast CLI.
#
#   pip install vastai && vastai set api-key <your key>      # once (https://cloud.vast.ai/cli/)
#   bash scripts/vast/create_template.sh                      # creates the template in "My Templates"
#   PRINT=1 bash scripts/vast/create_template.sh              # only print the command
#
# What the template does: stock vastai/pytorch image (CUDA 12.8 build, so RTX 30/40/50 hosts), Jupyter + SSH launch
# mode, and PROVISIONING_SCRIPT = scripts/vast/provision.sh from this repo, which installs RulzTrainer into
# /workspace on first boot and registers the WebUI behind the Instance Portal (external port 7870, with the portal's
# authentication). Model downloads (about 30 GB) land on the instance disk, hence the disk size.
#
# Secrets never go into the template: put HF_TOKEN (gated models) into your account's environment variables at
# https://cloud.vast.ai/account/ ("Environment Variables"); they are injected into every instance.
set -eo pipefail

NAME="${NAME:-RulzTrainer}"
IMAGE="${IMAGE:-vastai/pytorch:cuda-12.8.1-auto}"
DISK="${DISK:-120}"
REPO="${REPO:-https://github.com/speedyrulz/RulzTrainer}"
BRANCH="${BRANCH:-main}"
MODEL_BUCKET="${MODEL_BUCKET:-}"  # e.g. speedyrulz/MOSS-Image-bucket: downloaded into models/bucket on first boot
PROVISION_URL="${PROVISION_URL:-https://raw.githubusercontent.com/speedyrulz/RulzTrainer/${BRANCH}/scripts/vast/provision.sh}"
PORTAL="localhost:1111:11111:/:Instance Portal|localhost:7870:17870:/:RulzTrainer|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal"
ENV_OPTS="-p 1111:1111 -p 7870:7870 -p 8080:8080 -p 22:22 -e OPEN_BUTTON_PORT=1111 -e PROVISIONING_SCRIPT=${PROVISION_URL} -e RULZ_REPO=${REPO}.git -e RULZ_BRANCH=${BRANCH} -e PORTAL_CONFIG='${PORTAL}'"
[ -n "${MODEL_BUCKET}" ] && ENV_OPTS="${ENV_OPTS} -e RULZ_MODEL_BUCKET=${MODEL_BUCKET}"
ONSTART="/opt/instance-tools/bin/entrypoint.sh"
SEARCH="${SEARCH:-gpu_ram>=16 num_gpus=1 disk_space>=${DISK} inet_down>=200 cuda_max_good>=12.8 rented=False}"
README_TEXT="RulzTrainer: MOSS-Image LoRA trainer with a WebUI. First boot installs the trainer (3-5 min, watch the provisioning log in the Instance Portal); then open RulzTrainer from the portal. Set HF_TOKEN in your account environment variables for gated models. Datasets: upload on the Dataset tab or with Jupyter into /workspace/RulzTrainer/datasets; results are in /workspace/RulzTrainer/output."

cmd=(vastai create template --name "$NAME" --image "$IMAGE"
     --env "$ENV_OPTS"
     --onstart-cmd "$ONSTART"
     --search_params "$SEARCH"
     --disk_space "$DISK" --jupyter --jupyter-lab --ssh --direct
     --href "$REPO" --repo "$REPO"
     --desc "MOSS-Image LoRA trainer (WebUI) - RulzTrainer"
     --readme "$README_TEXT")

printf '%q ' "${cmd[@]}"; echo
if [ "${PRINT:-0}" != "1" ]; then
    command -v vastai >/dev/null || { echo "vastai CLI not found: pip install vastai && vastai set api-key <key>"; exit 1; }
    "${cmd[@]}"
fi
