#!/bin/bash
# RulzTrainer provisioning for vast.ai (PROVISIONING_SCRIPT of the template; runs once on first boot of an instance
# built from vastai/pytorch or vastai/base-image, after supervisor is up).
#
#   - clones (or updates) the repo into ${WORKSPACE}/RulzTrainer (persistent instance disk)
#   - installs the requirements into the image's /venv/main (its torch is reused)
#   - registers the WebUI as a supervisor service behind the Instance Portal (external port 7870 -> local 17870,
#     Caddy adds the portal's token/basic auth)
#
# Variables (template or account environment): RULZ_REPO, RULZ_BRANCH (default main), HF_TOKEN (gated models).
set -eo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${RULZ_REPO:-https://github.com/speedyrulz/RulzTrainer.git}"
BRANCH="${RULZ_BRANCH:-main}"
DIR="${WORKSPACE}/RulzTrainer"

echo "== RulzTrainer provisioning: ${REPO_URL} (${BRANCH}) -> ${DIR}"
if [ -d "${DIR}/.git" ]; then
    git -C "${DIR}" fetch --quiet origin "${BRANCH}" && git -C "${DIR}" checkout --quiet "${BRANCH}" \
        && git -C "${DIR}" pull --quiet --ff-only || echo "[WARN] could not update the existing checkout; keeping it"
else
    git clone --quiet --branch "${BRANCH}" "${REPO_URL}" "${DIR}"
fi
mkdir -p "${DIR}/datasets" "${DIR}/models" "${DIR}/output"

echo "== Installing requirements into /venv/main (keeps the image's torch)"
. /venv/main/bin/activate
cd "${DIR}"
bash install.sh system

echo "== Registering the WebUI with supervisor"
install -m 0755 scripts/vast/rulztrainer.sh /opt/supervisor-scripts/rulztrainer.sh
install -m 0644 scripts/vast/rulztrainer.conf /etc/supervisor/conf.d/rulztrainer.conf
if [ -s /etc/portal.yaml ] && ! grep -qi rulztrainer /etc/portal.yaml; then
    echo "[WARN] 'RulzTrainer' is not in /etc/portal.yaml: the service will skip startup. Add"
    echo "       'localhost:7870:17870:/:RulzTrainer' to the template's PORTAL_CONFIG (and -p 7870:7870 to its ports)."
fi
supervisorctl reread && supervisorctl update
echo "== RulzTrainer ready: open it from the Instance Portal (or the instance's external port for 7870)"
