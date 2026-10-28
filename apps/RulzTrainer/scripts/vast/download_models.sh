#!/bin/bash
# Download model files from a Hugging Face Storage Bucket into the RulzTrainer models folder.
#
#   RULZ_MODEL_BUCKET=speedyrulz/MOSS-Image-bucket bash scripts/vast/download_models.sh
#
# provision.sh runs this in the background on first boot when RULZ_MODEL_BUCKET is set; you can rerun it any time
# (it syncs: files that are already complete are skipped, new or changed ones are fetched). The bucket's folder
# layout is kept, so a ComfyUI-style bucket lands as models/bucket/diffusion_models/..., text_encoders/..., vae/...
#
# Variables:
#   RULZ_MODEL_BUCKET    namespace/name of the bucket (required), or a full hf://buckets/... path with a prefix
#   RULZ_MODEL_DIR       destination (default: ${WORKSPACE}/RulzTrainer/models/bucket)
#   RULZ_MODEL_INCLUDE   optional space-separated patterns to fetch only some files, e.g. "vae/* text_encoders/*"
#   HF_TOKEN             needed only for a private bucket
set -eo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BUCKET="${RULZ_MODEL_BUCKET:-${1:-}}"
DEST="${RULZ_MODEL_DIR:-${WORKSPACE}/RulzTrainer/models/bucket}"
if [ -z "${BUCKET}" ]; then
    echo "[ERROR] set RULZ_MODEL_BUCKET (e.g. speedyrulz/MOSS-Image-bucket)"
    exit 1
fi
case "${BUCKET}" in
    hf://*) SRC="${BUCKET}" ;;
    *) SRC="hf://buckets/${BUCKET}" ;;
esac

[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate
if ! hf buckets --help >/dev/null 2>&1; then
    echo "== Upgrading huggingface_hub (the installed version has no bucket support)"
    pip install --quiet --upgrade "huggingface_hub[hf_xet]"
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

args=()
for pat in ${RULZ_MODEL_INCLUDE:-}; do
    args+=(--include "${pat}")
done

mkdir -p "${DEST}"
rm -f "${DEST}/.download_complete"
echo "== Syncing ${SRC} -> ${DEST} ($(date))"
for attempt in 1 2 3; do
    if hf buckets sync "${SRC}" "${DEST}" "${args[@]}"; then
        date > "${DEST}/.download_complete"
        echo "== Model download complete ($(date)):"
        (cd "${DEST}" && find . -type f ! -name '.download_complete' -printf '   %p  %s bytes\n' | sort)
        exit 0
    fi
    echo "[WARN] sync attempt ${attempt} failed; retrying in 15 s"
    sleep 15
done
echo "[ERROR] model download failed after 3 attempts; rerun: bash ${BASH_SOURCE[0]}"
exit 1
