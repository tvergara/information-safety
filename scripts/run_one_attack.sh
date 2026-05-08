#!/bin/bash
# Run one AdversariaLLM attack shard end-to-end.
# Invoked by the cluster job-pool worker (CUDA_VISIBLE_DEVICES is preset).
set -euo pipefail

while [ $# -gt 0 ]; do
    case "$1" in
        --attack) ATTACK="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --shard-start) SHARD_START="$2"; shift 2 ;;
        --shard-end) SHARD_END="$2"; shift 2 ;;
        --behaviors-csv) BEHAVIORS_CSV="$2"; shift 2 ;;
        --output-jsonl) OUTPUT_JSONL="$2"; shift 2 ;;
        --adv-save-dir) ADV_SAVE_DIR="$2"; shift 2 ;;
        --adversariallm-venv) ADVERSARIALLM_VENV="$2"; shift 2 ;;
        --repo-root) REPO_ROOT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

DATASET_IDX="[$(seq -s, "${SHARD_START}" "$((SHARD_END - 1))")]"

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "${ADV_SAVE_DIR}"

MAIN_VENV="${REPO_ROOT}/.venv"

(
    cd "${REPO_ROOT}/third_party/adversariallm"
    # shellcheck disable=SC1091
    source "${ADVERSARIALLM_VENV}/bin/activate"
    export ADVERSARIALLM_USE_MONGOMOCK=1
    export MONGODB_DB=adversariallm
    python run_attacks.py \
        attack="${ATTACK}" \
        model="${MODEL}" \
        dataset=adv_behaviors \
        "datasets.adv_behaviors.messages_path=${BEHAVIORS_CSV}" \
        "datasets.adv_behaviors.idx=${DATASET_IDX}" \
        "datasets.adv_behaviors.shuffle=false" \
        classifiers=null \
        "save_dir=${ADV_SAVE_DIR}"
)

# shellcheck disable=SC1091
source "${MAIN_VENV}/bin/activate"
python scripts/extract_adversariallm_attacks.py \
    --adversariallm_run_dir "${ADV_SAVE_DIR}" \
    --output "${OUTPUT_JSONL}"

echo "Wrote ${OUTPUT_JSONL}"
