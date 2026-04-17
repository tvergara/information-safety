#!/bin/bash
# ------------------------------------------------------------------------------
# Run a GCG attack via the AdversariaLLM submodule and convert its output to a
# flat `{behavior, adversarial_prompt}` JSONL under
# /network/scratch/b/brownet/information-safety/attacks/.
#
# One-time setup (not performed by this script):
#   1. Initialize the submodule:
#      git submodule update --init --recursive
#   2. Create an isolated AdversariaLLM environment. The upstream README
#      recommends pixi (`pixi install --locked` inside
#      third_party/adversariallm/), or alternatively a fresh venv using
#      third_party/adversariallm/requirements.txt. Do NOT install AdversariaLLM
#      into the parent .venv — dependency pins collide with ours.
#   3. Point ADVERSARIALLM_ACTIVATE below at the activation command for that
#      environment (e.g. the path to a `pixi shell-hook`-emitted script or a
#      `source .venv/bin/activate` line in the submodule directory).
#
# Environment-overridable args (defaults match the smoke-run config in the
# spec: 5 behaviors × 100 steps × 256 search width × 20-token suffix):
#   MODEL_KEY          — AdversariaLLM model key (e.g. llama2)
#   DATASET_KEY        — AdversariaLLM dataset key (default adv_behaviors)
#   NUM_STEPS          — GCG steps per behavior (default 100)
#   SEARCH_WIDTH       — GCG search width (default 256)
#   SUFFIX_LENGTH      — optim_str token count (default 20)
#   DATASET_IDX        — list selector passed to AdversariaLLM (default [0,1,2,3,4])
#   OUTPUT_JSONL       — destination flat JSONL (default under attacks/)
# ------------------------------------------------------------------------------
#SBATCH --job-name=attack-gcg
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:15:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

set -euo pipefail

MODEL_KEY="${MODEL_KEY:-llama2}"
DATASET_KEY="${DATASET_KEY:-adv_behaviors}"
NUM_STEPS="${NUM_STEPS:-100}"
SEARCH_WIDTH="${SEARCH_WIDTH:-256}"
SUFFIX_LENGTH="${SUFFIX_LENGTH:-20}"
DATASET_IDX="${DATASET_IDX:-[0,1,2,3,4]}"

SCRATCH_ROOT=/network/scratch/b/brownet/information-safety
ATTACKS_DIR="${SCRATCH_ROOT}/attacks"
ADV_SAVE_DIR="${SCRATCH_ROOT}/adversariallm-outputs/${SLURM_JOB_ID:-manual}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${ATTACKS_DIR}/gcg-${MODEL_KEY}-${SLURM_JOB_ID:-manual}.jsonl}"

mkdir -p "${ATTACKS_DIR}" "${ADV_SAVE_DIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1 || true

cd /home/mila/b/brownet/information-safety

SUFFIX_INIT=$(printf 'x %.0s' $(seq "${SUFFIX_LENGTH}"))
SUFFIX_INIT="${SUFFIX_INIT% }"

# --- GCG optimization via AdversariaLLM --------------------------------------
ADVERSARIALLM_ACTIVATE="${ADVERSARIALLM_ACTIVATE:-third_party/adversariallm/.venv/bin/activate}"
if [[ ! -f "${ADVERSARIALLM_ACTIVATE}" ]]; then
    echo "ERROR: AdversariaLLM env activation script not found at ${ADVERSARIALLM_ACTIVATE}" >&2
    echo "See the header of this script for setup instructions." >&2
    exit 1
fi

(
    cd third_party/adversariallm
    # shellcheck disable=SC1090
    source "${ADVERSARIALLM_ACTIVATE}"
    python run_attacks.py \
        attack=gcg \
        model="${MODEL_KEY}" \
        dataset="${DATASET_KEY}" \
        datasets.${DATASET_KEY}.idx="${DATASET_IDX}" \
        datasets.${DATASET_KEY}.shuffle=false \
        attacks.gcg.num_steps="${NUM_STEPS}" \
        attacks.gcg.search_width="${SEARCH_WIDTH}" \
        attacks.gcg.optim_str_init="${SUFFIX_INIT}" \
        classifiers=null \
        save_dir="${ADV_SAVE_DIR}"
)

# --- Convert to flat JSONL ---------------------------------------------------
source .venv/bin/activate
python scripts/extract_adversariallm_attacks.py \
    --adversariallm_run_dir "${ADV_SAVE_DIR}" \
    --output "${OUTPUT_JSONL}"

echo "Wrote ${OUTPUT_JSONL}"
