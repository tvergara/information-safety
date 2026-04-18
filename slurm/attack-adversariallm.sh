#!/bin/bash
# ------------------------------------------------------------------------------
# Run an AdversariaLLM attack (GCG, AutoDAN, PAIR, PGD, ...) and convert its
# output to a flat `{behavior, adversarial_prompt, attack_flops}` JSONL under
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
# Runs as an 8-way array sweep over the 200 HarmBench standard behaviors:
# each task owns 25 contiguous behaviors (DATASET_IDX auto-derived from
# SLURM_ARRAY_TASK_ID). To submit a single shard as a smoke, override with
# `sbatch --array=0` and/or `DATASET_IDX=[0,1]`.
#
# Environment-overridable args:
#   ATTACK_KEY             — AdversariaLLM attack key (default gcg; e.g. autodan,
#                            pair, pgd, ...)
#   MODEL_KEY              — AdversariaLLM model key (a full HF path, e.g.
#                            meta-llama/Llama-2-7b-chat-hf — see
#                            third_party/adversariallm/conf/models/models.yaml)
#   DATASET_KEY            — AdversariaLLM dataset key (default adv_behaviors)
#   HYDRA_ATTACK_OVERRIDES — space-separated Hydra key=value tokens applied to
#                            `run_attacks.py`. Use to override per-attack knobs
#                            (e.g. "attacks.gcg.num_steps=500 attacks.gcg.search_width=512")
#   DATASET_IDX            — explicit list selector; when unset, derived from
#                            SLURM_ARRAY_TASK_ID as a 25-behavior slice of [0,200)
#   OUTPUT_JSONL           — destination flat JSONL (default under attacks/)
#
# Example invocations:
#   GCG with legacy knobs:
#     ATTACK_KEY=gcg \
#     MODEL_KEY=meta-llama/Llama-2-7b-chat-hf \
#     HYDRA_ATTACK_OVERRIDES="attacks.gcg.num_steps=250 attacks.gcg.search_width=256 \
#         attacks.gcg.optim_str_init='x x x x x x x x x x x x x x x x x x x x'" \
#     sbatch slurm/attack-adversariallm.sh
#
#   AutoDAN with defaults:
#     ATTACK_KEY=autodan \
#     MODEL_KEY=meta-llama/Meta-Llama-3.1-8B-Instruct \
#     sbatch slurm/attack-adversariallm.sh
# ------------------------------------------------------------------------------
#SBATCH --job-name=attack-adversariallm
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|hopper|lovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%A_%a.out

set -euo pipefail

ATTACK_KEY="${ATTACK_KEY:-gcg}"
MODEL_KEY="${MODEL_KEY:-meta-llama/Llama-2-7b-chat-hf}"
DATASET_KEY="${DATASET_KEY:-adv_behaviors}"
HYDRA_ATTACK_OVERRIDES="${HYDRA_ATTACK_OVERRIDES:-}"

# Array task → 25-behavior slice of the 200-behavior standard split. Honor an
# explicit DATASET_IDX override (e.g. for manual single-behavior smoke runs).
TOTAL_BEHAVIORS=200
SHARD_SIZE=25
if [[ -z "${DATASET_IDX:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SHARD_START=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
    SHARD_END=$(( SHARD_START + SHARD_SIZE - 1 ))
    if (( SHARD_END >= TOTAL_BEHAVIORS )); then
        SHARD_END=$(( TOTAL_BEHAVIORS - 1 ))
    fi
    DATASET_IDX="[$(seq -s, "${SHARD_START}" "${SHARD_END}")]"
fi
DATASET_IDX="${DATASET_IDX:-[0,1,2,3,4]}"

RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
SHARD_SUFFIX="${SLURM_ARRAY_TASK_ID:+-shard${SLURM_ARRAY_TASK_ID}}"

SCRATCH_ROOT=/network/scratch/b/brownet/information-safety
ATTACKS_DIR="${SCRATCH_ROOT}/attacks"
ADV_SAVE_DIR="${SCRATCH_ROOT}/adversariallm-outputs/${RUN_ID}${SHARD_SUFFIX}"
MODEL_SLUG="${MODEL_KEY//\//_}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${ATTACKS_DIR}/${ATTACK_KEY}-${MODEL_SLUG}-${RUN_ID}${SHARD_SUFFIX}.jsonl}"
# Restrict AdversariaLLM to the 200 `standard` HarmBench behaviors (what our
# eval uses via walledai/HarmBench "standard"). The upstream CSV also includes
# 100 contextual + 100 copyright rows, which would waste attack compute and never
# match our eval set.
STANDARD_CSV="${ATTACKS_DIR}/harmbench_behaviors_standard.csv"
UPSTREAM_CSV=/home/mila/b/brownet/information-safety/third_party/adversariallm/data/behavior_datasets/harmbench_behaviors_text_all.csv

mkdir -p "${ATTACKS_DIR}" "${ADV_SAVE_DIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1 || true

cd /home/mila/b/brownet/information-safety

source .venv/bin/activate
python -c "
import pandas as pd
df = pd.read_csv('${UPSTREAM_CSV}')
df[df['FunctionalCategory'] == 'standard'].to_csv('${STANDARD_CSV}', index=False)
"

# --- Attack optimization via AdversariaLLM -----------------------------------
ADVERSARIALLM_ACTIVATE="${ADVERSARIALLM_ACTIVATE:-/network/scratch/b/brownet/information-safety/envs/adversariallm/bin/activate}"
if [[ ! -f "${ADVERSARIALLM_ACTIVATE}" ]]; then
    echo "ERROR: AdversariaLLM env activation script not found at ${ADVERSARIALLM_ACTIVATE}" >&2
    echo "See the header of this script for setup instructions." >&2
    exit 1
fi

(
    cd third_party/adversariallm
    # shellcheck disable=SC1090
    source "${ADVERSARIALLM_ACTIVATE}"
    # AdversariaLLM hard-depends on MongoDB for run dedup + config logging,
    # neither of which we use. The venv ships a .pth that swaps
    # pymongo.MongoClient for mongomock when this flag is set.
    export ADVERSARIALLM_USE_MONGOMOCK=1
    export MONGODB_DB=adversariallm
    # shellcheck disable=SC2086
    python run_attacks.py \
        attack="${ATTACK_KEY}" \
        model="${MODEL_KEY}" \
        dataset="${DATASET_KEY}" \
        datasets.${DATASET_KEY}.messages_path="${STANDARD_CSV}" \
        datasets.${DATASET_KEY}.idx="${DATASET_IDX}" \
        datasets.${DATASET_KEY}.shuffle=false \
        ${HYDRA_ATTACK_OVERRIDES} \
        classifiers=null \
        save_dir="${ADV_SAVE_DIR}"
)

# --- Convert to flat JSONL ---------------------------------------------------
source .venv/bin/activate
python scripts/extract_adversariallm_attacks.py \
    --adversariallm_run_dir "${ADV_SAVE_DIR}" \
    --output "${OUTPUT_JSONL}"

echo "Wrote ${OUTPUT_JSONL}"
