#!/bin/bash
# POLICY: For full-run experiments, submit this from Tamia or Nibi
# (see CLAUDE.md "Cluster Policy" and slurm/bootstrap-remote-cluster.sh).
# On Mila only run debug-sized invocations (--max-steps 2 / --dry-run).
# Caller passes --output=$SCRATCH/slurm-logs/slurm-%j.out via sbatch CLI.
#SBATCH --job-name=train-defense-tar
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1 2>/dev/null || true

REPO_ROOT="${REPO_ROOT:-${HOME}/information-safety}"
cd "${REPO_ROOT}"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1

TARGET="${TARGET:-wmdp}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SEED="${SEED:-0}"
EVILMATH_REWRITE_PATH="${EVILMATH_REWRITE_PATH:-${SCRATCH}/information-safety/defense-data/evilmath/llm_rewrite.jsonl}"

DEFENSE_ID=$(python scripts/train_defense.py \
    --print-defense-id \
    --method tar \
    --target "${TARGET}" \
    --base-model "${BASE_MODEL}" \
    --seed "${SEED}")
OUTPUT_DIR="${SCRATCH}/information-safety/defenses/${DEFENSE_ID}"

echo "Training TAR defense: target=${TARGET} base=${BASE_MODEL} seed=${SEED}"
echo "Output dir: ${OUTPUT_DIR}"

python scripts/train_defense.py \
    --method tar \
    --target "${TARGET}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --evilmath-rewrite-path "${EVILMATH_REWRITE_PATH}" \
    "$@"
