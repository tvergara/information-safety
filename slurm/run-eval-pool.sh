#!/bin/bash
# Cluster vLLM Eval Pool: a single-GPU SLURM job that drains pending eval
# specs produced by DataStrategyDeferredEval and writes DataStrategyDeferredEval
# rows to the centralized results file.
#
# Re-submit the same QUEUE_ROOT to resume after a time limit.
#
#SBATCH --job-name=eval-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00

set -euo pipefail

QUEUE_ROOT="${1:-${QUEUE_ROOT:-}}"
BASE_MODEL="${2:-${BASE_MODEL:-}}"
if [ -z "${QUEUE_ROOT}" ] || [ -z "${BASE_MODEL}" ]; then
    echo "Usage: sbatch run-eval-pool.sh <queue-root> <base-model>" >&2
    exit 2
fi

SCRATCH_BASE="${SCRATCH:-/network/scratch/b/brownet}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${SCRATCH_BASE}/information-safety/adapters}"
RESULTS_FILE="${RESULTS_FILE:-${SCRATCH_BASE}/information-safety/results/final-results.jsonl}"
GENERATIONS_DIR="${GENERATIONS_DIR:-${SCRATCH_BASE}/information-safety/generations}"

mkdir -p "${QUEUE_ROOT}/pending" "${QUEUE_ROOT}/claimed" \
    "${QUEUE_ROOT}/done" "${QUEUE_ROOT}/failed" "${QUEUE_ROOT}/logs"

REPO_ROOT="${REPO_ROOT:-${HOME}/information-safety}"
cd "${REPO_ROOT}"

if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
module load cuda/12.4.1 2>/dev/null || true

# shellcheck disable=SC1091
source .venv/bin/activate

export PROJECT_ROOT="${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Eval pool starting at $(date)"
echo "QUEUE_ROOT=${QUEUE_ROOT}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "ADAPTER_ROOT=${ADAPTER_ROOT}"
echo "Initial pending count: $(find "${QUEUE_ROOT}/pending" -name '*.json' | wc -l)"

python scripts/run_eval_pool.py \
    --queue-root "${QUEUE_ROOT}" \
    --base-model "${BASE_MODEL}" \
    --adapter-root "${ADAPTER_ROOT}" \
    --results-file "${RESULTS_FILE}" \
    --generations-dir "${GENERATIONS_DIR}" \
    > "${QUEUE_ROOT}/logs/eval-worker.log" 2>&1

DONE_COUNT=$(find "${QUEUE_ROOT}/done" -name '*.json' | wc -l)
FAILED_COUNT=$(find "${QUEUE_ROOT}/failed" -name '*.json' | wc -l)
PENDING_COUNT=$(find "${QUEUE_ROOT}/pending" -name '*.json' | wc -l)

echo "Eval pool finished at $(date)"
echo "Done: ${DONE_COUNT}  Failed: ${FAILED_COUNT}  Pending left: ${PENDING_COUNT}"
