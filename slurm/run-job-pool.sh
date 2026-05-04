#!/bin/bash
# Cluster Job Pool: one fat-allocation SLURM job that runs 4 single-GPU
# experiments in parallel from a queue under QUEUE_ROOT.
#
# Tamia/Nibi force whole-node allocations (4xH100); this script grabs one
# whole node and launches 4 background workers, each pinned to a distinct
# CUDA device. Re-submit the same QUEUE_ROOT to resume after a time limit.
#
#SBATCH --job-name=job-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00

set -euo pipefail

QUEUE_ROOT="${1:-${QUEUE_ROOT:-}}"
if [ -z "${QUEUE_ROOT}" ]; then
    SCRATCH_BASE="${SCRATCH:-/network/scratch/b/brownet}"
    QUEUE_ROOT="${SCRATCH_BASE}/information-safety/job-pool/default"
fi
export QUEUE_ROOT

mkdir -p "${QUEUE_ROOT}/pending" "${QUEUE_ROOT}/claimed" \
    "${QUEUE_ROOT}/done" "${QUEUE_ROOT}/failed" "${QUEUE_ROOT}/logs"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
export HYDRA_FULL_ERROR=1

echo "Job pool starting at $(date)"
echo "QUEUE_ROOT=${QUEUE_ROOT}"
echo "Initial pending count: $(find "${QUEUE_ROOT}/pending" -name '*.json' | wc -l)"

# Reaper failure is non-fatal: stale claims in claimed/ do not block workers
# from draining pending/, and the next pool startup retries the reap. Don't
# abort a 12h allocation over a transient squeue blip.
python scripts/reap_stale_claims.py --queue-root "${QUEUE_ROOT}" \
    || echo "WARNING: reaper failed; proceeding"

PIDS=()
for WORKER_INDEX in 0 1 2 3; do
    WORKER_INDEX="${WORKER_INDEX}" QUEUE_ROOT="${QUEUE_ROOT}" \
        python scripts/job_pool_worker.py \
            --queue-root "${QUEUE_ROOT}" \
            --worker-index "${WORKER_INDEX}" \
            > "${QUEUE_ROOT}/logs/worker-${WORKER_INDEX}.log" 2>&1 &
    PIDS+=($!)
done

EXIT_CODE=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        EXIT_CODE=1
    fi
done

DONE_COUNT=$(find "${QUEUE_ROOT}/done" -name '*.json' | wc -l)
FAILED_COUNT=$(find "${QUEUE_ROOT}/failed" -name '*.json' | wc -l)
PENDING_COUNT=$(find "${QUEUE_ROOT}/pending" -name '*.json' | wc -l)

echo "Job pool finished at $(date)"
echo "Done: ${DONE_COUNT}  Failed: ${FAILED_COUNT}  Pending left: ${PENDING_COUNT}"

exit "${EXIT_CODE}"
