#!/bin/bash
# Pull results+generations from trc /work and merge them into the canonical
# Mila final-results.jsonl. Idempotent. Refuses to run if an eval-sweep job
# is already queued/running on Mila (concurrent rewrites would race against
# this merge).
#
# Usage: bash slurm/sync-and-merge-from-trc.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

eval_sweep_jobs="$(squeue -u "$USER" -n eval-sweep --noheader 2>/dev/null || true)"
if [ -n "$eval_sweep_jobs" ]; then
    echo "Refusing to merge: eval-sweep job(s) currently in the queue:" >&2
    echo "$eval_sweep_jobs" >&2
    echo "Wait for them to finish (or scancel) before re-running." >&2
    exit 1
fi

trc_base="/work/information-safety-results"
mila_results_dir="/network/scratch/b/brownet/information-safety/results"
mila_generations_dir="/network/scratch/b/brownet/information-safety/generations"
mkdir -p "$mila_results_dir" "$mila_generations_dir/from-trc"

remote_results="trc:${trc_base}/results/final-results.jsonl"
remote_generations="trc:${trc_base}/generations/"

local_results="$mila_results_dir/final-results.jsonl.trc"
local_generations="$mila_generations_dir/from-trc/"

rsync -avz "$remote_results" "$local_results"
rsync -avz --update "$remote_generations" "$local_generations"

echo ""
echo "Synced from trc:"
echo "  results -> $local_results"
echo "  generations -> $local_generations"

python "$repo_dir/scripts/merge_synced_results.py" --cluster trc

echo ""
echo "Merge complete. When ready, backfill ASR with:"
echo "  sbatch slurm/eval-sweep-results.sh"
