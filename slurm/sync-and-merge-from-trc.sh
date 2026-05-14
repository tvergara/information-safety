#!/bin/bash
# Pull results+generations from trc and merge them into the canonical Mila
# final-results.jsonl. Refuses to run if an eval-sweep job is queued/running
# on Mila (concurrent rewrites would race the merge).
#
# trc's /work mount isn't visible on the login node, so we `eai data pull`
# (detached on trc, since its progress output is several GB) into a staging
# dir and rsync that back. Mila polls for a pull.done sentinel.
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

data_resource="snow.research.mmteb.safety"
trc_staging_subdir="trc-results-staging"

mila_results_dir="/network/scratch/b/brownet/information-safety/results"
mila_generations_dir="/network/scratch/b/brownet/information-safety/generations"
mkdir -p "$mila_results_dir" "$mila_generations_dir/from-trc"

local_results="$mila_results_dir/final-results.jsonl.trc"
local_generations="$mila_generations_dir/from-trc/"

remote_log="$trc_staging_subdir/pull.log"
remote_done="$trc_staging_subdir/pull.done"
remote_results_path="$trc_staging_subdir/information-safety-results/results/final-results.jsonl"
remote_generations_path="$trc_staging_subdir/information-safety-results/generations/"

echo "[trc] Launching detached eai data pull into ~/$trc_staging_subdir..."
ssh trc bash -s <<REMOTE_LAUNCH
set -euo pipefail
mkdir -p ~/$trc_staging_subdir
cd ~/$trc_staging_subdir
rm -f pull.done pull.log
nohup bash -c '
  set -e
  eai data pull $data_resource ./information-safety-results/results/final-results.jsonl . > pull.log 2>&1
  eai data pull $data_resource ./information-safety-results/generations . >> pull.log 2>&1
  touch pull.done
' > /dev/null 2>&1 &
disown
echo "launched"
REMOTE_LAUNCH

echo "[mila] Waiting for ~/$remote_done sentinel on trc..."
while ! ssh trc "test -f ~/$remote_done"; do
    n=$(ssh trc "find ~/$remote_generations_path -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l")
    echo "[trc] still pulling... staged generation dirs so far: $n"
    sleep 60
done
echo "[trc] pull complete."

echo "[mila] rsync results+generations from trc login..."
rsync -avz "trc:$remote_results_path" "$local_results"
rsync -avz --update "trc:$remote_generations_path" "$local_generations"

echo ""
echo "Synced from trc:"
echo "  results -> $local_results"
echo "  generations -> $local_generations"

python "$repo_dir/scripts/merge_synced_results.py" --cluster trc

echo ""
echo "Merge complete. When ready, backfill ASR with:"
echo "  sbatch slurm/eval-sweep-results.sh"
