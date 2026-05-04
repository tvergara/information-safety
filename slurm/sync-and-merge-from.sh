#!/bin/bash
# Pull results+generations from a remote cluster (tamia or nibi) and merge
# them into the canonical Mila final-results.jsonl. Idempotent. Refuses to
# run if an eval-sweep job is already queued/running (concurrent rewrites
# would race against this merge).
#
# Usage: bash slurm/sync-and-merge-from.sh <tamia|nibi>

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tamia|nibi>" >&2
    exit 2
fi

cluster="$1"
case "$cluster" in
    tamia|nibi) ;;
    *)
        echo "Unknown cluster: $cluster (expected tamia or nibi)" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

eval_sweep_jobs="$(squeue -u "$USER" -n eval-sweep --noheader 2>/dev/null || true)"
if [ -n "$eval_sweep_jobs" ]; then
    echo "Refusing to merge: eval-sweep job(s) currently in the queue:" >&2
    echo "$eval_sweep_jobs" >&2
    echo "Wait for them to finish (or scancel) before re-running." >&2
    exit 1
fi

bash "$script_dir/sync-results-from.sh" "$cluster"

python "$repo_dir/scripts/merge_synced_results.py" --cluster "$cluster"

echo ""
echo "Merge complete. When ready, backfill ASR with:"
echo "  sbatch slurm/eval-sweep-results.sh"
