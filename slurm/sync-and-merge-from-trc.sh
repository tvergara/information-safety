#!/bin/bash
# Pull results+generations from trc and merge them into the canonical Mila
# final-results.jsonl. Refuses to run if an eval-sweep job is queued/running
# on Mila (concurrent rewrites would race the merge).
#
# trc's /work mount isn't visible on the login node, and `eai data pull` of
# a directory tree (~7.5k small files across 3.8k dirs) stalls indefinitely.
# Workaround: a small eai CPU job tars /work/...-results/generations into a
# single ~200 MB tarball and copies the jsonl alongside it; then we pull
# those two single files and rsync them home.
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
eai_image="registry.toolkit-sp.yul201.service-now.com/snow.research.mmteb/mteb-lite:v1"
trc_staging_subdir="trc-results-staging"
work_results_root="/work/information-safety-results"
work_staging="/work/sync-staging"

mila_results_dir="/network/scratch/b/brownet/information-safety/results"
mila_generations_dir="/network/scratch/b/brownet/information-safety/generations"
mkdir -p "$mila_results_dir" "$mila_generations_dir/from-trc"

local_results="$mila_results_dir/final-results.jsonl.trc"
local_generations="$mila_generations_dir/from-trc/"

tar_cmd="mkdir -p $work_staging && rm -f $work_staging/generations.tar.gz $work_staging/main-generations.tar.gz $work_staging/final-results.jsonl $work_staging/main-final-results.jsonl"
tar_cmd="$tar_cmd && cd $work_results_root && tar czf $work_staging/generations.tar.gz generations"
tar_cmd="$tar_cmd && cp $work_results_root/results/final-results.jsonl $work_staging/final-results.jsonl"
tar_cmd="$tar_cmd && if [ -d $work_results_root/main/generations ]; then cd $work_results_root/main && tar czf $work_staging/main-generations.tar.gz generations; fi"
tar_cmd="$tar_cmd && if [ -f $work_results_root/main/final-results.jsonl ]; then cp $work_results_root/main/final-results.jsonl $work_staging/main-final-results.jsonl; fi"
tar_cmd="$tar_cmd && ls -lh $work_staging/"

echo "[trc] Submitting eai tar job..."
tar_uuid=$(ssh trc "eai job submit --image $eai_image --data $data_resource:/work:rw --cpu 4 --mem 16 --max-run-time 1800 -- bash -lc $(printf %q "$tar_cmd")" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)
echo "[trc] tar job uuid: $tar_uuid"

echo "[trc] Waiting for tar job to finish..."
while true; do
    state=$(ssh trc "eai job info $tar_uuid" | awk '/^state:/ {print $2}')
    case "$state" in
        SUCCEEDED) break ;;
        FAILED|CANCELLED) echo "[trc] tar job entered terminal state $state"; ssh trc "eai job logs $tar_uuid | tail -40" >&2; exit 1 ;;
        *) echo "[trc] tar job state: $state"; sleep 30 ;;
    esac
done
echo "[trc] tar job SUCCEEDED."

echo "[trc] Pulling tarballs + jsonls out of the data resource..."
ssh trc "mkdir -p ~/$trc_staging_subdir/sync-staging && rm -f ~/$trc_staging_subdir/sync-staging/generations.tar.gz ~/$trc_staging_subdir/sync-staging/main-generations.tar.gz ~/$trc_staging_subdir/sync-staging/final-results.jsonl ~/$trc_staging_subdir/sync-staging/main-final-results.jsonl"
ssh trc "cd ~/$trc_staging_subdir && eai data pull $data_resource ./sync-staging/generations.tar.gz ."
ssh trc "cd ~/$trc_staging_subdir && eai data pull $data_resource ./sync-staging/final-results.jsonl ."
ssh trc "cd ~/$trc_staging_subdir && eai data pull $data_resource ./sync-staging/main-generations.tar.gz . || echo 'no main-generations tarball'"
ssh trc "cd ~/$trc_staging_subdir && eai data pull $data_resource ./sync-staging/main-final-results.jsonl . || echo 'no main jsonl'"

echo "[mila] rsync tarballs + jsonls from trc login..."
local_adversariallm_jsonl="$mila_results_dir/final-results.jsonl.trc.adversariallm"
local_pool_jsonl="$mila_results_dir/final-results.jsonl.trc.pool"
rsync -avz "trc:$trc_staging_subdir/sync-staging/final-results.jsonl" "$local_adversariallm_jsonl"
rsync -avz "trc:$trc_staging_subdir/sync-staging/generations.tar.gz" "$mila_generations_dir/from-trc/generations.tar.gz"
rsync -avz "trc:$trc_staging_subdir/sync-staging/main-final-results.jsonl" "$local_pool_jsonl" 2>/dev/null || rm -f "$local_pool_jsonl"
rsync -avz "trc:$trc_staging_subdir/sync-staging/main-generations.tar.gz" "$mila_generations_dir/from-trc/main-generations.tar.gz" 2>/dev/null || true

echo "[mila] Concatenating per-source jsonls into $local_results ..."
cat "$local_adversariallm_jsonl" > "$local_results"
if [ -f "$local_pool_jsonl" ]; then
    cat "$local_pool_jsonl" >> "$local_results"
fi

echo "[mila] Extracting tarballs into $local_generations ..."
tar xzf "$mila_generations_dir/from-trc/generations.tar.gz" -C "$mila_generations_dir/from-trc/" --strip-components=1
rm -f "$mila_generations_dir/from-trc/generations.tar.gz"
if [ -f "$mila_generations_dir/from-trc/main-generations.tar.gz" ]; then
    tar xzf "$mila_generations_dir/from-trc/main-generations.tar.gz" -C "$mila_generations_dir/from-trc/" --strip-components=1
    rm -f "$mila_generations_dir/from-trc/main-generations.tar.gz"
fi

echo ""
echo "Synced from trc:"
echo "  results -> $local_results"
echo "  generations -> $local_generations"

python "$repo_dir/scripts/merge_synced_results.py" --cluster trc

echo ""
echo "Merge complete. When ready, backfill ASR with:"
echo "  sbatch slurm/eval-sweep-results.sh"
