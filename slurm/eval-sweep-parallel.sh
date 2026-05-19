#!/bin/bash
# Parallel ASR backfill dispatcher on Mila.
#
# Submits SLURM job arrays (one per dataset) that each compute ASR for a hash-disjoint
# shard of the unevaluated rows in final-results.jsonl. Each shard writes a per-shard
# delta JSONL to $SHARD_DIR; nothing rewrites the canonical results file. A final
# merger job (dependent on all shards) applies the deltas atomically.
#
# Policy: this script lives on Mila and respects the same constraint as
# slurm/eval-sweep-results.sh -- do not run while any eval-pool job is queued or
# running on Mila that writes to the canonical jsonl. The TRC eval pool writes to
# TRC's own results file and does not race.
#
# Usage:
#     bash slurm/eval-sweep-parallel.sh
#
# Optional environment overrides:
#     HARMBENCH_SHARDS    (default 4) -- GPU job array width for HarmBench
#     STRONGREJECT_SHARDS (default 2) -- GPU job array width for StrongREJECT
#     SHARD_DIR           (default $SCRATCH/.../asr-backfill/<timestamp>)
#     RESULTS_FILE        (default canonical Mila final-results.jsonl)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
cd "$repo_dir"

results_file="${RESULTS_FILE:-/network/scratch/b/brownet/information-safety/results/final-results.jsonl}"
timestamp="$(date +%Y%m%d-%H%M%S)"
shard_dir="${SHARD_DIR:-/network/scratch/b/brownet/information-safety/asr-backfill/${timestamp}}"
mkdir -p "$shard_dir"

slurm_logs="/network/scratch/b/brownet/information-safety/slurm-logs"
mkdir -p "$slurm_logs"

harmbench_shards="${HARMBENCH_SHARDS:-4}"
strongreject_shards="${STRONGREJECT_SHARDS:-2}"

echo "[dispatch] shard_dir = $shard_dir"
echo "[dispatch] results_file = $results_file"

# Refuse to dispatch if any conflicting job is queued/running.
inflight=$(squeue -u "$USER" --noheader -h -o "%j" 2>/dev/null \
    | grep -E '^(eval-sweep|eval-shard-)' || true)
if [ -n "$inflight" ]; then
    echo "ERROR: conflicting jobs already queued/running:" >&2
    echo "$inflight" >&2
    exit 1
fi

worker_script="$script_dir/_eval-shard.sh"

submit_gpu_array () {
    local name="$1"; local evaluator="$2"; local num_shards="$3"; local prefix="$4"
    local array_spec
    if [ "$num_shards" -le 1 ]; then
        array_spec="--array=0-0"
    else
        array_spec="--array=0-$((num_shards - 1))"
    fi
    sbatch --parsable \
        --job-name="$name" \
        --partition=long \
        --gres=gpu:a100l:1 \
        --cpus-per-task=4 \
        --mem=48G \
        --time=6:00:00 \
        $array_spec \
        --output="$slurm_logs/slurm-%A_%a.out" \
        "$worker_script" "$evaluator" "1" "$results_file" "$shard_dir" "$num_shards" "$prefix"
}

submit_cpu_single () {
    local name="$1"; local evaluator="$2"; local prefix="$3"
    sbatch --parsable \
        --job-name="$name" \
        --partition=long-cpu \
        --cpus-per-task=2 \
        --mem=8G \
        --time=2:00:00 \
        --output="$slurm_logs/slurm-%j.out" \
        "$worker_script" "$evaluator" "0" "$results_file" "$shard_dir" "1" "$prefix"
}

hb_id=$(submit_gpu_array eval-shard-harmbench scripts/evaluate_harmbench.py \
    "$harmbench_shards" "harmbench")
echo "[dispatch] HarmBench array: $hb_id (shards=$harmbench_shards)"

sr_id=$(submit_gpu_array eval-shard-strongreject scripts/evaluate_strongreject.py \
    "$strongreject_shards" "strongreject")
echo "[dispatch] StrongREJECT array: $sr_id (shards=$strongreject_shards)"

wmdp_id=$(submit_cpu_single eval-shard-wmdp scripts/evaluate_wmdp.py "wmdp")
echo "[dispatch] WMDP cpu job: $wmdp_id"

evilmath_id=$(submit_cpu_single eval-shard-evilmath scripts/evaluate_evilmath.py "evilmath")
echo "[dispatch] EvilMath cpu job: $evilmath_id"

dependency="afterok:${hb_id}:${sr_id}:${wmdp_id}:${evilmath_id}"

merge_id=$(sbatch --parsable \
    --job-name=eval-shard-merge \
    --partition=long-cpu \
    --cpus-per-task=2 \
    --mem=4G \
    --time=00:30:00 \
    --dependency="$dependency" \
    --kill-on-invalid-dep=yes \
    --output="$slurm_logs/slurm-%j.out" \
    "$script_dir/_eval-shard-merge.sh" "$results_file" "$shard_dir")
echo "[dispatch] Merge job (depends on all shards): $merge_id"

echo ""
echo "Submitted. Watch with:"
echo "  squeue -u \$USER --name=eval-shard-harmbench,eval-shard-strongreject,eval-shard-wmdp,eval-shard-evilmath,eval-shard-merge"
echo ""
echo "Shard deltas: $shard_dir"
echo "Logs:         $slurm_logs/slurm-*.out"
