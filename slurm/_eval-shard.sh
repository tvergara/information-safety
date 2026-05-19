#!/bin/bash
# Worker script for one ASR-backfill shard.
#
# Invoked via slurm/eval-sweep-parallel.sh. Args:
#     $1 = evaluator python script (e.g. scripts/evaluate_harmbench.py)
#     $2 = "1" to module-load CUDA before running, anything else to skip
#     $3 = canonical final-results.jsonl
#     $4 = shard output directory
#     $5 = total number of shards in this dataset's array
#     $6 = dataset prefix (used in the shard delta filename)
#
# Shard index comes from $SLURM_ARRAY_TASK_ID, defaulting to 0 for single-shot
# CPU jobs that are not submitted as an array.

set -euo pipefail

evaluator_script="$1"
load_cuda="$2"
results_file="$3"
shard_dir="$4"
num_shards="$5"
prefix="$6"

if [ "$load_cuda" = "1" ]; then
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    module load cuda/12.4.1
fi

cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

shard_index="${SLURM_ARRAY_TASK_ID:-0}"
output_file="$shard_dir/${prefix}.shard${shard_index}of${num_shards}.jsonl"
tmp_file="${output_file}.tmp"

echo "[shard] dataset=$prefix shard=$shard_index/$num_shards"
echo "[shard] writing to $output_file"

python "$evaluator_script" \
    --results-file "$results_file" \
    --shard-index "$shard_index" \
    --num-shards "$num_shards" \
    --output-shard-file "$tmp_file"

mv "$tmp_file" "$output_file"
echo "[shard] done."
