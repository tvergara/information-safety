#!/bin/bash
# Final merge job for the parallel ASR backfill. Applies all per-shard delta
# files to the canonical final-results.jsonl atomically.

set -euo pipefail

cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

results_file="$1"
shard_dir="$2"

echo "[merge] results_file=$results_file"
echo "[merge] shard_dir=$shard_dir"

# Skip the writer-lock check: by the time this dependent job runs, the shard
# arrays have finished and the deltas are stable side-files. The legacy
# eval-sweep collision is moot because we replaced that workflow.
python scripts/merge_asr_backfill.py \
    --results-file "$results_file" \
    --shard-dir "$shard_dir" \
    --backup \
    --skip-writer-lock-check

echo "[merge] done."
