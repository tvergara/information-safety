#!/bin/bash
# Pull results+generations from trc and merge them into the canonical Mila
# final-results.jsonl.
#
# trc's /work mount isn't visible on the login node. The sync round-trips
# through the private HuggingFace dataset bucket: an eai job uploads the
# results jsonl + generations tarballs to <repo>/trc-sync/, then Mila pulls
# them back with snapshot_download. See scripts/sync_trc_via_hf.py.
#
# Refuses to run if an eval-sweep job is queued/running on Mila (concurrent
# rewrites would race the merge).
#
# Usage: bash slurm/sync-and-merge-from-trc.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

cd "$repo_dir"
exec python -m scripts.sync_trc_via_hf "$@"
