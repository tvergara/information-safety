#!/bin/bash
# Pull results from a remote cluster (tamia or nibi) back to Mila scratch.
#
# Usage: bash slurm/sync-results-from.sh <tamia|nibi>
#
# Writes to per-cluster suffixed files / subdirs to avoid clobbering the
# canonical Mila final-results.jsonl. Prints a follow-up command to merge
# by hand once the user has reviewed the pulled rows.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tamia|nibi>" >&2
    exit 2
fi

cluster="$1"

# Both tamia and nibi go through the robot host (publickey, no Duo).
case "$cluster" in
    tamia)
        rsync_host="robot.tamia.ecpia.ca"
        # Hardcoded because the robot wrapper rejects shell expansion so we
        # cannot probe $SCRATCH. Tamia uses an initial-letter scratch subdir.
        remote_scratch="/scratch/t/tvergara"
        ;;
    nibi)
        rsync_host="robot.nibi.alliancecan.ca"
        # Hardcoded — the robot wrapper rejects shell expansion so we can't
        # probe $SCRATCH, and the remote (Alliance) username differs from
        # Mila's $USER. Nibi scratch is flat (unlike Tamia's /scratch/t/...).
        remote_scratch="/scratch/tvergara"
        ;;
    *)
        echo "Unknown cluster: $cluster (expected tamia or nibi)" >&2
        exit 2
        ;;
esac

mila_results_dir="/network/scratch/b/brownet/information-safety/results"
mila_generations_dir="/network/scratch/b/brownet/information-safety/generations"
mkdir -p "$mila_results_dir" "$mila_generations_dir/from-$cluster"

remote_results="$remote_scratch/information-safety/results/final-results.jsonl"
remote_generations="$remote_scratch/information-safety/generations/"

local_results="$mila_results_dir/final-results.jsonl.$cluster"
local_generations="$mila_generations_dir/from-$cluster/"

rsync -avz "$rsync_host:$remote_results" "$local_results"
rsync -avz --update "$rsync_host:$remote_generations" "$local_generations"

echo ""
echo "Synced from $cluster:"
echo "  results -> $local_results"
echo "  generations -> $local_generations"
echo ""
echo "Sync complete; run 'bash slurm/sync-and-merge-from.sh $cluster' to merge into the canonical results file."
