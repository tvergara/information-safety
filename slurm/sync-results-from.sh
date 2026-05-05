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

# tamia goes through the robot host (publickey, no Duo); nibi's robot grant is
# unconfirmed, so keep the ControlMaster path until it is.
case "$cluster" in
    tamia)
        rsync_host="robot.tamia.ecpia.ca"
        # Hardcoded because the robot wrapper rejects `bash`/shell expansion,
        # so we cannot probe $SCRATCH the way the nibi branch does.
        remote_scratch="/scratch/t/tvergara"
        ;;
    nibi)
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        if ! bash "$script_dir/check-cluster-ssh.sh" "$cluster"; then
            echo "Aborting: SSH ControlMaster to $cluster is closed. From Mila, run 'ssh $cluster true' and approve the 2FA prompt on your phone, then re-run this script." >&2
            exit 1
        fi
        rsync_host="$cluster"
        remote_scratch="$(ssh "$cluster" 'echo $SCRATCH')"
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
