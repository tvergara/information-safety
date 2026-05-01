#!/bin/bash
# Probe SSH ControlMaster sockets for one or more clusters.
#
# Usage: bash slurm/check-cluster-ssh.sh <cluster> [<cluster> ...]
#
# For each cluster, runs a non-interactive ssh; if it fails, prints a
# phone-approval message and exits non-zero at the end.

set -uo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cluster> [<cluster> ...]" >&2
    exit 2
fi

failed=0
for cluster in "$@"; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$cluster" true >/dev/null 2>&1; then
        echo "ControlMaster to $cluster is live."
    else
        echo "ControlMaster to $cluster is closed. From Mila, run 'ssh $cluster true' and approve the 2FA prompt on your phone." >&2
        failed=1
    fi
done

exit "$failed"
