#!/bin/bash
# Rsync defenses, attacks, and train data from Mila SCRATCH to trc /work.
# Idempotent. Run before submit_trc_jobs.py so the containers can find
# the precomputed-attack suffix files and defense weights.
#
# Usage: bash slurm/sync-defenses-to-trc.sh [--dry-run]

set -euo pipefail

dry_run=""
if [ "${1:-}" = "--dry-run" ]; then
    dry_run="1"
fi

mila_base="${SCRATCH:-/network/scratch/b/brownet}/information-safety"
trc_base="/work/information-safety-results"

mila_defenses="${mila_base}/defenses/"
mila_attacks="${mila_base}/attacks/"
mila_data="${mila_base}/data/"

trc_defenses="trc:${trc_base}/defenses/"
trc_attacks="trc:${trc_base}/attacks/"
trc_data="trc:${trc_base}/data/"

sync_one() {
    local src="$1"
    local dst="$2"
    if [ -n "${dry_run}" ]; then
        echo "rsync -avz '${src}' '${dst}'"
    else
        rsync -avz "${src}" "${dst}"
    fi
}

if [ -z "${dry_run}" ]; then
    ssh trc "mkdir -p ${trc_base}/defenses ${trc_base}/attacks ${trc_base}/data ${trc_base}/results ${trc_base}/generations" >/dev/null
fi

sync_one "${mila_defenses}" "${trc_defenses}"
sync_one "${mila_attacks}" "${trc_attacks}"
sync_one "${mila_data}" "${trc_data}"

echo ""
if [ -n "${dry_run}" ]; then
    echo "Done (dry-run — no files copied)."
else
    echo "Done."
fi
