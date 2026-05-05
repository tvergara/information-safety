#!/bin/bash
# Bootstrap (or refresh) the information-safety repo on Tamia or Nibi.
#
# Usage: bash slurm/bootstrap-remote-cluster.sh <tamia|nibi>
#
# Idempotent: on first run clones the repo; on subsequent runs falls back to
# git pull + uv sync. Requires a live SSH ControlMaster — calls
# slurm/check-cluster-ssh.sh first and aborts with a phone-approval message
# if the socket is closed.

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
if ! bash "$script_dir/check-cluster-ssh.sh" "$cluster"; then
    echo "Aborting: SSH ControlMaster to $cluster is closed. From Mila, run 'ssh $cluster true' and approve the 2FA prompt on your phone, then re-run this script." >&2
    exit 1
fi

ssh "$cluster" bash -s <<'EOF'
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ -d "$HOME/information-safety/.git" ]; then
    cd "$HOME/information-safety"
    chmod -R u+w information_safety/configs
    git pull
else
    git clone https://github.com/tvergara/information-safety.git "$HOME/information-safety"
    cd "$HOME/information-safety"
fi

git submodule update --init --recursive

# Tamia/Nibi (Lustre HOME) silently truncates working-tree files after a
# successful git checkout. Flush the pull writes so `find -size 0` sees
# real on-disk state, restore any zero-byte tracked YAMLs from the index,
# then lock configs read-only so the truncation cannot recur.
# trainer/callbacks/none.yaml is intentionally empty (Hydra null-group sentinel).
sync
find information_safety/configs -name '*.yaml' -size 0 \
    ! -path 'information_safety/configs/trainer/callbacks/none.yaml' -print0 \
    | xargs -0 -r git checkout HEAD --
chmod -R a-w information_safety/configs

uv sync
uv run python -c 'import information_safety'
EOF
