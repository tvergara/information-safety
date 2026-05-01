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
    git pull
else
    git clone https://github.com/tvergara/information-safety.git "$HOME/information-safety"
    cd "$HOME/information-safety"
fi

git submodule update --init --recursive

uv sync
uv run python -c 'import information_safety'
EOF
