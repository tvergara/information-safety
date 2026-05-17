#!/bin/bash
# Nibi single-GPU launcher: run one queue spec on one H100.
#
# Submitted by scripts/migrate_pool_to_nibi.py as a separate sbatch per spec.
# Unlike Tamia (whole-node pool), Nibi's gpubase_bygpu_b2 partition takes
# single-GPU requests directly, so we skip the queue-worker indirection.
#
# Usage (on Nibi): sbatch run-nibi-single.sh python information_safety/main.py <args...>
#
#SBATCH --account=def-sreddy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/tvergara/information-safety/slurm-logs/nibi-single-%j.out

set -euo pipefail

REPO_ROOT="${HOME}/information-safety"
cd "${REPO_ROOT}"

# Lustre HOME silently truncates working-tree files; restore zero-byte
# tracked YAMLs before launching so Hydra does not fail config composition.
# trainer/callbacks/none.yaml is intentionally empty (Hydra null-group sentinel).
empty_configs=$(find information_safety/configs -name '*.yaml' -size 0 \
    ! -path 'information_safety/configs/trainer/callbacks/none.yaml')
if [ -n "${empty_configs}" ]; then
    echo "WARNING: zero-byte config files detected; restoring from git:" >&2
    echo "${empty_configs}" >&2
    chmod -R u+w information_safety/configs
    find information_safety/configs -name '*.yaml' -size 0 \
        ! -path 'information_safety/configs/trainer/callbacks/none.yaml' -print0 \
        | xargs -0 -r git checkout HEAD --
    chmod -R a-w information_safety/configs
fi

if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
module load cuda/12.4.1 2>/dev/null || true

# shellcheck disable=SC1091
source .venv/bin/activate

export PROJECT_ROOT="${REPO_ROOT}"
export RESULTS_FILE="/scratch/tvergara/information-safety/results/final-results.jsonl"
export GENERATIONS_DIR="/scratch/tvergara/information-safety/generations"

exec "$@"
