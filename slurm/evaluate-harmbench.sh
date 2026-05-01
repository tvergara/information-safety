#!/bin/bash
# POLICY: For full-run experiments, submit this from Tamia or Nibi
# (see CLAUDE.md "Cluster Policy" and slurm/bootstrap-remote-cluster.sh).
# On Mila only run debug-sized invocations (debug=true / max_examples=2).
#SBATCH --job-name=eval-harmbench
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

GENERATIONS_DIR="${1:-/network/scratch/b/brownet/information-safety/generations/epoch-0}"

python scripts/evaluate_harmbench.py --generations-dir "$GENERATIONS_DIR"
