#!/bin/bash
# Push defense weights to HF Hub. Runs from a Mila CPU node because the
# login node kills long-running uploads. ~122 GB at ~14 MB/s ≈ 2.5 h.
#SBATCH --job-name=hf-push-defenses
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

python scripts/push_artifacts_to_hf.py --skip-attacks-data
