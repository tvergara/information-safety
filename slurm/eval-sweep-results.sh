#!/bin/bash
#SBATCH --job-name=eval-sweep
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

python scripts/evaluate_harmbench.py \
    --results-file /network/scratch/b/brownet/information-safety/results/final-results.jsonl
