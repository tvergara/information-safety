#!/bin/bash
#SBATCH --job-name=eval-mmlu-bench
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1/cudnn/9.3 2>/dev/null || true
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

OUT=/network/scratch/b/brownet/information-safety/mmlu-bench/base.json

time python -u scripts/evaluate_mmlu.py \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --output-path "${OUT}" \
    --max-examples 50
