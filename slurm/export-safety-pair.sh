#!/bin/bash
#SBATCH --job-name=export-model
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

# Export safe model
python scripts/export_checkpoint_to_hf.py \
    --checkpoint /network/scratch/b/brownet/information-safety/checkpoints/safety-pair-safe/last.ckpt \
    --output-dir /network/scratch/b/brownet/information-safety/models/safety-pair-safe

# Export unsafe model
python scripts/export_checkpoint_to_hf.py \
    --checkpoint /network/scratch/b/brownet/information-safety/checkpoints/safety-pair-unsafe/last-v1.ckpt \
    --output-dir /network/scratch/b/brownet/information-safety/models/safety-pair-unsafe
