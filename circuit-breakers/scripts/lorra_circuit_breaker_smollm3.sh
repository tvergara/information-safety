#!/bin/bash
#SBATCH --job-name=lorra-smollm3
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety/circuit-breakers
source ../.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

python src/lorra_smollm3.py
