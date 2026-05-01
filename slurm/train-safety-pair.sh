#!/bin/bash
# POLICY: For full-run experiments, submit this from Tamia or Nibi
# (see CLAUDE.md "Cluster Policy" and slurm/bootstrap-remote-cluster.sh).
# On Mila only run debug-sized invocations (debug=true / max_examples=2).
#SBATCH --job-name=safety-pair
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

export PROJECT_ROOT=/home/mila/b/brownet/information-safety
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

INCLUDE_REFUSALS="${1:-false}"

if [ "$INCLUDE_REFUSALS" = "true" ]; then
    EXPERIMENT=safety-pair-safe
    CKPT_DIR=/network/scratch/b/brownet/information-safety/checkpoints/safety-pair-safe/
else
    EXPERIMENT=safety-pair-unsafe
    CKPT_DIR=/network/scratch/b/brownet/information-safety/checkpoints/safety-pair-unsafe/
fi

echo "Running experiment=$EXPERIMENT include_refusals=$INCLUDE_REFUSALS"

python information_safety/main.py \
    experiment=$EXPERIMENT \
    algorithm/model=smollm3 \
    algorithm.dataset_handler.batch_size=2 \
    trainer.precision=bf16-mixed \
    trainer.callbacks.model_checkpoint.dirpath=$CKPT_DIR
