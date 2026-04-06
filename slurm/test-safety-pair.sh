#!/bin/bash
#SBATCH --job-name=test-safety-pair
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

source /etc/profile.d/modules.sh
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

export PROJECT_ROOT=/home/mila/b/brownet/information-safety
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

python information_safety/main.py \
    experiment=safety-pair-unsafe \
    algorithm/model=smollm3 \
    algorithm.dataset_handler.num_benign_examples=50 \
    algorithm.dataset_handler.batch_size=4 \
    trainer.max_epochs=1 \
    trainer.fast_dev_run=3 \
    trainer.precision=bf16-mixed \
    trainer.callbacks.model_checkpoint.dirpath=/network/scratch/b/brownet/information-safety/checkpoints/test-safety-pair/
