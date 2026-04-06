#!/bin/bash
# Sweep over max_examples on AdvBench using both safety-pair models with data strategy.
# Measures data bits (arithmetic coding) and evaluates ASR on HarmBench.

MODELS_DIR=/network/scratch/b/brownet/information-safety/models
TRAIN_DATA=/network/scratch/b/brownet/information-safety/data/advbench-completions-smollm3.jsonl

for MODEL_NAME in safety-pair-safe safety-pair-unsafe; do
    for MAX_EX in 10 50 100 250; do
        sbatch \
            --job-name="sweep-${MODEL_NAME}-${MAX_EX}" \
            --partition=long \
            --gres=gpu:a100l:1 \
            --cpus-per-task=4 \
            --mem=48G \
            --time=2:00:00 \
            --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out \
            <<SBATCH
#!/bin/bash
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate
export PROJECT_ROOT=/home/mila/b/brownet/information-safety
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
python information_safety/main.py \
    experiment=finetune-with-strategy \
    algorithm/dataset_handler=advbench_harmbench \
    algorithm/strategy=data \
    algorithm.strategy.r=16 \
    algorithm.strategy.lora_alpha=16 \
    algorithm.model.pretrained_model_name_or_path=${MODELS_DIR}/${MODEL_NAME} \
    algorithm.dataset_handler.train_data_path=${TRAIN_DATA} \
    algorithm.dataset_handler.max_examples=$MAX_EX \
    trainer.precision=bf16-mixed \
    name=sweep-${MODEL_NAME}-${MAX_EX}
SBATCH

        echo "Submitted sweep-${MODEL_NAME}-${MAX_EX}"
    done

    # Full dataset (520 examples)
    sbatch \
        --job-name="sweep-${MODEL_NAME}-all" \
        --partition=long \
        --gres=gpu:a100l:1 \
        --cpus-per-task=4 \
        --mem=48G \
        --time=2:00:00 \
        --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out \
        <<SBATCH
#!/bin/bash
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate
export PROJECT_ROOT=/home/mila/b/brownet/information-safety
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
python information_safety/main.py \
    experiment=finetune-with-strategy \
    algorithm/dataset_handler=advbench_harmbench \
    algorithm/strategy=data \
    algorithm.strategy.r=16 \
    algorithm.strategy.lora_alpha=16 \
    algorithm.model.pretrained_model_name_or_path=${MODELS_DIR}/${MODEL_NAME} \
    algorithm.dataset_handler.train_data_path=${TRAIN_DATA} \
    trainer.precision=bf16-mixed \
    name=sweep-${MODEL_NAME}-all
SBATCH

    echo "Submitted sweep-${MODEL_NAME}-all"
done
