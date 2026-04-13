#!/bin/bash

declare -A MODELS
MODELS[safety-pair-safe]=/network/scratch/b/brownet/information-safety/models/safety-pair-safe
MODELS[safety-pair-unsafe]=/network/scratch/b/brownet/information-safety/models/safety-pair-unsafe
MODELS[llama3-rr]=GraySwanAI/Llama-3-8B-Instruct-RR

for MODEL_KEY in safety-pair-safe safety-pair-unsafe llama3-rr; do
    MODEL_PATH=${MODELS[$MODEL_KEY]}

    TRUST_REMOTE=""
    if [[ "$MODEL_KEY" == "safety-pair-safe" || "$MODEL_KEY" == "safety-pair-unsafe" ]]; then
        TRUST_REMOTE="algorithm.model.trust_remote_code=true"
    fi

    sbatch \
        --job-name="baseline-${MODEL_KEY}" \
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
    algorithm/strategy=baseline \
    algorithm.model.pretrained_model_name_or_path=${MODEL_PATH} \
    ${TRUST_REMOTE} \
    trainer.precision=bf16-mixed \
    name=baseline-${MODEL_KEY}
SBATCH

    echo "Submitted baseline-${MODEL_KEY}"
done
