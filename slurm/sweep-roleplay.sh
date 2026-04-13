#!/bin/bash

declare -A MODELS
MODELS[safety-pair-safe]=/network/scratch/b/brownet/information-safety/models/safety-pair-safe
MODELS[safety-pair-unsafe]=/network/scratch/b/brownet/information-safety/models/safety-pair-unsafe
MODELS[llama2]=meta-llama/Llama-2-7b-chat-hf
MODELS[llama3]=meta-llama/Meta-Llama-3-8B-Instruct
MODELS[llama3-rr]=GraySwanAI/Llama-3-8B-Instruct-RR
MODELS[olmo3]=allenai/Olmo-3-7B-Instruct

for MODEL_KEY in safety-pair-safe safety-pair-unsafe llama2 llama3 llama3-rr olmo3; do
    MODEL_PATH=${MODELS[$MODEL_KEY]}

    TRUST_REMOTE=""
    if [[ "$MODEL_KEY" == "safety-pair-safe" || "$MODEL_KEY" == "safety-pair-unsafe" || "$MODEL_KEY" == "olmo3" ]]; then
        TRUST_REMOTE="algorithm.model.trust_remote_code=true"
    fi

    sbatch \
        --job-name="roleplay-${MODEL_KEY}" \
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
    algorithm/strategy=roleplay \
    algorithm.model.pretrained_model_name_or_path=${MODEL_PATH} \
    ${TRUST_REMOTE} \
    trainer.precision=bf16-mixed \
    name=roleplay-${MODEL_KEY}
SBATCH

    echo "Submitted roleplay-${MODEL_KEY}"
done
