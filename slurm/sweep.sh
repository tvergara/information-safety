#!/bin/bash
# POLICY: For full-run experiments, submit this from Tamia or Nibi
# (see CLAUDE.md "Cluster Policy" and slurm/bootstrap-remote-cluster.sh).
# On Mila only run debug-sized invocations (debug=true / max_examples=2).

TRAIN_DATA=/network/scratch/b/brownet/information-safety/data/advbench-completions-smollm3.jsonl
MODELS_DIR=/network/scratch/b/brownet/information-safety/models

declare -A MODEL_PATHS
MODEL_PATHS[safety-pair-safe]=${MODELS_DIR}/safety-pair-safe
MODEL_PATHS[safety-pair-unsafe]=${MODELS_DIR}/safety-pair-unsafe
MODEL_PATHS[llama2]=meta-llama/Llama-2-7b-chat-hf
MODEL_PATHS[llama3]=meta-llama/Meta-Llama-3-8B-Instruct
MODEL_PATHS[olmo3]=allenai/Olmo-3-7B-Instruct
MODEL_PATHS[qwen3-4b]=Qwen/Qwen3-4B

TRUST_REMOTE_MODELS="safety-pair-safe safety-pair-unsafe olmo3"

for MODEL_KEY in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$MODEL_KEY]}

    if [[ " $TRUST_REMOTE_MODELS " == *" $MODEL_KEY "* ]]; then
        TRUST_REMOTE_VALUE=true
    else
        TRUST_REMOTE_VALUE=false
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
    experiment=prompt-attack \
    algorithm/dataset_handler=advbench_harmbench \
    algorithm/strategy=baseline \
    algorithm.model.pretrained_model_name_or_path=${MODEL_PATH} \
    algorithm.model.trust_remote_code=${TRUST_REMOTE_VALUE} \
    trainer.precision=bf16-mixed \
    name=baseline-${MODEL_KEY}
SBATCH
    echo "Submitted baseline-${MODEL_KEY}"

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
    experiment=prompt-attack \
    algorithm/dataset_handler=advbench_harmbench \
    algorithm/strategy=roleplay \
    algorithm.model.pretrained_model_name_or_path=${MODEL_PATH} \
    algorithm.model.trust_remote_code=${TRUST_REMOTE_VALUE} \
    trainer.precision=bf16-mixed \
    name=roleplay-${MODEL_KEY}
SBATCH
    echo "Submitted roleplay-${MODEL_KEY}"

    for MAX_EX in 10 50 100 250 null; do
        JOB_SUFFIX=${MAX_EX}

        sbatch \
            --job-name="sweep-${MODEL_KEY}-${JOB_SUFFIX}" \
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
    algorithm.model.pretrained_model_name_or_path=${MODEL_PATH} \
    algorithm.model.trust_remote_code=${TRUST_REMOTE_VALUE} \
    algorithm.dataset_handler.train_data_path=${TRAIN_DATA} \
    algorithm.dataset_handler.max_examples=${MAX_EX} \
    trainer.precision=bf16-mixed \
    name=sweep-${MODEL_KEY}-${JOB_SUFFIX}
SBATCH
        echo "Submitted sweep-${MODEL_KEY}-${JOB_SUFFIX}"
    done
done

# ------------------------------------------------------------------------------
# GCG smoke run (llama2 only, not part of the full matrix yet).
#
# Prereq: run the AdversariaLLM launcher first to produce the flat suffix JSONL,
# then plug its path below as GCG_SUFFIX_FILE and uncomment the sbatch block.
#
#   ATTACK_KEY=gcg \
#   MODEL_KEY=meta-llama/Llama-2-7b-chat-hf \
#   HYDRA_ATTACK_OVERRIDES="attacks.gcg.num_steps=250 attacks.gcg.search_width=256" \
#   sbatch slurm/attack-adversariallm.sh
# ------------------------------------------------------------------------------
# GCG_SUFFIX_FILE=/network/scratch/b/brownet/information-safety/attacks/gcg-llama2-<job>.jsonl
# sbatch \
#     --job-name="gcg-llama2" \
#     --partition=long \
#     --gres=gpu:a100l:1 \
#     --cpus-per-task=4 \
#     --mem=48G \
#     --time=2:00:00 \
#     --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out \
#     <<SBATCH
# #!/bin/bash
# source /etc/profile.d/modules.sh 2>/dev/null
# module load cuda/12.4.1
# cd /home/mila/b/brownet/information-safety
# source .venv/bin/activate
# export PROJECT_ROOT=/home/mila/b/brownet/information-safety
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HYDRA_FULL_ERROR=1
# python information_safety/main.py \
#     experiment=prompt-attack \
#     algorithm/dataset_handler=advbench_harmbench \
#     algorithm/strategy=precomputed_gcg \
#     algorithm.strategy.suffix_file=${GCG_SUFFIX_FILE} \
#     algorithm.model.pretrained_model_name_or_path=${MODEL_PATHS[llama2]} \
#     algorithm.model.trust_remote_code=false \
#     trainer.precision=bf16-mixed \
#     name=gcg-llama2
# SBATCH
# echo "Submitted gcg-llama2"

# ------------------------------------------------------------------------------
# AutoDAN smoke run (same plumbing as GCG; different prompt strategy).
#
# Prereq: run the AdversariaLLM launcher with ATTACK_KEY=autodan first to produce
# the flat suffix JSONL, then plug its path below as AUTODAN_SUFFIX_FILE and
# uncomment the sbatch block.
#
#   ATTACK_KEY=autodan \
#   MODEL_KEY=meta-llama/Meta-Llama-3.1-8B-Instruct \
#   HYDRA_ATTACK_OVERRIDES="attacks.autodan.num_steps=100" \
#   sbatch slurm/attack-adversariallm.sh
# ------------------------------------------------------------------------------
# AUTODAN_SUFFIX_FILE=/network/scratch/b/brownet/information-safety/attacks/autodan-llama3-<job>.jsonl
# sbatch \
#     --job-name="autodan-llama3" \
#     --partition=long \
#     --gres=gpu:a100l:1 \
#     --cpus-per-task=4 \
#     --mem=48G \
#     --time=2:00:00 \
#     --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out \
#     <<SBATCH
# #!/bin/bash
# source /etc/profile.d/modules.sh 2>/dev/null
# module load cuda/12.4.1
# cd /home/mila/b/brownet/information-safety
# source .venv/bin/activate
# export PROJECT_ROOT=/home/mila/b/brownet/information-safety
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HYDRA_FULL_ERROR=1
# python information_safety/main.py \
#     experiment=prompt-attack \
#     algorithm/dataset_handler=advbench_harmbench \
#     algorithm/strategy=precomputed_autodan \
#     algorithm.strategy.suffix_file=${AUTODAN_SUFFIX_FILE} \
#     algorithm.model.pretrained_model_name_or_path=${MODEL_PATHS[llama3]} \
#     algorithm.model.trust_remote_code=false \
#     trainer.precision=bf16-mixed \
#     name=autodan-llama3
# SBATCH
# echo "Submitted autodan-llama3"
