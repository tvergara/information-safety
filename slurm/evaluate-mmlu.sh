#!/bin/bash
#SBATCH --job-name=eval-mmlu
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1/cudnn/9.3 2>/dev/null || true
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate

DEFENSES_DIR=/network/scratch/b/brownet/information-safety/defenses
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct

run_one() {
    local model="$1"
    local out_dir="$2"
    local out_path="${out_dir}/mmlu_eval.json"
    if [[ -f "${out_path}" ]]; then
        return
    fi
    python scripts/evaluate_mmlu.py \
        --model-name-or-path "${model}" \
        --output-path "${out_path}"
}

run_one "${BASE_MODEL}" "${DEFENSES_DIR}/base-Llama-3.1-8B-Instruct"

for defense in \
    sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0 \
    sft-evilmath-Llama-3.1-8B-Instruct-d650794f965d \
    cb-wmdp-Llama-3.1-8B-Instruct-bfbf3e38793c \
    cb-evilmath-Llama-3.1-8B-Instruct-d7ba262bbc28 \
    tar-wmdp-Llama-3.1-8B-Instruct-73d8c8e83c07 \
    tar-evilmath-Llama-3.1-8B-Instruct-09003ee4e852
do
    run_one "${DEFENSES_DIR}/${defense}" "${DEFENSES_DIR}/${defense}"
done
