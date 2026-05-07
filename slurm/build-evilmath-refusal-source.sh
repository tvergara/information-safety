#!/bin/bash
# POLICY: One-time generation step on a multi-GPU box.
# Caller passes --output=$SCRATCH/slurm-logs/slurm-%j.out via sbatch CLI.
#SBATCH --job-name=build-evilmath-refusal-source
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2:00:00

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1 2>/dev/null || true

REPO_ROOT="${REPO_ROOT:-${HOME}/information-safety}"
cd "${REPO_ROOT}"
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH}/information-safety/defense-data/evilmath/llm_rewrite.jsonl}"
NUM_EXAMPLES="${NUM_EXAMPLES:-3000}"
SEED="${SEED:-0}"
REWRITER_MODEL="${REWRITER_MODEL:-meta-llama/Llama-3.1-70B-Instruct}"

echo "Generating EvilMath rewrites: ${NUM_EXAMPLES} examples -> ${OUTPUT_PATH}"

python scripts/build_evilmath_refusal_source.py \
    --output-path "$OUTPUT_PATH" \
    --num-examples "$NUM_EXAMPLES" \
    --seed "$SEED" \
    --rewriter-model "$REWRITER_MODEL" \
    "$@"
