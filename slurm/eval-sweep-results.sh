#!/bin/bash
# POLICY: This is the eval-sweep classifier. It scores existing
# generations and is allowed on Mila (see CLAUDE.md "Cluster Policy").
# Do not run while any eval job is queued/running -- its rewrite
# clobbers concurrent appends to final-results.jsonl.
#SBATCH --job-name=eval-sweep
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

RESULTS_FILE=/network/scratch/b/brownet/information-safety/results/final-results.jsonl
# keep in sync with plots/utils.py:DEFAULT_RESULTS_FILE
CLEAN_RESULTS_FILE=/network/scratch/b/brownet/information-safety/results/final-results-clean.jsonl

python scripts/evaluate_harmbench.py    --results-file "$RESULTS_FILE"
python scripts/evaluate_strongreject.py --results-file "$RESULTS_FILE"
python scripts/evaluate_wmdp.py         --results-file "$RESULTS_FILE"
python scripts/evaluate_evilmath.py     --results-file "$RESULTS_FILE"

python scripts/check_results.py --emit-clean "$CLEAN_RESULTS_FILE"
