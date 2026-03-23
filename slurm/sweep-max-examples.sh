#!/bin/bash
# Sweep over max_examples for AdvBench/HarmBench training data ablation.
# Each job trains 1 epoch, generates completions, and writes a result row.

for MAX_EX in 10 50 100 250; do
    sbatch \
        --job-name="harmbench-$MAX_EX" \
        --partition=long \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=16G \
        --time=1:00:00 \
        --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out \
        <<SBATCH
#!/bin/bash
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate
python information_safety/main.py \
    experiment=finetune-with-strategy \
    algorithm/dataset_handler=advbench_harmbench \
    algorithm/strategy=data \
    algorithm.dataset_handler.max_examples=$MAX_EX \
    name=sweep-max-examples-$MAX_EX
SBATCH

    echo "Submitted harmbench-$MAX_EX (max_examples=$MAX_EX)"
done

# Full dataset (no max_examples override needed, default is null)
sbatch \
    --job-name="harmbench-all" \
    --partition=long \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=1:00:00 \
    --output=/network/scratch/b/brownet/information-safety/slurm-logs/slurm-%j.out \
    <<'SBATCH'
#!/bin/bash
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.4.1
cd /home/mila/b/brownet/information-safety
source .venv/bin/activate
python information_safety/main.py \
    experiment=finetune-with-strategy \
    algorithm/dataset_handler=advbench_harmbench \
    algorithm/strategy=data \
    name=sweep-max-examples-all
SBATCH

echo "Submitted harmbench-all (max_examples=null)"
