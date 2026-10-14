#!/bin/bash
#SBATCH --job-name=adv-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-eval-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
mkdir -p logs evaluation_results "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PYTHON=/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Evaluating agent: smoke_test on resnet50 victim ==="
$PYTHON -u evaluate_agent.py \
  --dataset cifar10 \
  --victim_model resnet50 \
  --rl_attack_type RLMaxLoss \
  --epsilon 0.3 \
  --N 5 \
  --theta 0.05 \
  --agent_file ./agents/smoke_test.pth

echo ""
echo "=== Evaluation results (first 30 lines): ==="
head -30 evaluation_results/episodic.csv

echo ""
echo "=== Total episodes: ==="
wc -l evaluation_results/episodic.csv

echo ""
echo "=== Attack success rate: ==="
$PYTHON -c "
import csv
with open('evaluation_results/episodic.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
total = len(rows)
successes = sum(1 for r in rows if r['label'] != r['prediction'])
print(f'Total: {total}, Successes: {successes}, Rate: {successes/total:.4f}' if total else 'No episodes')
"

echo "=== Done ==="
