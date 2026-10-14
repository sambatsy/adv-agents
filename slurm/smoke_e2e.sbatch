#!/bin/bash
#SBATCH --job-name=adv-e2e-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-e2e-smoke-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
mkdir -p logs learning_results agents victim_models "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PYTHON=/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Step 1: Train the victim model (short run) ----
echo ""
echo "=== Step 1: Training victim model (resnet50, 10 epochs) ==="
$PYTHON -u train_victim_model.py \
  --dataset cifar10 \
  --model resnet50 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.001 \
  --num_workers 4

echo ""
echo "=== Step 1 done. Victim checkpoint: ==="
ls -lh victim_models/cifar10+resnet50.pth

# ---- Step 2: Train the adversarial agent (short run) ----
echo ""
echo "=== Step 2: Training adversarial agent (4096 steps, 4 envs) ==="
$PYTHON -u train_agent.py \
  --dataset cifar10 \
  --victim_model resnet50 \
  --rl_attack_type RLMaxLoss \
  --epsilon 0.3 \
  --N 5 \
  --theta 0.05 \
  --training_steps 4096 \
  --n_envs 4 \
  --label smoke_test

echo ""
echo "=== Step 2 done. Episodic results: ==="
head -30 learning_results/episodic.csv
wc -l learning_results/episodic.csv

echo ""
echo "=== All done ==="
