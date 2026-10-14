#!/bin/bash
#SBATCH --job-name=adv-iap-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-iap-smoke-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
mkdir -p logs learning_results agents "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PYTHON=/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Unit tests ==="
$PYTHON test_iap_perturbation.py

echo ""
echo "=== Smoke test: --perturbation iap_segment (G=16), resnet50 victim ==="
$PYTHON -u train_agent.py \
  --dataset cifar10 \
  --victim_model resnet50 \
  --rl_attack_type RLMaxLoss \
  --epsilon 0.3 \
  --N 5 \
  --theta 0.05 \
  --perturbation iap_segment \
  --epsilon_linf 0.0313725 \
  --delta_grid_size 16 \
  --training_steps 2048 \
  --n_envs 4 \
  --label smoke_iap

echo ""
echo "=== Episodic results (iap_segment): ==="
head -15 learning_results/episodic.csv
wc -l learning_results/episodic.csv

echo ""
echo "=== Sanity: --perturbation paper still runs (default path unchanged) ==="
$PYTHON -u train_agent.py \
  --dataset cifar10 \
  --victim_model resnet50 \
  --rl_attack_type RLMaxLoss \
  --epsilon 0.3 \
  --N 5 \
  --theta 0.05 \
  --perturbation paper \
  --training_steps 1024 \
  --n_envs 4 \
  --label smoke_paper

echo ""
echo "=== Episodic results (paper): ==="
head -10 learning_results/episodic.csv
wc -l learning_results/episodic.csv

echo "=== Done ==="
