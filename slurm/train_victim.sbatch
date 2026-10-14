#!/bin/bash
#SBATCH --job-name=adv-victim
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-victim-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
mkdir -p logs victim_models "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python -u train_victim_model.py \
  --dataset cifar10 \
  --model vit_b_16 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.0001 \
  --num_workers 16 \
  --pretrained \
  --amp
