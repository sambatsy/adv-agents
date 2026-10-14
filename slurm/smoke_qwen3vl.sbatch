#!/bin/bash
#SBATCH --job-name=adv-qwen3vl-smoke
#SBATCH --account=pi-ghanembs
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-qwen3vl-smoke-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
mkdir -p logs "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python -u smoke_test_qwen3vl.py \
  --dataset cifar10 \
  --n 16
