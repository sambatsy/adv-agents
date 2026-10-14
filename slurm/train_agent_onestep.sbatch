#!/bin/bash
#SBATCH --job-name=adv-train-1step
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-train-1step-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
# Keep wandb's run dir in the project (not node-local /tmp).
export WANDB_DIR="$ROOT"
mkdir -p logs "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# One-step attack: image -> policy -> one complete (iap_segment) perturbation ->
# model score, with a classifier-based (CW margin) reward. training_steps is a
# ceiling (each step is now a full episode, so episodes accumulate fast) -- we
# watch the reward/attack-success trend in wandb and scancel once it is clear.
/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python -u train_agent.py \
  --victim_model vit_b_16 \
  --dataset cifar10 \
  --perturbation iap_segment \
  --one_step \
  --delta_grid_size 16 \
  --target_kl 0.03 \
  --n_envs 4 \
  --training_steps 100000 \
  --use_wandb \
  --wandb_project adversarial-agents \
  --label vit_b16_onestep_signal
