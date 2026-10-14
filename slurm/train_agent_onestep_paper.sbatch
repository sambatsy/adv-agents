#!/bin/bash
#SBATCH --job-name=adv-train-1step-paper
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-train-1step-paper-%j.log

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

# CONTROL for the iap_segment one-step run: identical config (vit_b_16 victim,
# one-step CW-margin reward, target_kl=0.03, n_envs=4, training_steps), the ONLY
# change being --perturbation paper -- the original codebase sparse-pixel attack,
# which (unlike iap_segment at 32x32) actually uses its distortion budget. Lets us
# see whether the one-step + CW-margin setup learns at all when the perturbation
# can move the image. Watch reward/attack-success in wandb and scancel once clear.
/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python -u train_agent.py \
  --victim_model vit_b_16 \
  --dataset cifar10 \
  --perturbation paper \
  --one_step \
  --target_kl 0.03 \
  --n_envs 4 \
  --training_steps 100000 \
  --use_wandb \
  --wandb_project adversarial-agents \
  --label vit_b16_onestep_paper_control
