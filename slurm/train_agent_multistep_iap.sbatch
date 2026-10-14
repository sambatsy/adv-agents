#!/bin/bash
#SBATCH --job-name=adv-multistep-iap
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-multistep-iap-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
export WANDB_DIR="$ROOT"
mkdir -p logs "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# MULTI-STEP (paper's design): hill-climbing, up to 1000 queries/episode, reward
# = 20*(loss improvement over best-so-far), clamped >=0. Perturbation = iap_segment
# against the real 97% vit_b_16 victim. The IAP std-pass is cached per image, so in
# multi-step it runs once per EPISODE (not per step) -- far cheaper than one-step.
# training_steps is a ceiling; watch reward/attack-success in wandb and scancel once clear.
/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python -u train_agent.py \
  --victim_model vit_b_16 \
  --dataset cifar10 \
  --perturbation iap_segment \
  --delta_grid_size 16 \
  --target_kl 0.03 \
  --n_envs 4 \
  --training_steps 200000 \
  --use_wandb \
  --wandb_project adversarial-agents \
  --label vit_b16_multistep_iap
