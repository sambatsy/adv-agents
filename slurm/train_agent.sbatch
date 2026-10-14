#!/bin/bash
#SBATCH --job-name=adv-train
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-train-%j.log

set -euo pipefail

ROOT=/ibex/user/shehrie/adversarial-agents
cd "$ROOT"

export TORCH_HOME=/ibex/user/shehrie/torch_cache
export HF_HOME=/ibex/user/shehrie/hf_cache
export TRITON_CACHE_DIR=/ibex/user/shehrie/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/ibex/user/shehrie/.inductor_cache
# Keep wandb's run dir in the project (not node-local /tmp, which SLURM nodes
# don't share).
export WANDB_DIR="$ROOT"
mkdir -p logs "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Original paper attack (multi-step, sparse-pixel "paper" perturbation,
# RLMaxLoss + hill-climbing). Small scale on purpose: few envs so PPO updates
# (and the reward trend) appear quickly, and training_steps is just a ceiling --
# we watch the reward trend in wandb and scancel this job once it is clear.
/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python -u train_agent.py \
  --victim_model vit_b_16 \
  --dataset cifar10 \
  --perturbation paper \
  --rl_attack_type RLMaxLoss \
  --n_envs 4 \
  --training_steps 100000 \
  --use_wandb \
  --wandb_project adversarial-agents \
  --label vit_b16_paper_signal
