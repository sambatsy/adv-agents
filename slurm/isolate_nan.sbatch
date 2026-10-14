#!/bin/bash
#SBATCH --job-name=adv-nan-iso
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/ibex/user/shehrie/adversarial-agents/logs/adv-nan-iso-%j.log

set -euo pipefail
cd /ibex/user/shehrie/adversarial-agents
export TORCH_HOME=/ibex/user/shehrie/torch_cache
/ibex/user/shehrie/conda/envs/adversarial-agents/bin/python -u scratch/isolate_nan.py
echo "=== isolation done ==="
