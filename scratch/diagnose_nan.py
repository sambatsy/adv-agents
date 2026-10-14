"""Reproduce the iap_segment NaN with instrumentation to find its exact source.

Logs, for the real iap_segment rollout (resnet50 victim, 4 envs, one update):
  - rollout reward stats + fraction of exactly-zero rewards (hill-climbing rejects)
  - advantage/return stats after GAE
  - per-minibatch: ratio max, policy/value loss, and global grad norm
  - first tensor to go NaN (torch anomaly detection)
"""

import os
import sys
import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import EvasionEnv
from ppo_agent import PPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.monitor import Monitor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Victim: the resnet50 checkpoint from the earlier smoke run
victim = torchvision.models.resnet50(weights=None)
victim.fc = torch.nn.Linear(victim.fc.in_features, 10)
victim.to(device)
sd = torch.load("./victim_models/cifar10+resnet50.pth", map_location=device)
victim.load_state_dict(sd["model_state_dict"])
victim.eval()

# Dataset D (training half)
raw = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
ds = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=raw)
ds = [(p[0], p[1]) for p in ds]
D = ds[: len(ds) // 2]

cfg = {"type": "RLMaxLoss", "epsilon": 0.3, "c": 0.01}
cache = {}


def make_env():
    def _init():
        env = EvasionEnv(
            victim, D, 5, 0.05, cfg,
            perturbation="iap_segment", epsilon_linf=8.0 / 255.0,
            delta_grid_size=16, sensitivity_cache=cache,
        )
        return Monitor(env)
    return _init


env = DummyVecEnv([make_env() for _ in range(4)])
# Check obs/action/reward for NaN/Inf at the vec-env boundary during rollout.
# If this raises, the NaN enters through the env (IAP render/observation/reward);
# if rollout is clean and the crash is still inside train(), it is purely a
# PPO-optimization blowup, not an env-origin NaN.
env = VecCheckNan(env, raise_exception=True, check_inf=True)

agent = PPOAgent(
    feature_extractor="efficientnet_b0", policy_head="default", env=env,
    learning_rate=lambda x: x * 0.0025, n_steps=128, batch_size=64,
    n_epochs=10, gamma=0.99, clip_range=lambda x: x * 0.1,
)
model = agent.model

# ---- Wrap the REAL train() to log, per PPO epoch/minibatch, the ratio and loss,
# and the log_std, and to catch the first minibatch where ratio explodes. ----
orig_train = model.train


def instrumented_train():
    rb = model.rollout_buffer
    rewards = np.asarray(rb.rewards).ravel()
    print(f"[rollout] reward frac_zero={(rewards == 0).mean():.3f} "
          f"max={rewards.max():.4f}; log_std[min/max]="
          f"{model.policy.log_std.min().item():.3f}/{model.policy.log_std.max().item():.3f}")

    clip_range = model.clip_range(model._current_progress_remaining)
    for epoch in range(model.n_epochs):
        for i, rd in enumerate(model.rollout_buffer.get(model.batch_size)):
            values, log_prob, entropy = model.policy.evaluate_actions(
                rd.observations, rd.actions
            )
            if torch.isnan(log_prob).any():
                print(f"!!! epoch {epoch} mb {i}: log_prob is NaN -> policy "
                      f"weights already NaN. log_std[max]="
                      f"{model.policy.log_std.max().item():.3f}")
                print("=== diagnosis complete ===")
                return
            adv = rd.advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(log_prob - rd.old_log_prob)
            pl1 = adv * ratio
            pl2 = adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(pl1, pl2).mean()
            value_loss = torch.nn.functional.mse_loss(rd.returns, values.flatten())
            ent_loss = -torch.mean(entropy) if entropy is not None else -(-log_prob.mean())
            loss = policy_loss + 0.0 * ent_loss + 0.5 * value_loss

            model.policy.optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.policy.parameters(), model.max_grad_norm
            )
            model.policy.optimizer.step()

            if i == 0 or torch.isinf(ratio).any() or not torch.isfinite(gnorm):
                print(f"[e{epoch} mb{i}] ratio[min/max]={ratio.min():.3e}/"
                      f"{ratio.max():.3e} ploss={policy_loss.item():.3e} "
                      f"vloss={value_loss.item():.3e} gnorm={float(gnorm):.3e} "
                      f"logstd_max={model.policy.log_std.max().item():.3f}")
            nan_w = any(torch.isnan(p).any().item() for p in model.policy.parameters())
            if nan_w or torch.isinf(ratio).any():
                print(f"!!! epoch {epoch} mb {i}: weights_nan={nan_w} "
                      f"ratio_inf={torch.isinf(ratio).any().item()} "
                      f"ratio_max={ratio.max().item():.3e} gnorm={float(gnorm):.3e}")
                print("=== diagnosis complete ===")
                return
    print("=== diagnosis complete (no NaN in this update) ===")


model.train = instrumented_train
torch.autograd.set_detect_anomaly(False)
model.learn(total_timesteps=1536, reset_num_timesteps=False)
