"""Item 5: PPO health for both arms under target_kl=0.03.

Mirrors SB3 PPO's update math faithfully (same normalize-advantage, same
clipped surrogate, same approx_kl = ((ratio-1) - logratio).mean(), same
max_grad_norm, same target_kl early-stop rule: break when approx_kl >
1.5*target_kl). Records per update: epochs completed before stop, final
approx_kl, explained_variance, and the rollout zero-reward fraction.
"""

import os
import sys
import json
import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import EvasionEnv
from ppo_agent import PPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_KL = 0.03
N_UPDATES = 8

victim = torchvision.models.resnet50(weights=None)
victim.fc = torch.nn.Linear(victim.fc.in_features, 10)
victim.to(device)
sd = torch.load("./victim_models/cifar10+resnet50.pth", map_location=device)
victim.load_state_dict(sd["model_state_dict"])
victim.eval()

raw = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
ds = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=raw)
ds = [(p[0], p[1]) for p in ds]
D = ds[: len(ds) // 2]
cfg = {"type": "RLMaxLoss", "epsilon": 0.3, "c": 0.01}


def run_arm(pert):
    torch.manual_seed(0); np.random.seed(0)
    cache = {}

    def mk():
        def _init():
            return Monitor(EvasionEnv(
                victim, D, 5, 0.05, cfg, perturbation=pert,
                epsilon_linf=8.0 / 255.0, delta_grid_size=16, sensitivity_cache=cache))
        return _init

    env = DummyVecEnv([mk() for _ in range(4)])
    agent = PPOAgent(
        feature_extractor="efficientnet_b0", policy_head="default", env=env,
        learning_rate=lambda x: x * 0.0025, n_steps=128, batch_size=64,
        n_epochs=10, gamma=0.99, clip_range=lambda x: x * 0.1, target_kl=TARGET_KL,
    )
    model = agent.model
    records = []

    def instrumented_train():
        rb = model.rollout_buffer
        rewards = np.asarray(rb.rewards).ravel()
        zero_frac = float((rewards == 0).mean())
        values_all = np.asarray(rb.values).ravel()
        returns_all = np.asarray(rb.returns).ravel()
        var_ret = returns_all.var()
        explained_var = float("nan") if var_ret == 0 else float(
            1 - (returns_all - values_all).var() / var_ret)

        clip_range = model.clip_range(model._current_progress_remaining)
        epochs_done = 0
        last_kl = float("nan")
        stop = False
        for epoch in range(model.n_epochs):
            kls = []
            for rd in model.rollout_buffer.get(model.batch_size):
                values, log_prob, entropy = model.policy.evaluate_actions(
                    rd.observations, rd.actions)
                values = values.flatten()
                adv = rd.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                logratio = log_prob - rd.old_log_prob
                ratio = torch.exp(logratio)
                p1 = adv * ratio
                p2 = adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(p1, p2).mean()
                value_loss = torch.nn.functional.mse_loss(rd.returns, values)
                loss = policy_loss + 0.5 * value_loss
                with torch.no_grad():
                    approx_kl = torch.mean((torch.exp(logratio) - 1) - logratio).item()
                kls.append(approx_kl)
                model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()
                if model.target_kl is not None and approx_kl > 1.5 * model.target_kl:
                    stop = True
                    break
            epochs_done += 1
            last_kl = float(np.mean(kls)) if kls else float("nan")
            if stop:
                break
        nan_w = any(torch.isnan(p).any().item() for p in model.policy.parameters())
        records.append({
            "update": len(records) + 1,
            "epochs_before_stop": epochs_done,
            "approx_kl": last_kl,
            "explained_variance": explained_var,
            "zero_reward_fraction": zero_frac,
            "nan_weights": nan_w,
        })
        print(f"[{pert}] update {len(records)}: epochs={epochs_done}/10 "
              f"approx_kl={last_kl:.4f} expl_var={explained_var:.3f} "
              f"zero_frac={zero_frac:.3f} nan={nan_w}")

    model.train = instrumented_train
    model.learn(total_timesteps=128 * 4 * N_UPDATES, reset_num_timesteps=False)
    return records


out = {}
for pert in ("paper", "iap_segment"):
    out[pert] = run_arm(pert)

with open("figures/item5_ppo_health.json", "w") as f:
    json.dump(out, f, indent=2)
print("=== item5 done ===")
