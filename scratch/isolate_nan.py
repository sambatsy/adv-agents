"""Isolate whether a 768-dim Gaussian action space makes PPO diverge to NaN,
independent of the victim / IAP rendering. Uses a trivial random env."""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo_agent import PPOAgent


class RandomEnv(gym.Env):
    """Dict obs matching EvasionEnv; reward = small function of action.
    No victim, no IAP -- just exercises PPO on the given action dim."""

    def __init__(self, action_dim, low, high):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(-np.inf, np.inf, (3, 224, 224), np.float32),
                "outputs": spaces.Box(0.0, 1.0, (10,), np.float32),
                "source_class": spaces.Discrete(10),
            }
        )
        self.action_space = spaces.Box(low, high, (action_dim,), np.float32)
        self.t = 0

    def _obs(self):
        return {
            "image": np.random.randn(3, 224, 224).astype(np.float32),
            "outputs": np.random.rand(10).astype(np.float32),
            "source_class": 0,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = float(np.tanh(action).mean())  # bounded, benign
        done = self.t >= 20
        return self._obs(), reward, done, False, {}


def run(action_dim, low, high, tag):
    torch.manual_seed(0)
    np.random.seed(0)
    env = DummyVecEnv([lambda: RandomEnv(action_dim, low, high) for _ in range(4)])
    agent = PPOAgent(
        feature_extractor="efficientnet_b0",
        policy_head="default",
        env=env,
        learning_rate=lambda x: x * 0.0025,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=lambda x: x * 0.1,
    )
    try:
        agent.model.learn(total_timesteps=1024, reset_num_timesteps=False)
        has_nan = any(
            torch.isnan(p).any().item() for p in agent.model.policy.parameters()
        )
        print(f"[{tag}] action_dim={action_dim} range=[{low},{high}] -> "
              f"{'NaN in weights' if has_nan else 'OK, no NaN'}")
    except ValueError as e:
        print(f"[{tag}] action_dim={action_dim} range=[{low},{high}] -> DIVERGED: "
              f"{str(e)[:80]}")


if __name__ == "__main__":
    run(25, -2.0, 2.0, "paper")
    run(768, -1.0, 1.0, "iap_G16")
