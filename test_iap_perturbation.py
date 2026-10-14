"""Unit tests for the iap_segment perturbation backend in EvasionEnv.

Run: python -m pytest test_iap_perturbation.py -v   (or: python test_iap_perturbation.py)
"""

import numpy as np
import torch
from PIL import Image

from iap_segment import apply_iap_segment_perturbation


EPS = 8.0 / 255.0
G = 16


def _clean_image():
    # A deterministic 32x32 RGB image with structure (so sensitivity is non-uniform).
    rng = np.random.default_rng(0)
    arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_output_shape_and_range():
    image = _clean_image()
    mask = np.ones((32, 32), dtype=np.uint8)
    action = torch.randn(3 * G * G)
    edited, actual_delta, upsampled, meta = apply_iap_segment_perturbation(
        image, mask, action, EPS, delta_grid_size=G
    )

    # Output image: same size, RGB, uint8 in [0, 255]
    assert edited.size == (32, 32)
    edited_arr = np.asarray(edited)
    assert edited_arr.shape == (32, 32, 3)
    assert edited_arr.dtype == np.uint8

    # actual_delta is (H, W, C) float in [-eps, eps]
    assert actual_delta.shape == (32, 32, 3)
    assert actual_delta.min() >= -EPS - 1e-6
    assert actual_delta.max() <= EPS + 1e-6


def test_linf_budget_respected():
    image = _clean_image()
    mask = np.ones((32, 32), dtype=np.uint8)
    # Large action values to push against the budget
    action = torch.full((3 * G * G,), 5.0)
    edited, actual_delta, _, _ = apply_iap_segment_perturbation(
        image, mask, action, EPS, delta_grid_size=G
    )
    clean_arr = np.asarray(image, dtype=np.float32) / 255.0
    edited_arr = np.asarray(edited, dtype=np.float32) / 255.0
    linf = np.abs(edited_arr - clean_arr).max()
    assert linf <= EPS + 1e-6, f"L-inf {linf} exceeds budget {EPS}"


def test_zero_action_is_identity():
    image = _clean_image()
    mask = np.ones((32, 32), dtype=np.uint8)
    action = torch.zeros(3 * G * G)
    edited, actual_delta, _, _ = apply_iap_segment_perturbation(
        image, mask, action, EPS, delta_grid_size=G
    )
    assert np.array_equal(np.asarray(edited), np.asarray(image))
    assert np.abs(actual_delta).max() == 0.0


def test_action_dim_mismatch_raises():
    image = _clean_image()
    mask = np.ones((32, 32), dtype=np.uint8)
    action = torch.zeros(10)  # wrong size
    try:
        apply_iap_segment_perturbation(image, mask, action, EPS, delta_grid_size=G)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong action dim")


def test_sensitivity_cache_runs_once():
    image = _clean_image()
    mask = np.ones((32, 32), dtype=np.uint8)
    cache = {}
    action = torch.randn(3 * G * G)
    apply_iap_segment_perturbation(
        image, mask, action, EPS, delta_grid_size=G,
        sensitivity_cache=cache, cache_key=7,
    )
    assert 7 in cache
    weight_first = cache[7].clone()
    # Second call with same key must reuse the cached weight (not recompute)
    apply_iap_segment_perturbation(
        image, mask, action, EPS, delta_grid_size=G,
        sensitivity_cache=cache, cache_key=7,
    )
    assert torch.equal(cache[7], weight_first)


def test_env_iap_step_shape_and_budget():
    # End-to-end through the env on a tiny fake victim (CPU, no GPU needed).
    from environment import EvasionEnv

    class FakeVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3 * 224 * 224, 10)

        def forward(self, x):
            return self.lin(x.flatten(1))

    # Do not override .to()/.eval(); nn.Module's versions correctly move params
    # to the env's device (the env calls victim_model.to(device)).
    victim = FakeVictim()
    # Build a tiny dataset of 32x32 images in [0,1]
    data = [(torch.rand(3, 32, 32), i % 10) for i in range(5)]
    cfg = {"type": "RLMaxLoss", "epsilon": 0.3, "c": 0.01}
    env = EvasionEnv(
        victim, data, N=5, theta=0.05, rl_attack_config=cfg,
        perturbation="iap_segment", epsilon_linf=EPS, delta_grid_size=G,
        sensitivity_cache={},
    )

    # Action space must match 3*G*G
    assert env.action_space.shape == (3 * G * G,)

    env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    # Observation image is the preprocessed 224x224
    assert obs["image"].shape == (3, 224, 224)
    # The best-so-far image stays a valid 32x32 in [0,1]
    assert env.sample.shape == (3, 32, 32)
    assert float(env.sample.min()) >= 0.0 and float(env.sample.max()) <= 1.0
    # L-inf of the kept image vs clean respects the budget
    linf = float(torch.abs(env.sample - env.sample_point[0]).max())
    assert linf <= EPS + 1e-6, f"env L-inf {linf} exceeds {EPS}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
