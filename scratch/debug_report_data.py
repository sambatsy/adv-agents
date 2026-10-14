"""Diagnostics for the paper-vs-iap_segment debug report (items 1-4).

Drives both perturbation arms with a RANDOM policy, manually replicating the
hill-climbing accept/reject so every proposal's margin is visible. Does NOT
modify environment.py. Saves figures to figures/ and stats to
figures/debug_stats.json.
"""

import os
import sys
import json
import numpy as np
import torch
import torchvision

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import EvasionEnv
from iap_segment import iap_sensitivity_map

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
EPS_LINF = 8.0 / 255.0
G = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)
torch.manual_seed(0)

# ---- Victim + data ----
victim = torchvision.models.resnet50(weights=None)
victim.fc = torch.nn.Linear(victim.fc.in_features, 10)
victim.to(device)
sd = torch.load("./victim_models/cifar10+resnet50.pth", map_location=device)
victim.load_state_dict(sd["model_state_dict"])
victim.eval()

raw = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
ds = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=raw)
ds = [(p[0], p[1]) for p in ds]
Dprime = ds[len(ds) // 2:]  # held-out half

cfg = {"type": "RLMaxLoss", "epsilon": 0.3, "c": 0.01}
cache = {}


def make_env(pert):
    return EvasionEnv(
        victim, Dprime, 5, 0.05, cfg,
        perturbation=pert, epsilon_linf=EPS_LINF, delta_grid_size=G,
        sensitivity_cache=cache,
    )


env_paper = make_env("paper")
env_iap = make_env("iap_segment")


def set_clean(env, img, label, idx):
    """Prime an env on a specific clean image without the random-reset loop."""
    env.sample_point = (img, label)
    env.sample = torch.clone(img)
    h, w = img.shape[-2], img.shape[-1]
    env.full_mask = np.ones((h, w), dtype=np.uint8)
    env.cache_key = int(idx)
    env.cumulative_lp = 0.0
    env.cumulative_linf = 0.0
    env.s_tminus1 = torch.clone(img)


def propose(env, best_sample, action):
    """Render ONE candidate for the given arm from the appropriate base.
    paper: accumulate on top of best_sample. iap: render from clean."""
    if env.perturbation == "paper":
        env.sample = torch.clone(best_sample)
        cand = env.generate_query(action).clone()
    else:
        cand = env.render_iap(action)
    return cand


def score(env, x):
    loss, pred, _ = env.dynamics(x)
    return loss, pred


# ============================================================
# Item 1: perturbation visuals + low-sensitivity-pixel fraction
# ============================================================
idxs = [0, 1, 2, 3]
low_sens_fracs = []
fig, axes = plt.subplots(len(idxs), 6, figsize=(20, 3.3 * len(idxs)))
to_pil = torchvision.transforms.ToPILImage()
for r, idx in enumerate(idxs):
    img, label = Dprime[idx]
    set_clean(env_paper, img, label, idx)
    set_clean(env_iap, img, label, idx)
    a_paper = env_paper.action_space.sample()
    a_iap = env_iap.action_space.sample()
    pert_paper = propose(env_paper, img, a_paper)
    pert_iap = propose(env_iap, img, a_iap)
    d_paper = (pert_paper - img).abs()
    d_iap = (pert_iap - img).abs()
    sens = iap_sensitivity_map(to_pil(img), (img.shape[-2], img.shape[-1]))
    frac_low = float((sens < 0.05 * sens.max()).mean())
    low_sens_fracs.append(frac_low)

    def show(ax, t, title):
        arr = t.detach().cpu().numpy().transpose(1, 2, 0) if torch.is_tensor(t) else t
        ax.imshow(np.clip(arr, 0, 1) if arr.ndim == 3 else arr,
                  cmap=None if (np.ndim(arr) == 3) else "viridis")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    show(axes[r, 0], img, f"clean (idx={idx}, y={label})")
    show(axes[r, 1], pert_paper, "paper perturbed")
    amp = 1.0 / max(float(d_paper.max()), 1e-6)
    show(axes[r, 2], (d_paper * amp).clamp(0, 1), f"paper |delta| x{amp:.0f}")
    show(axes[r, 3], pert_iap, "iap perturbed")
    amp2 = 1.0 / max(float(d_iap.max()), 1e-6)
    show(axes[r, 4], (d_iap * amp2).clamp(0, 1), f"iap |delta| x{amp2:.0f}")
    im = axes[r, 5].imshow(sens, cmap="viridis")
    axes[r, 5].set_title(f"IAP sensitivity\n<5%max: {frac_low*100:.1f}%", fontsize=9)
    axes[r, 5].axis("off")
    fig.colorbar(im, ax=axes[r, 5], fraction=0.046)
plt.tight_layout()
plt.savefig(f"{OUT}/item1_perturbation_visuals.png", dpi=110)
plt.close()
print(f"[item1] low-sensitivity (<5% of max) pixel fraction per image: "
      f"{[round(f, 3) for f in low_sens_fracs]} mean={np.mean(low_sens_fracs):.3f}")

# ============================================================
# Item 2: proposal quality -- 20 images x 50 random actions, margin vs clean
# ============================================================
N_IMG, N_ACT = 20, 50
margins = {"paper": [], "iap_segment": []}
for env, key in ((env_paper, "paper"), (env_iap, "iap_segment")):
    for idx in range(N_IMG):
        img, label = Dprime[idx]
        set_clean(env, img, label, idx)
        f_clean, _ = score(env, img)
        for _ in range(N_ACT):
            a = env.action_space.sample()
            cand = propose(env, img, a)  # from clean (current best at step 0)
            f_cand, _ = score(env, cand)
            margins[key].append(f_cand - f_clean)
accept_pct = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, key in zip(axes, ("paper", "iap_segment")):
    m = np.array(margins[key])
    accept_pct[key] = float((m >= 0).mean() * 100)
    ax.hist(np.clip(m, -0.5, 0.5), bins=60, color="#4c72b0", alpha=0.85)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_title(f"{key}: margin change (f_cand - f_clean)\n"
                 f"would-accept={accept_pct[key]:.1f}%  "
                 f"mean={m.mean():.4f} max={m.max():.4f}", fontsize=10)
    ax.set_xlabel("margin change")
axes[0].set_ylabel("count")
plt.tight_layout()
plt.savefig(f"{OUT}/item2_proposal_margins.png", dpi=110)
plt.close()
print(f"[item2] would-accept %: {accept_pct}")
print(f"[item2] margin mean paper={np.mean(margins['paper']):.5f} "
      f"iap={np.mean(margins['iap_segment']):.5f}")

# ============================================================
# Item 3 + 4: acceptance over time, and L2/L-inf per episode
# ============================================================
N_EP, MAX_STEPS = 15, 60
accept_curves = {"paper": np.zeros(MAX_STEPS), "iap_segment": np.zeros(MAX_STEPS)}
accept_counts = {"paper": np.zeros(MAX_STEPS), "iap_segment": np.zeros(MAX_STEPS)}
dist_l2 = {"paper": [], "iap_segment": []}
dist_linf = {"paper": [], "iap_segment": []}
for env, key in ((env_paper, "paper"), (env_iap, "iap_segment")):
    for ep in range(N_EP):
        img, label = Dprime[ep]
        set_clean(env, img, label, ep)
        best = torch.clone(img)
        best_loss, _ = score(env, best)
        for step in range(MAX_STEPS):
            a = env.action_space.sample()
            cand = propose(env, best, a)
            f_cand, _ = score(env, cand)
            accepted = (f_cand - best_loss) >= 0
            accept_curves[key][step] += 1.0 if accepted else 0.0
            accept_counts[key][step] += 1.0
            if accepted:
                best = cand
                best_loss = f_cand
        delta = best - img
        dist_l2[key].append(float(torch.norm(delta, p=2)))
        dist_linf[key].append(float(torch.norm(delta.flatten(), p=float("inf"))))

# Item 3 figure
plt.figure(figsize=(8, 5))
for key, c in (("paper", "#4c72b0"), ("iap_segment", "#dd8452")):
    rate = accept_curves[key] / np.maximum(accept_counts[key], 1)
    plt.plot(range(MAX_STEPS), rate, label=key, color=c, lw=2)
plt.xlabel("step within episode")
plt.ylabel(f"acceptance rate (over {N_EP} episodes)")
plt.title("Item 3: acceptance rate vs step")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/item3_acceptance_over_time.png", dpi=110)
plt.close()

# Item 4 figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].boxplot([dist_l2["paper"], dist_l2["iap_segment"]], labels=["paper", "iap"])
axes[0].set_title("Item 4: L2 distortion per episode")
axes[0].set_ylabel("L2")
axes[1].boxplot([dist_linf["paper"], dist_linf["iap_segment"]], labels=["paper", "iap"])
axes[1].axhline(EPS_LINF, color="red", ls="--", lw=1, label=f"eps_linf={EPS_LINF:.4f}")
axes[1].set_title("Item 4: L-inf distortion per episode")
axes[1].set_ylabel("L-inf")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/item4_distortion.png", dpi=110)
plt.close()

# Acceptance-collapse summary for item 3
collapse = {}
for key in ("paper", "iap_segment"):
    rate = accept_curves[key] / np.maximum(accept_counts[key], 1)
    collapse[key] = {
        "step0_rate": float(rate[0]),
        "steps1_5_rate": float(rate[1:6].mean()),
        "steps20plus_rate": float(rate[20:].mean()),
    }
print(f"[item3] acceptance collapse: {json.dumps(collapse, indent=2)}")
print(f"[item4] L2 mean paper={np.mean(dist_l2['paper']):.4f} "
      f"iap={np.mean(dist_l2['iap_segment']):.4f}; "
      f"Linf mean paper={np.mean(dist_linf['paper']):.4f} "
      f"iap={np.mean(dist_linf['iap_segment']):.4f}")

stats = {
    "item1_low_sensitivity_fraction": {
        "per_image": low_sens_fracs, "mean": float(np.mean(low_sens_fracs))
    },
    "item2_would_accept_pct": accept_pct,
    "item2_margin_mean": {k: float(np.mean(v)) for k, v in margins.items()},
    "item2_margin_max": {k: float(np.max(v)) for k, v in margins.items()},
    "item3_acceptance_collapse": collapse,
    "item4_distortion": {
        "l2_mean": {k: float(np.mean(v)) for k, v in dist_l2.items()},
        "l2_max": {k: float(np.max(v)) for k, v in dist_l2.items()},
        "linf_mean": {k: float(np.mean(v)) for k, v in dist_linf.items()},
        "linf_max": {k: float(np.max(v)) for k, v in dist_linf.items()},
    },
}
with open(f"{OUT}/debug_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("=== debug_report_data done ===")
