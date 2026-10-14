import stable_baselines3
import gymnasium as gym
import numpy as np
import torch
import torchvision
import argparse
import os
import random
from collections import deque
from environment import EvasionEnv
from ppo_agent import PPOAgent
from qwen3vl_victim import load_qwen3vl_victim
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure, KVWriter

# Create a parser to parse command line arguments
parser = argparse.ArgumentParser(
    description="Perform a black-box evasion attack with reinforcement learning."
)

# Dataset Argument
parser.add_argument(
    "--dataset", type=str, default="cifar10", help="The dataset to use."
)  # Options: 'cifar10', 'svhn' (Detailed in the 'Experimental Setup' subsection of the paper)

# Environment Arguments
parser.add_argument(
    "--rl_attack_type",
    type=str,
    default="RLMaxLoss",
    help="The type of RL attack to perform.",
)  # Options: 'RLMaxLoss', 'RLMinNorm' (Detailed in the 'Learning Adversarial Policies' section of the paper)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.3,
    help="The epsilon value for the RL Max Loss attack.",
)  # Used only for RLMaxLoss attack (Detailed in the 'RL Max Loss' subsection of the paper)
parser.add_argument(
    "--c", type=float, default=0.01, help="The c value for the RL Min Norm attack."
)  # Used only for RLMinNorm attack (Detailed in the 'RL Min Norm' subsection of the paper)
parser.add_argument(
    "--N", type=int, default=5, help="The number of actions to take in the environment."
)  # The number of pixels to perturb in the RL attack (Detailed in the 'States and Actions' subsection of the paper)
parser.add_argument(
    "--theta", type=float, default=0.05, help="The theta value for the RL attack."
)  # The perturbation threshold for the RL attack (Detailed in the 'States and Actions' subsection of the paper)
parser.add_argument(
    "--victim_model", type=str, default="resnet50", help="The victim model to attack."
)  # Options: 'resnet50', 'vgg16', 'vit_b_16', 'qwen3vl' (the last is a zero-shot VLM classifier)


# Training Arguments
parser.add_argument(
    "--training_steps",
    type=int,
    default=4_915_200,
    help="The number of steps to run the attack for.",
)  # Total number of updates is 1200 (Detailed in the 'Evaluation' section of the paper) * 4096 (number of steps per update) = 4,915,200
parser.add_argument(
    "--n_envs", type=int, default=32, help="The number of environments to use."
)  # Number of parallel environments

# Label (for saving the trained adversarial agent)
parser.add_argument(
    "--label",
    type=str,
    default="agent0",
    help="The label for saving the trained adversarial agent.",
)  # This will be used to save the agent's policy to a file (e.g., agents/agent0.pth)

# Perturbation backend
parser.add_argument(
    "--perturbation",
    type=str,
    default="paper",
    choices=["paper", "iap_segment"],
    help="Perturbation backend: 'paper' (sparse-pixel) or 'iap_segment' (IAP grid).",
)
parser.add_argument(
    "--epsilon_linf",
    type=float,
    default=8.0 / 255.0,
    help="L-inf budget for the iap_segment perturbation.",
)
parser.add_argument(
    "--delta_grid_size",
    type=int,
    default=16,
    help="Low-resolution action grid side G for iap_segment (action dim = 3*G*G).",
)
parser.add_argument(
    "--one_step",
    action="store_true",
    help=(
        "Reduce the attack to a single step: image -> policy -> one complete "
        "perturbation -> model score, with a classifier-based (CW margin) reward "
        "and the episode ending after that one query. Removes the multi-step "
        "credit-assignment problem. Use with --perturbation iap_segment so the "
        "single action is a full-image perturbation."
    ),
)
parser.add_argument(
    "--target_kl",
    type=float,
    default=None,
    help=(
        "PPO target-KL early-stop threshold. Default None reproduces the paper's "
        "exact PPO settings. With a high-dimensional action (e.g. iap_segment G=16 "
        "= 768 dims) the diagonal-Gaussian log-probs are ~-1100, so the importance "
        "ratio exp(new_logp - old_logp) compounds across dims and can overflow to "
        "inf within one update, producing NaN policy weights. Setting e.g. 0.03 "
        "caps the per-update policy drift and prevents the blow-up. Apply the SAME "
        "value to both perturbation arms to keep the comparison fair."
    ),
)

# Wandb Arguments
parser.add_argument(
    "--use_wandb", action="store_true", help="Enable Weights & Biases logging."
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="adversarial-agents",
    help="W&B project name.",
)
parser.add_argument(
    "--wandb_entity", type=str, default=None, help="W&B entity (team or user)."
)

# Parse the arguments
args = parser.parse_args()


# ── Wandb integration ────────────────────────────────────────────────────────

class WandbLogWriter(KVWriter):
    """SB3 logger output that forwards every metric dump to wandb."""

    def write(self, key_values, key_excluded, step):
        import wandb

        filtered = {}
        for key, value in key_values.items():
            if not isinstance(value, (int, float)):
                continue
            excluded_formats = key_excluded.get(key, ())
            if "wandb" not in excluded_formats:
                filtered[key] = value
        if filtered:
            wandb.log(filtered, step=step)

    def close(self):
        pass


class WandbEpisodeCallback(BaseCallback):
    """Logs per-episode attack metrics and rolling averages to wandb."""

    def __init__(self, window=100, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.episode_count = 0
        self.success_buf = deque(maxlen=window)
        self.queries_buf = deque(maxlen=window)
        self.distortion_l2_buf = deque(maxlen=window)
        self.distortion_linf_buf = deque(maxlen=window)
        self.reward_buf = deque(maxlen=window)
        # Rolling fraction of steps with exactly-zero reward (hill-climbing rejects)
        self.zero_reward_buf = deque(maxlen=window * 128)

    def _on_step(self) -> bool:
        import wandb

        # Track zero-reward steps across all parallel envs (reject fraction)
        rewards = self.locals.get("rewards", [])
        for r in rewards:
            self.zero_reward_buf.append(1.0 if float(r) == 0.0 else 0.0)

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue

            self.episode_count += 1
            self.success_buf.append(metrics["attack_success"])
            self.queries_buf.append(metrics["queries"])
            self.distortion_l2_buf.append(metrics["distortion_l2"])
            self.distortion_linf_buf.append(metrics["distortion_linf"])
            self.reward_buf.append(metrics["cumulative_reward"])

            log_dict = {
                "episode/queries": metrics["queries"],
                "episode/f_start": metrics["f_start"],
                "episode/f_end": metrics["f_end"],
                "episode/cumulative_reward": metrics["cumulative_reward"],
                "episode/discounted_reward": metrics["discounted_reward"],
                "episode/distortion_l2": metrics["distortion_l2"],
                "episode/distortion_linf": metrics["distortion_linf"],
                "episode/attack_success": metrics["attack_success"],
                "episode/count": self.episode_count,
            }

            if len(self.success_buf) >= 10:
                log_dict["episode/attack_success_rate"] = np.mean(self.success_buf)
                log_dict["episode/mean_queries"] = np.mean(self.queries_buf)
                log_dict["episode/mean_distortion_l2"] = np.mean(self.distortion_l2_buf)
                log_dict["episode/mean_distortion_linf"] = np.mean(
                    self.distortion_linf_buf
                )
                log_dict["episode/mean_reward"] = np.mean(self.reward_buf)

            if self.zero_reward_buf:
                log_dict["episode/zero_reward_step_fraction"] = np.mean(
                    self.zero_reward_buf
                )

            wandb.log(log_dict, step=self.num_timesteps)

        return True


if args.use_wandb:
    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "dataset": args.dataset,
            "victim_model": args.victim_model,
            "rl_attack_type": args.rl_attack_type,
            "epsilon": args.epsilon,
            "c": args.c,
            "N": args.N,
            "theta": args.theta,
            "training_steps": args.training_steps,
            "n_envs": args.n_envs,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "label": args.label,
            "perturbation": args.perturbation,
            "epsilon_linf": args.epsilon_linf,
            "delta_grid_size": args.delta_grid_size,
            "one_step": args.one_step,
            "target_kl": args.target_kl,
            "mask": "full_image" if args.perturbation == "iap_segment" else "n/a",
        },
    )


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Make a directory called learning_results if it does not exist (for logging)
if not os.path.exists("learning_results"):
    os.makedirs("learning_results")

# Load the victim model
if args.victim_model == "qwen3vl":
    # Qwen3-VL is a zero-shot VLM classifier; its weights come from the HF hub
    # (or a local snapshot), so it skips the torchvision + .pth loading below.
    victim_model = load_qwen3vl_victim(args.dataset)
else:
    if args.victim_model == "resnet50":
        victim_model = torchvision.models.resnet50(weights=None)
        victim_model.fc = torch.nn.Linear(
            victim_model.fc.in_features, 10
        )  # 10 classes for CIFAR-10/SVHN
    elif args.victim_model == "vgg16":
        victim_model = torchvision.models.vgg16(weights=None)
        victim_model.classifier[6] = torch.nn.Linear(
            victim_model.classifier[6].in_features, 10
        )  # 10 classes for CIFAR-10/SVHN
    elif args.victim_model == "vit_b_16":
        victim_model = torchvision.models.vit_b_16(weights=None)
        victim_model.heads.head = torch.nn.Linear(
            victim_model.heads.head.in_features, 10
        )  # 10 classes for CIFAR-10/SVHN
    else:
        raise ValueError(f"Victim model {args.victim_model} not supported.")
    victim_model.to(device)

    # Load the victim model weights (from the ./victim_models/ directory)
    state_dict = torch.load(
        "./victim_models/{}+{}.pth".format(args.dataset, args.victim_model), map_location=device
    )
    victim_model.load_state_dict(state_dict['model_state_dict'])

    # Set the victim model to evaluation mode
    victim_model.eval()

# Load the RL training dataset D (50% of the original test dataset detailed in the 'Experimental Setup' subsection of the paper)
raw_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)
if args.dataset == "cifar10":
    dataset = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=raw_transform
    )
elif args.dataset == "svhn":
    dataset = torchvision.datasets.SVHN(
        root="./dataset", split="test", download=True, transform=raw_transform
    )
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")
dataset = [
    (datapoint[0], datapoint[1]) for datapoint in dataset
]  # Convert to list of tuples (image, label)
split_index = int(len(dataset) * 0.5)
D = dataset[:split_index]

# Create the attack configuration (details in the 'Learning Adversarial Policies' section of the paper)
rl_attack_config = {"type": args.rl_attack_type, "epsilon": args.epsilon, "c": args.c}

# Shared IAP sensitivity cache across the parallel envs (keyed by dataset index),
# so the expensive official-IAP std pass runs at most once per CIFAR image.
iap_sensitivity_cache = {}


# Create the environment(s) for the RL attack (n_envs parallel environments)
def make_env():
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        env = EvasionEnv(
            victim_model,
            D,
            args.N,
            args.theta,
            rl_attack_config,
            perturbation=args.perturbation,
            epsilon_linf=args.epsilon_linf,
            delta_grid_size=args.delta_grid_size,
            sensitivity_cache=iap_sensitivity_cache,
            one_step=args.one_step,
        )
        env = Monitor(env)
        return env

    return _init


training_env = DummyVecEnv([make_env() for i in range(args.n_envs)])

# Create the logger and the attacker (detailed in the 'Experimental Setup' subsection of the paper)
logger = configure("./learning_results/", ["stdout", "csv"])
if args.use_wandb:
    logger.output_formats.append(WandbLogWriter())
adversarial_agent = PPOAgent(
    feature_extractor="efficientnet_b0",
    policy_head="default",
    env=training_env,
    learning_rate=lambda x: x
    * 0.0025,  # Learning rate schedule (detailed in the 'Experimental Setup' subsection of the paper)
    n_steps=128,  # Number of steps to run in each environment before updating the policy (128 * n_envs)
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=lambda x: x
    * 0.1,  # Clip range schedule (detailed in the 'Experimental Setup' subsection of the paper)
    target_kl=args.target_kl,  # None = paper default; set for high-dim actions (see --target_kl help)
)
adversarial_agent.model.set_logger(logger)

# Write the episodic data file header
with open("learning_results/episodic.csv", "w") as f:
    f.write(
        "queries,f_start,f_end,cumulative_reward,discounted_reward,label,prediction,distortion\n"
    )

# Train the adversarial agent
callback = WandbEpisodeCallback() if args.use_wandb else None
adversarial_agent.train(steps=args.training_steps, callback=callback)

# Make sure the agents directory exists
if not os.path.exists("agents"):
    os.makedirs("agents")

# Save the attacker policy to a file - {args.label}.pth
torch.save(
    adversarial_agent.model.policy.state_dict(), "./agents/{}.pth".format(args.label)
)

if args.use_wandb:
    wandb.finish()
