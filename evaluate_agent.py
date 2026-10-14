import stable_baselines3
import gymnasium as gym
import numpy as np
import torch
import torchvision
import argparse
import os
import random
from environment import EvasionEnv
from ppo_agent import PPOAgent
from qwen3vl_victim import load_qwen3vl_victim
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Create a parser to parse command line arguments
parser = argparse.ArgumentParser(
    description="Perform a black-box evasion attack with reinforcement learning."
)

# Dataset Argument
parser.add_argument(
    "--dataset", type=str, default="cifar10", help="The dataset to use."
)

# Environment Arguments
parser.add_argument(
    "--rl_attack_type",
    type=str,
    default="RLMaxLoss",
    help="The type of RL attack to perform.",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.35,
    help="The epsilon value for the RL Max Loss attack.",
)
parser.add_argument(
    "--c", type=float, default=1.0, help="The c value for the RL Min Norm attack."
)
parser.add_argument(
    "--N", type=int, default=5, help="The number of actions to take in the environment."
)
parser.add_argument(
    "--theta", type=float, default=0.05, help="The theta value for the RL attack."
)
parser.add_argument(
    "--victim_model", type=str, default="resnet50", help="The victim model to attack."
)  # Options: 'resnet50', 'vgg16', 'vit_b_16', 'qwen3vl' (the last is a zero-shot VLM classifier)

# Evaluation Arguments
parser.add_argument(
    "--agent_file",
    type=str,
    default="./agents/agent0.pth",
    help="The file path of the trained adversarial agent.",
)

# Perturbation backend (must match the backend the agent was trained with)
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
    "--random_policy",
    action="store_true",
    help="Use a random (untrained) policy instead of loading the agent file.",
)

# Parse the arguments
args = parser.parse_args()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Make a directory called evaluation_results if it does not exist (for logging)
if not os.path.exists("evaluation_results"):
    os.makedirs("evaluation_results")

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

# Load the RL testing dataset D_prime (50% of the original test dataset)
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
D_prime = dataset[split_index:]  # Use the second half of the dataset for testing

# Create the attack configuration
rl_attack_config = {"type": args.rl_attack_type, "epsilon": args.epsilon, "c": args.c}

# Create the evaluation environment
evaluation_env = EvasionEnv(
    victim_model,
    D_prime,
    args.N,
    args.theta,
    rl_attack_config,
    perturbation=args.perturbation,
    epsilon_linf=args.epsilon_linf,
    delta_grid_size=args.delta_grid_size,
)

# Create the logger and the attacker
logger = configure("./evaluation_results/", ["stdout", "csv"])
adversarial_agent = PPOAgent(
    feature_extractor="efficientnet_b0",
    policy_head="default",
    env=evaluation_env,
    learning_rate=lambda x: x * 0.002,  # Learning rate schedule
    n_steps=128,  # Number of steps to run in each environment before updating the policy (128 * n_envs)
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=lambda x: x * 0.1,  # Clip range schedule
)
adversarial_agent.model.set_logger(logger)

# Load the trained adversarial agent's policy from the specified file (unless we
# are evaluating a random/untrained policy as a baseline).
if args.random_policy:
    print("Using a random (untrained) policy baseline.")
else:
    adversarial_agent.model.policy.load_state_dict(
        torch.load(args.agent_file, map_location=device)
    )

# Write the episodic data file header
with open("evaluation_results/episodic.csv", "w") as f:
    f.write(
        "queries,f_start,f_end,cumulative_reward,discounted_reward,label,prediction,distortion\n"
    )

# Evaluate on each sample in the evaluation dataset D_prime
for i, sample in enumerate(D_prime):
    image, label = sample

    # Reset the environment with the current sample
    evaluation_env.reset()
    evaluation_env.setImage((image, label))

    # Run the agent on the environment
    obs = evaluation_env.get_obs()
    done, truncated = False, False
    while not done and not truncated:
        action, _ = adversarial_agent.model.predict(obs)
        obs, reward, done, truncated, info = evaluation_env.step(action)
