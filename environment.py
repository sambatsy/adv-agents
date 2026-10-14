import stable_baselines3
import gymnasium as gym
import numpy as np
import random
import torch
import torchvision
from PIL import Image
import math
import os
import torch

from iap_segment import apply_iap_segment_perturbation


class EvasionEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        victim_model,
        dataset,
        N,
        theta,
        rl_attack_config: dict,
        perturbation: str = "paper",
        epsilon_linf: float = 8.0 / 255.0,
        delta_grid_size: int = 16,
        sensitivity_cache: dict = None,
        one_step: bool = False,
    ):
        super(EvasionEnv, self).__init__()
        self.victim_model = victim_model
        self.dataset = dataset
        self.N = N
        self.theta = theta
        self.rl_attack_config = rl_attack_config
        self.victim_model.eval()

        # One-step mode: image -> policy -> one complete perturbation -> score.
        # The episode is a single query (done=True after one step), with no
        # hill-climbing accumulation, and a dense classifier-based reward. This
        # removes the multi-step credit-assignment problem. Pair it with a
        # full-image backend ("iap_segment") so the single action is a complete
        # perturbation rather than only N sparse pixels.
        self.one_step = one_step

        # Perturbation backend: "paper" (sparse-pixel) or "iap_segment" (IAP grid)
        if perturbation not in {"paper", "iap_segment"}:
            raise ValueError(f"Unknown perturbation '{perturbation}'")
        self.perturbation = perturbation
        self.epsilon_linf = epsilon_linf
        self.delta_grid_size = delta_grid_size
        # Shared across the parallel envs so the IAP std pass runs once per image
        self.sensitivity_cache = sensitivity_cache if sensitivity_cache is not None else {}

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.victim_model.to(self.device)

        # PIL for visualization / IAP rendering
        self.to_pil = torchvision.transforms.ToPILImage()
        self.to_tensor = torchvision.transforms.ToTensor()

        # Set logging directory
        self.log_dir = "learning_results"

        # Initialize the preprocess transform for the pytorch vision models
        self.preprocess = torch.nn.Sequential(
            torchvision.transforms.Resize(256, antialias=True),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        )

        # Initialize the observation and action spaces
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float32
                ),
                "outputs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(10,), dtype=np.float32
                ),
                "source_class": gym.spaces.Discrete(10),
            }
        )
        if self.perturbation == "paper":
            self.action_space = gym.spaces.Box(
                low=-2.0, high=2.0, shape=(self.N * (5),), dtype=np.float32
            )  # Each N has (x, y, c1, c2, c3) where x and y are the coordinates and c1, c2, c3 are the perturbations
        else:
            # IAP: a low-resolution (3, G, G) RGB delta grid, flattened and clamped to [-1, 1]
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3 * self.delta_grid_size * self.delta_grid_size,),
                dtype=np.float32,
            )

    def project(self, perturbations):
        l2_norm = torch.norm(perturbations, p=2)
        perturbations = perturbations / l2_norm * self.rl_attack_config["epsilon"]
        self.sample = self.sample_point[0] + perturbations

    def generate_query(self, action):
        # (N,\theta) Configuration (detailed in the 'States and Actions' subsection of the paper)
        for i in range(
            0, self.N * (5), 5
        ):  # Each action consists of (x, y, c1, c2, c3)
            x = min(
                int((action[i] + 2.0) * 32 / 4.0), 31
            )  # Assuming 32x32 input size for CIFAR-10 and SVHN
            y = min(
                int((action[i + 1] + 2.0) * 32 / 4.0), 31
            )  # Assuming 32x32 input size for CIFAR-10 and SVHN
            for c in range(3):  # Assuming RGB channels
                distortion = action[i + 2 + c] * self.theta / 2.0
                self.sample[c, x, y] = min(
                    1.0, max(self.sample[c, x, y] + distortion, 0.0)
                )

        # Project the perturbations if the attack is RL Max Loss (to epsilon distortion)
        if self.rl_attack_config["type"] == "RLMaxLoss":
            perturbations = torch.clone(self.sample - self.sample_point[0])
            lp_norm = torch.norm(perturbations, p=2)
            if lp_norm > self.rl_attack_config["epsilon"]:
                self.project(perturbations)
                self.movement = self.rl_attack_config["epsilon"]
            else:
                self.movement = torch.norm(
                    self.sample - self.sample_point[0], p=2
                ).item()

        return self.sample

    def render_iap(self, action):
        # Render a candidate perturbed image from the CLEAN image with this action.
        # Unlike the paper's in-place accumulation, IAP is a stateless renderer:
        # the full perturbation is determined by `action` applied to the clean
        # source, so each step proposes a fresh candidate from clean. Hill climbing
        # (in step) then keeps it only if it improves the margin over best-so-far.
        clean = self.sample_point[0]  # (3, 32, 32) float in [0, 1]
        clean_pil = self.to_pil(clean.detach().cpu())
        action_t = torch.as_tensor(action, dtype=torch.float32)
        edited_pil, _actual_delta, _upsampled, _meta = apply_iap_segment_perturbation(
            clean_pil,
            self.full_mask,
            action_t,
            self.epsilon_linf,
            delta_grid_size=self.delta_grid_size,
            sensitivity_cache=self.sensitivity_cache,
            cache_key=self.cache_key,
            support_label="instance",
        )
        candidate = self.to_tensor(edited_pil).to(clean.dtype)  # (3, 32, 32) in [0, 1]
        return candidate

    def dynamics(self, x):
        # Get the model outputs, apply softmax if necessary (to get the Z(x_t) outputs)
        outputs = self.victim_model(self.preprocess(x).to(self.device).unsqueeze(0))[
            0
        ].detach()
        if torch.sum(outputs).item() != 1:
            outputs = torch.nn.functional.softmax(outputs, dim=0)

        # Get the value indexed by the sample point, and the classification
        confidence = outputs[self.sample_point[1]].item()
        prediction = torch.argmax(outputs).item()

        # Get the entropy loss (negative log of the confidence) (labeled as the f(x,y) function in the 'Reward and Transition Functions' subsection of the paper)
        loss = -torch.log(
            torch.tensor(confidence + 1e-10)
        ).item()  # Adding a small value to avoid log(0)

        # Return the loss, prediction, and outputs
        return loss, prediction, outputs

    def _classifier_reward(self, outputs):
        # Simple classifier-based reward: CW-style margin on the single resulting
        # image. reward = max_{j != y} p_j - p_y, in [-1, 1]. Negative while the
        # true class y stays on top, positive once the attack makes some other
        # class win (i.e. a successful evasion). Dense everywhere, so it gives a
        # gradient of signal even before the prediction actually flips.
        y = self.sample_point[1]
        p_true = outputs[y].item()
        other = outputs.clone()
        other[y] = -float("inf")
        p_other = torch.max(other).item()
        return p_other - p_true

    def _step_one_shot(self, action):
        # image -> policy -> one complete perturbation -> model score.
        # One action produces the whole perturbation from the CLEAN image; we
        # query the victim once, score it with a classifier-based reward, and end
        # the episode. No hill-climbing, no best-so-far accumulation.
        if self.perturbation == "paper":
            self.sample = self.generate_query(action)
        else:
            self.sample = self.render_iap(action)

        # Distortion of the single perturbation, relative to the clean image.
        delta = self.sample - self.sample_point[0]
        self.movement = torch.norm(delta, p=2).item()
        self.movement_linf = torch.norm(delta.flatten(), p=float("inf")).item()
        self.cumulative_lp = self.movement
        self.cumulative_linf = self.movement_linf

        # Score the single resulting image.
        model_loss, prediction, outputs = self.dynamics(self.sample)
        self.queries += 1
        self.model_loss = model_loss
        self.prediction = prediction
        self.outputs = outputs

        reward = self._classifier_reward(outputs)
        self.cumulative_reward = reward
        self.discounted_reward = reward

        # One step -> always terminal (a genuine episode end, not a truncation).
        done, truncated = True, False

        with open("{}/episodic.csv".format(self.log_dir), "a") as f:
            f.write(
                "{},{},{},{},{},{},{},{}\n".format(
                    self.queries,
                    self.start_state_model_loss,
                    self.model_loss,
                    self.cumulative_reward,
                    self.discounted_reward,
                    self.sample_point[1],
                    self.prediction,
                    self.cumulative_lp,
                )
            )

        info = self.get_info()
        info["episode_metrics"] = {
            "queries": self.queries,
            "f_start": self.start_state_model_loss,
            "f_end": self.model_loss,
            "cumulative_reward": self.cumulative_reward,
            "discounted_reward": self.discounted_reward,
            "distortion": self.cumulative_lp,
            "distortion_l2": self.cumulative_lp,
            "distortion_linf": self.cumulative_linf,
            "attack_success": float(prediction != self.sample_point[1]),
        }
        return self.get_obs(), reward, done, truncated, info

    def step(self, action):
        if self.one_step:
            return self._step_one_shot(action)

        # Generate the candidate perturbed image for this action.
        # paper: accumulate the sparse-pixel perturbation on top of the best-so-far
        #        image (self.sample was restored to s_tminus1 on the last reject).
        # iap_segment: render a fresh candidate from the CLEAN image.
        if self.perturbation == "paper":
            self.sample = self.generate_query(action)
        else:
            self.sample = self.render_iap(action)

        # Get the movement (both norms, relative to the clean image)
        delta = self.sample - self.sample_point[0]
        self.movement = torch.norm(delta, p=2).item()
        self.movement_linf = torch.norm(delta.flatten(), p=float("inf")).item()

        # Get the model loss, prediction, and outputs
        model_loss, prediction, outputs = self.dynamics(self.sample)
        self.queries += 1

        # Calculate the reward
        if self.rl_attack_config["type"] == "RLMinNorm":
            reward = (
                self.rl_attack_config["c"] * (self.cumulative_lp - self.movement)
                + 20.0 * (model_loss - self.model_loss)
            )  # Scale the model loss to avoid negligible differences in confidence for effective RL
        elif self.rl_attack_config["type"] == "RLMaxLoss":
            reward = (
                20.0 * (model_loss - self.model_loss)
            )  # Scale the model loss to avoid negligible differences in confidence for effective RL
        else:
            raise ValueError("Invalid reward type.")

        # Employ Hill Climbing transition (detailed in both 'RL Max Loss' and 'RL Min Norm' subsections of the paper)
        if reward < 0:
            reward = 0.0
            # Reject: restore BOTH the image and the observation to best-so-far.
            # self.outputs/self.prediction are left untouched below so get_obs()
            # returns the best-so-far victim scores, not the rejected candidate's.
            self.sample = torch.clone(self.s_tminus1)
            accepted = False
        else:
            self.model_loss = model_loss
            self.cumulative_lp = self.movement
            self.cumulative_linf = self.movement_linf
            self.s_tminus1 = torch.clone(self.sample)
            accepted = True

        # Determine the done condition (terminal states defined in 'The Reinforcement Learning Attack' subsection of the paper)
        if prediction != self.sample_point[1]:
            done = True
        else:
            done = False

        # Determine the truncated condition
        if self.queries >= 1000:  # Assuming a maximum of 1000 queries per episode
            truncated = True
        else:
            truncated = False

        # self.prediction feeds info/CSV logging only (not the observation), so it
        # tracks the candidate as in the original code. self.outputs IS part of the
        # observation, so it is restored to best-so-far on a reject by only updating
        # it when the candidate was accepted.
        self.prediction = prediction
        if accepted:
            self.outputs = outputs
        self.cumulative_reward += reward
        self.discounted_reward += reward * (
            0.99 ** (self.queries - 2)
        )  # Discounted reward with gamma = 0.99

        # If done or truncated, write to the episodic data file
        if done or truncated:
            with open("{}/episodic.csv".format(self.log_dir), "a") as f:
                f.write(
                    "{},{},{},{},{},{},{},{}\n".format(
                        self.queries,
                        self.start_state_model_loss,
                        self.model_loss,
                        self.cumulative_reward,
                        self.discounted_reward,
                        self.sample_point[1],
                        self.prediction,
                        self.cumulative_lp,
                    )
                )

        # Get the info
        info = self.get_info()

        if done or truncated:
            info["episode_metrics"] = {
                "queries": self.queries,
                "f_start": self.start_state_model_loss,
                "f_end": self.model_loss,
                "cumulative_reward": self.cumulative_reward,
                "discounted_reward": self.discounted_reward,
                "distortion": self.cumulative_lp,
                "distortion_l2": self.cumulative_lp,
                "distortion_linf": self.cumulative_linf,
                "attack_success": float(prediction != self.sample_point[1]),
            }

        # Return the observation, reward, done, truncated, and info
        final_state = self.get_obs()

        return final_state, reward, done, truncated, info

    def get_obs(self):
        return {
            "image": self.preprocess(self.sample),
            "outputs": self.outputs.to("cpu"),
            "source_class": self.sample_point[1],
        }

    def get_info(self):
        return {
            "image_size": self.sample.shape,
            "movement": self.movement,
            "prediction": self.prediction,
            "original_label": self.sample_point[1],
        }

    def reset(self, seed=42):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)

        # Randomly sample a sample from the dataset
        self.sample_point = None
        while self.sample_point is None:
            self.index = np.random.randint(0, len(self.dataset))
            x = self.dataset[self.index][0]
            y = self.dataset[self.index][1]
            if (
                self.victim_model(
                    self.preprocess(torch.clone(x)).to(self.device).unsqueeze(0)
                )
                .argmax()
                .item()
                == y
            ):
                self.sample_point = (x, y)
        self.sample = torch.clone(self.sample_point[0])

        # IAP rendering state: full-image mask (CIFAR has no instance masks) and a
        # per-image cache key so the IAP sensitivity std pass runs once per image.
        h, w = self.sample.shape[-2], self.sample.shape[-1]
        self.full_mask = np.ones((h, w), dtype=np.uint8)
        self.cache_key = int(self.index)

        # Initialize Cumulative Lp Distance
        self.cumulative_lp = 0.0
        self.cumulative_linf = 0.0

        # Initialize log confidence of the original sample
        self.model_loss, self.prediction, self.outputs = self.dynamics(self.sample)

        # Record Original model loss
        self.start_state_model_loss = self.model_loss
        self.misclass = False

        # Initialize the Timestep
        self.queries = 1

        # Initialize the movement and the previous state
        self.movement = 0.0
        self.movement_linf = 0.0
        self.s_tminus1 = torch.clone(self.sample)

        # Initialize the Cumulative Reward
        self.cumulative_reward = 0.0
        self.discounted_reward = 0.0

        # Return the initial observation and info
        return self.get_obs(), self.get_info()

    def setImage(self, sample_point):
        # Set the sample point to the given image and reset the environment
        self.sample_point = sample_point
        self.sample = torch.clone(self.sample_point[0])

        # IAP rendering state: full-image mask and a per-image cache key. In eval
        # there is no dataset index, so key on the clean tensor's identity (stable
        # and unique within the held-out list).
        h, w = self.sample.shape[-2], self.sample.shape[-1]
        self.full_mask = np.ones((h, w), dtype=np.uint8)
        self.cache_key = id(self.sample_point[0])

        # Initialize log confidence of the original sample
        self.model_loss, self.prediction, self.outputs = self.dynamics(self.sample)

        # Initialize Cumulative Lp Distance
        self.cumulative_lp = 0.0
        self.cumulative_linf = 0.0

        # Record Original model loss
        self.start_state_model_loss = self.model_loss
        self.misclass = False

        # Initialize the Timestep
        self.queries = 1

        # Initialize the movement and the previous state
        self.movement = 0.0
        self.movement_linf = 0.0
        self.s_tminus1 = torch.clone(self.sample)

        # Initialize the Cumulative Reward
        self.cumulative_reward = 0.0
        self.discounted_reward = 0.0

        self.log_dir = "evaluation_results"

        # Return the initial observation and info
        return self.get_obs(), self.get_info()

    def render(self, mode="human"):
        # convert the normalized torch tensor to an integer numpy array corresponding to the image
        image = (self.sample_point[0] * 255).squeeze(0).type(torch.uint8)
        game_image = (self.sample * 255).squeeze(0).type(torch.uint8)

        return image, game_image

    def close(self):
        pass
