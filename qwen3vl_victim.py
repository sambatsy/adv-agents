"""Qwen3-VL wrapped as a black-box image-classifier victim.

The RL evasion environment (see ``environment.py``) treats a victim model as a
plain callable ``victim_model(images) -> (batch, num_classes)`` scores tensor,
plus ``.eval()`` and ``.to(device)``. Torchvision classifiers satisfy that out
of the box; a generative VLM like Qwen3-VL does not. This module adapts one to
that interface by using it as a *zero-shot* classifier:

  * the image tensor is turned back into an RGB picture,
  * the model is shown a multiple-choice prompt whose options are the dataset's
    class names (A, B, C, ...), and
  * the logits over the option *letters* at the first generated position are
    read off as the per-class scores.

This is the same first-token probe the MSSBench recognition loader used, but
over ``num_classes`` options instead of two. The environment softmaxes the
returned scores, so raw logits are fine.

NOTE ON COST: an 8B VLM forward runs per query, and the RL loop issues up to
1000 queries per episode across many parallel environments. This is orders of
magnitude slower than the torchvision victims and is intended for small-scale
evaluation, not the full multi-million-step training run.
"""

import json
import os
from pathlib import Path

import torch
import torchvision
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# Default checkpoint and knobs, overridable from the environment so the same
# code path works online (HF hub) and offline (a local snapshot / PEFT adapter).
MODEL_PATH = os.environ.get("MSSBENCH_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
MAX_NEW_TOKENS = 1  # only the first answer token is scored
VICTIM_BATCH_SIZE = int(os.environ.get("MSSBENCH_VICTIM_BATCH_SIZE", "8"))
LOCAL_FILES_ONLY = os.environ.get("HF_HUB_OFFLINE") == "1"

# ImageNet stats used by the environment's preprocess transform. Inputs that
# arrive already normalized (from ``EvasionEnv.dynamics``) are inverted back to
# a viewable [0, 1] image before being handed to the VLM; raw [0, 1] inputs
# (from ``EvasionEnv.reset``) pass through unchanged.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# The environment's observation space fixes 10 classes; both datasets have 10.
CLASS_NAMES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "svhn": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}

# Option letters A, B, C, ... one per class.
_OPTION_LETTERS = [chr(ord("A") + i) for i in range(26)]


def _load_model_and_processor(model_path):
    """Load Qwen3-VL, transparently resolving a PEFT adapter checkpoint."""
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open() as f:
            base_model_path = json.load(f)["base_model_name_or_path"]

        processor = AutoProcessor.from_pretrained(
            base_model_path, local_files_only=LOCAL_FILES_ONLY
        )
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            dtype="auto",
            device_map="auto",
            local_files_only=LOCAL_FILES_ONLY,
        )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "The selected checkpoint is a PEFT adapter, but peft is not installed."
            ) from exc
        model = PeftModel.from_pretrained(base_model, model_path)
        return model, processor

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=LOCAL_FILES_ONLY
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        local_files_only=LOCAL_FILES_ONLY,
    )
    return model, processor


def _first_token_ids(tokenizer, options):
    """First-token id for each option string (the letter the model emits)."""
    ids = []
    for option in options:
        token_ids = tokenizer.encode(option, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Option {option!r} tokenized to nothing.")
        ids.append(token_ids[0])
    return ids


class Qwen3VLClassifier:
    """Zero-shot Qwen3-VL classifier exposing the victim-model interface.

    Callable ``self(images)`` where ``images`` is a ``(B, 3, H, W)`` (or a bare
    ``(3, H, W)``) float tensor and returns a ``(B, num_classes)`` logit tensor
    on the input's device. ``eval()`` and ``to()`` are accepted for parity with
    the torchvision victims; device placement is handled by ``device_map`` so
    ``to()`` only records the target for the returned tensor.
    """

    def __init__(self, dataset, model_path=MODEL_PATH, batch_size=VICTIM_BATCH_SIZE):
        if dataset not in CLASS_NAMES:
            raise ValueError(
                f"Dataset {dataset!r} has no class names; add it to CLASS_NAMES."
            )
        self.class_names = CLASS_NAMES[dataset]
        self.num_classes = len(self.class_names)
        self.batch_size = batch_size

        self.model, self.processor = _load_model_and_processor(model_path)
        self.model.eval()

        self._option_letters = _OPTION_LETTERS[: self.num_classes]
        self._letter_token_ids = _first_token_ids(
            self.processor.tokenizer, self._option_letters
        )
        self._prompt = self._build_prompt(dataset)
        self._to_pil = torchvision.transforms.ToPILImage()
        self._mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def _build_prompt(self, dataset):
        lines = [
            f"{letter}. {name}"
            for letter, name in zip(self._option_letters, self.class_names)
        ]
        task = (
            "Which digit is shown in the image?"
            if dataset == "svhn"
            else "Classify the main object in the image into exactly one category."
        )
        return (
            f"{task}\n"
            + "\n".join(lines)
            + "\nAnswer with only the letter of the correct option."
        )

    def eval(self):
        self.model.eval()
        return self

    def to(self, *args, **kwargs):
        # Weights are placed by device_map="auto"; nothing to move here. Kept so
        # the environment's ``victim_model.to(self.device)`` call is a no-op.
        return self

    def _tensor_to_pil(self, image):
        image = image.detach().to("cpu", torch.float32)
        # Inputs from EvasionEnv.dynamics are ImageNet-normalized (values escape
        # [0, 1]); invert that. Raw [0, 1] inputs (from reset) are left alone.
        if image.min() < -1e-3 or image.max() > 1.0 + 1e-3:
            image = image * self._std + self._mean
        image = image.clamp(0.0, 1.0)
        return self._to_pil(image)

    def _messages(self, pil_image):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

    @torch.inference_mode()
    def __call__(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        pil_images = [self._tensor_to_pil(img) for img in images]

        model_device = getattr(self.model, "device", None)
        letter_ids = torch.tensor(self._letter_token_ids)

        scores = []
        for start in range(0, len(pil_images), self.batch_size):
            chunk = pil_images[start : start + self.batch_size]

            previous_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left"
            try:
                inputs = self.processor.apply_chat_template(
                    [self._messages(image) for image in chunk],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)
            finally:
                self.processor.tokenizer.padding_side = previous_padding_side

            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # scores[0]: first generated position's logits, shape (chunk, vocab).
            first_logits = generated.scores[0]
            class_logits = first_logits[:, letter_ids.to(first_logits.device)]
            scores.append(class_logits)

        scores = torch.cat(scores, dim=0).float()
        return scores.to(images.device)


def load_qwen3vl_victim(dataset, model_path=MODEL_PATH):
    """Build an eval-mode Qwen3-VL victim classifier for ``dataset``."""
    return Qwen3VLClassifier(dataset, model_path=model_path).eval()
