"""Apply an official-IAP-weighted action inside one cached binary mask.

The functions ``get_std_x``, ``get_std_y``, and ``get_final_std`` below are
vendored verbatim from the official IAP repository at commit
0c800ce72370d6d44873c7029ca528b140e7e9f4:
https://github.com/subratkishoredutta/IAP/blob/0c800ce72370d6d44873c7029ca528b140e7e9f4/src/utils.py

MIT License

Copyright (c) 2024 SUBRAT KISHORE DUTTA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Hashable, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IAP_SOURCE_COMMIT = "0c800ce72370d6d44873c7029ca528b140e7e9f4"
IAP_SENSITIVITY_SIZE = 224
# Stabilizer added to the local standard deviation before inverting it. This
# matches the value used by the action-construction weighting below, so the
# perceptibility reward and the perturbation transform share one sensitivity
# definition Sens(x; p) = 1 / (sigma_p + IAP_SENSITIVITY_LAMBDA).
IAP_SENSITIVITY_LAMBDA = 0.001


# Begin verbatim official IAP functions.
def get_std_x(input_tensor,c=36):
  # Assuming you have an image with dimensions 224x290
  image = input_tensor.clone()
  image.requires_grad_()
  #plt.imshow(image.detach().numpy())
  #print(image.shape)
  subset_size_x=3
  subset_size_y=1
  stride = 1
  stdten = torch.zeros(input_tensor.shape[2],input_tensor.shape[3])
  for i in range(0, image.size(2) - subset_size_y + 1, stride):
    for j in range(0, image.size(3) - subset_size_x + 1, stride):
      subset = image[0][:,i:i+subset_size_y, j:j+subset_size_x]
      subsetR=subset[0]
      subsetG=subset[1]
      subsetB=subset[2]
      std = (subsetR.std()+subsetG.std()+subsetB.std())/3
      stdten[i][j]=std
  #plt.figure()
  #plt.imshow(stdten.detach().numpy())
  return stdten

def get_std_y(input_tensor):
  # Assuming you have an image with dimensions 224x290
  image = input_tensor.clone()
  image.requires_grad_()
  #plt.imshow(image.detach().numpy())
  #print(image.shape)
  subset_size_x=1
  subset_size_y=3
  stride = 1
  stdten = torch.zeros(input_tensor.shape[2],input_tensor.shape[3])
  for i in range(0, image.size(2) - subset_size_y + 1, stride):
    for j in range(0, image.size(3) - subset_size_x + 1, stride):
      subset = image[0][:,i:i+subset_size_y, j:j+subset_size_x]
      subsetR=subset[0]
      subsetG=subset[1]
      subsetB=subset[2]
      std = (subsetR.std()+subsetG.std()+subsetB.std())/3
      stdten[i][j]=std
      #print(subsetB)
      #print(subsetB)
  #plt.figure()
  #plt.imshow(stdten.detach().numpy())
  return stdten

def get_final_std(image):
  stdx=get_std_x(image)
  stdy=get_std_y(image)
  min_tensor = torch.sqrt(torch.min(stdx, stdy))
  return min_tensor
# End verbatim official IAP functions.


def _load_instance_mask(
    instance_mask: Path | Image.Image | np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    if isinstance(instance_mask, (str, Path)):
        with Image.open(instance_mask) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
    elif isinstance(instance_mask, Image.Image):
        mask = np.asarray(instance_mask.convert("L"), dtype=np.uint8) > 0
    else:
        mask = np.asarray(instance_mask) > 0
    width, height = image_size
    if mask.shape != (height, width):
        raise ValueError(
            f"instance mask shape {mask.shape} does not match {(height, width)}"
        )
    if not np.any(mask):
        raise ValueError("instance mask is empty")
    return mask


def _iap_weight_at_224(
    image: Image.Image,
    sensitivity_cache: MutableMapping[Hashable, torch.Tensor] | None,
    cache_key: Hashable | None,
) -> torch.Tensor:
    if sensitivity_cache is not None and cache_key in sensitivity_cache:
        return sensitivity_cache[cache_key]

    resized = image.resize(
        (IAP_SENSITIVITY_SIZE, IAP_SENSITIVITY_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        stdimg = get_final_std(image_tensor)
        sensitivity = 1 / (stdimg + 0.001)
        weight = (1 / sensitivity).detach().to(dtype=torch.float32, device="cpu")
    if sensitivity_cache is not None and cache_key is not None:
        sensitivity_cache[cache_key] = weight
    return weight


def iap_sensitivity_map(
    image: Image.Image,
    size: tuple[int, int],
    *,
    sensitivity_cache: MutableMapping[Hashable, torch.Tensor] | None = None,
    cache_key: Hashable | None = None,
) -> np.ndarray:
    """Return the official-IAP sensitivity map upsampled to ``size``.

    ``size`` is ``(height, width)``. The returned map is
    ``Sens(x; p) = 1 / (sigma_p + IAP_SENSITIVITY_LAMBDA)`` where ``sigma_p`` is
    the official IAP local standard deviation computed at 224x224. The result
    reuses ``sensitivity_cache`` (shared with the perturbation renderer, keyed on
    ``cache_key``) so the expensive std pass runs at most once per image.
    """

    height, width = size
    if height < 1 or width < 1:
        raise ValueError("size must be positive (height, width)")
    # ``_iap_weight_at_224`` returns sigma_p + IAP_SENSITIVITY_LAMBDA at 224x224
    # and populates the shared cache; invert it to recover the sensitivity.
    weight_224 = _iap_weight_at_224(image, sensitivity_cache, cache_key)
    sensitivity_224 = 1.0 / weight_224
    upsampled = F.interpolate(
        sensitivity_224[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return upsampled.to(dtype=torch.float32, device="cpu").numpy()


def masked_iap_distance(
    clean: np.ndarray,
    adversarial: np.ndarray,
    mask: np.ndarray,
    *,
    numerical_epsilon: float = 1.0e-8,
    sensitivity_map: np.ndarray | None = None,
) -> float:
    """Return the mask-normalized IAP perceptibility distance DIAPM.

    Implements the mask-adapted IAP distance from the adapted pipeline:

        DIAPM = sum_p M_p Sens(x; p) sum_c |x_{p,c} - x_adv,{p,c}|
                / (C * sum_p M_p + numerical_epsilon)

    where ``Sens(x; p)`` is ``sensitivity_map`` (defaulting to ones), ``M`` is
    the boolean support ``mask``, and ``C`` is the channel count. Normalizing by
    the number of *active* mask pixels (rather than the fixed ``h * w`` patch
    area IAP uses) makes individual-object and union masks directly comparable.

    ``clean`` and ``adversarial`` are ``(H, W, C)`` arrays in the same units;
    only the in-mask difference contributes, so out-of-mask changes are ignored.
    """

    clean = np.asarray(clean, dtype=np.float32)
    adversarial = np.asarray(adversarial, dtype=np.float32)
    if clean.ndim != 3:
        raise ValueError("clean must be (H, W, C)")
    if clean.shape != adversarial.shape:
        raise ValueError("clean and adversarial must have the same shape")
    mask_bool = np.asarray(mask) > 0
    if mask_bool.shape != clean.shape[:2]:
        raise ValueError("mask and image resolutions differ")
    if not mask_bool.any():
        raise ValueError("mask is empty")

    channels = clean.shape[2]
    if sensitivity_map is None:
        sensitivity = np.ones(clean.shape[:2], dtype=np.float32)
    else:
        sensitivity = np.asarray(sensitivity_map, dtype=np.float32)
        if sensitivity.shape != clean.shape[:2]:
            raise ValueError("sensitivity_map and image resolutions differ")

    mask_float = mask_bool.astype(np.float32)
    per_pixel_abs = np.abs(adversarial - clean).sum(axis=2)
    numerator = float((mask_float * sensitivity * per_pixel_abs).sum())
    denominator = channels * float(mask_float.sum()) + numerical_epsilon
    return numerator / denominator


def apply_iap_segment_perturbation(
    image: Image.Image,
    instance_mask: Path | Image.Image | np.ndarray,
    action: torch.Tensor,
    epsilon: float,
    *,
    delta_grid_size: int = 16,
    sensitivity_cache: MutableMapping[Hashable, torch.Tensor] | None = None,
    cache_key: Hashable | None = None,
    support_label: str = "instance",
) -> tuple[Image.Image, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one low-resolution RGB action through official IAP weighting."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if delta_grid_size < 1:
        raise ValueError("delta_grid_size must be positive")
    if support_label not in {"instance", "union"}:
        raise ValueError("support_label must be instance or union")

    image = image.convert("RGB")
    width, height = image.size
    mask = _load_instance_mask(instance_mask, image.size)
    flat_action = action.detach().to(dtype=torch.float32, device="cpu").flatten()
    expected_values = 3 * delta_grid_size * delta_grid_size
    if flat_action.numel() != expected_values:
        raise ValueError(
            f"action has {flat_action.numel()} values, expected {expected_values}"
        )

    low_resolution = flat_action.reshape(
        1, 3, delta_grid_size, delta_grid_size
    ).clamp(-1.0, 1.0)
    upsampled = F.interpolate(
        low_resolution,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    iap_weight = F.interpolate(
        _iap_weight_at_224(image, sensitivity_cache, cache_key)[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    delta = (upsampled * iap_weight.unsqueeze(0) * float(epsilon)).clamp(
        -float(epsilon), float(epsilon)
    )

    original = np.asarray(image, dtype=np.uint8).copy()
    base = torch.from_numpy(original).permute(2, 0, 1).to(torch.int16)
    proposed = torch.round(delta * 255.0).to(torch.int16)
    epsilon_uint8 = int(np.floor(float(epsilon) * 255.0 + 1e-6))
    proposed = proposed.clamp(-epsilon_uint8, epsilon_uint8)
    support = torch.from_numpy(mask).unsqueeze(0)
    proposed = torch.where(support, proposed, torch.zeros_like(proposed))
    adversarial = (base + proposed).clamp(0, 255).to(torch.uint8)
    adversarial_array = adversarial.permute(1, 2, 0).numpy()
    adversarial_array[~mask] = original[~mask]

    if not np.array_equal(adversarial_array[~mask], original[~mask]):
        raise AssertionError("pixels outside the selected mask changed")
    actual_delta = (
        adversarial_array.astype(np.float32) - original.astype(np.float32)
    ) / 255.0
    max_abs_delta = float(np.abs(actual_delta).max(initial=0.0))
    if max_abs_delta > float(epsilon) + 1e-7:
        raise AssertionError(f"epsilon exceeded: {max_abs_delta} > {epsilon}")

    metadata = {
        "renderer": (
            "iap_segment" if support_label == "instance" else "iap_union"
        ),
        "official_iap_source_commit": IAP_SOURCE_COMMIT,
        "iap_sensitivity_size": IAP_SENSITIVITY_SIZE,
        "iap_weighting": "1 / (1 / (get_final_std(image) + 0.001))",
        "delta_grid_shape": [3, delta_grid_size, delta_grid_size],
        "epsilon": float(epsilon),
        f"{support_label}_mask_area_pixels": int(mask.sum()),
        f"{support_label}_mask_area_fraction": float(mask.mean()),
        "realized_max_rgb_difference": max_abs_delta * 255.0,
        f"outside_{support_label}_max_rgb_difference": 0.0,
    }
    return (
        Image.fromarray(adversarial_array, mode="RGB"),
        actual_delta,
        upsampled.numpy(),
        metadata,
    )
