"""Save IAP sensitivity maps for 4 CIFAR-10 images at 32x32.

Diagnostic to confirm the official-IAP sensitivity is NOT nearly uniform at 32x32
(if it were, the IAP weighting would add nothing over a plain grid perturbation).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

from iap_segment import iap_sensitivity_map

raw_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, download=True, transform=raw_transform
)

to_pil = torchvision.transforms.ToPILImage()

indices = [0, 1, 2, 3]
fig, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 8))

for col, idx in enumerate(indices):
    img_tensor, label = dataset[idx]
    img_pil = to_pil(img_tensor)  # 32x32 RGB

    sens = iap_sensitivity_map(img_pil, (32, 32))  # (32, 32) float

    # Top row: the CIFAR image
    axes[0, col].imshow(np.asarray(img_pil))
    axes[0, col].set_title(f"CIFAR idx={idx}, label={label}")
    axes[0, col].axis("off")

    # Bottom row: the sensitivity map
    im = axes[1, col].imshow(sens, cmap="viridis")
    rng = sens.max() - sens.min()
    axes[1, col].set_title(
        f"sensitivity\nmin={sens.min():.3f} max={sens.max():.3f}\n"
        f"std={sens.std():.3f} range={rng:.3f}"
    )
    axes[1, col].axis("off")
    fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

plt.tight_layout()
out = "sensitivity_maps_32x32.png"
plt.savefig(out, dpi=120)
print(f"Saved {out}")

# Also print numeric summary so uniformity is checkable without opening the image.
print("\nPer-image sensitivity-map statistics at 32x32:")
for idx in indices:
    img_tensor, label = dataset[idx]
    sens = iap_sensitivity_map(to_pil(img_tensor), (32, 32))
    cv = sens.std() / (sens.mean() + 1e-12)
    print(
        f"  idx={idx}: min={sens.min():.4f} max={sens.max():.4f} "
        f"mean={sens.mean():.4f} std={sens.std():.4f} "
        f"coeff_of_variation={cv:.4f}"
    )
