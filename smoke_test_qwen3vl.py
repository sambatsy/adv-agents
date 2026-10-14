"""Smoke test for the Qwen3-VL victim wrapper.

Validates the new zero-shot victim in isolation, before any RL run:
  * it loads and returns a (B, num_classes) score tensor,
  * scores softmax to a valid probability vector,
  * argmax classifies clean CIFAR-10 images meaningfully above chance,
  * both env call paths work -- the raw [0, 1] tensor used in
    ``EvasionEnv.reset`` and the ImageNet-normalized tensor used in
    ``EvasionEnv.dynamics`` (which the wrapper must denormalize).

Run on a GPU node (see slurm/smoke_qwen3vl.sbatch). Usage:
    python smoke_test_qwen3vl.py --dataset cifar10 --n 16
"""

import argparse

import torch
import torchvision
import torch.nn.functional as F

from qwen3vl_victim import CLASS_NAMES, load_qwen3vl_victim


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL victim smoke test")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--n", type=int, default=16, help="Number of test images")
    parser.add_argument("--data_dir", type=str, default="./data")
    return parser.parse_args()


def main():
    args = parse_args()
    class_names = CLASS_NAMES[args.dataset]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Same raw transform the RL scripts use for the dataset.
    raw_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    if args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=raw_transform
        )
    elif args.dataset == "svhn":
        dataset = torchvision.datasets.SVHN(
            root=args.data_dir, split="test", download=True, transform=raw_transform
        )
    else:
        raise ValueError(f"Unsupported dataset {args.dataset!r}")

    images = torch.stack([dataset[i][0] for i in range(args.n)])  # (n, 3, 32, 32)
    labels = torch.tensor([dataset[i][1] for i in range(args.n)])

    # The env's dynamics preprocess: resize -> center crop -> ImageNet-normalize.
    preprocess = torch.nn.Sequential(
        torchvision.transforms.Resize(256, antialias=True),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    )
    normalized = preprocess(images)  # values escape [0, 1]; wrapper must invert

    print("Loading Qwen3-VL victim (this downloads/loads the checkpoint)...")
    victim = load_qwen3vl_victim(args.dataset)

    for name, batch in (("reset (raw [0,1])", images), ("dynamics (normalized)", normalized)):
        scores = victim(batch.to(device))
        assert scores.shape == (args.n, num_classes), scores.shape
        probs = F.softmax(scores, dim=1)
        assert torch.all(probs >= 0) and torch.all(probs <= 1), "probs out of range"
        preds = scores.argmax(dim=1)
        acc = (preds.cpu() == labels).float().mean().item()

        print(f"\n=== path: {name} ===")
        print(f"scores shape: {tuple(scores.shape)}  accuracy: {acc:.3f} (chance = {1/num_classes:.3f})")
        for i in range(min(args.n, 8)):
            print(
                f"  true={class_names[labels[i]]:<11} "
                f"pred={class_names[preds[i]]:<11} "
                f"p={probs[i, preds[i]].item():.2f}"
            )

    print("\nSmoke test finished.")


if __name__ == "__main__":
    main()
