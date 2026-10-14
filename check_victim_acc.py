"""Measure a victim checkpoint's real CIFAR-10 test accuracy.

Uses the SAME val transform as train_victim_model.py (Resize 256 -> CenterCrop
224 -> ImageNet normalize). Usage:
    python check_victim_acc.py --model vit_b_16 --dataset cifar10
"""

import argparse
import torch
import torchvision
import torchvision.transforms as transforms

p = argparse.ArgumentParser()
p.add_argument("--model", default="vit_b_16")
p.add_argument("--dataset", default="cifar10")
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--limit", type=int, default=0, help="0 = full test set")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

if args.dataset == "cifar10":
    ds = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=val_tf)
elif args.dataset == "svhn":
    ds = torchvision.datasets.SVHN(root="./dataset", split="test", download=True, transform=val_tf)
else:
    raise ValueError(args.dataset)

if args.limit > 0:
    ds = torch.utils.data.Subset(ds, list(range(args.limit)))

loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

if args.model == "resnet50":
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
elif args.model == "vgg16":
    model = torchvision.models.vgg16(weights=None)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 10)
elif args.model == "vit_b_16":
    model = torchvision.models.vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 10)
else:
    raise ValueError(args.model)

ckpt = "./victim_models/{}+{}.pth".format(args.dataset, args.model)
sd = torch.load(ckpt, map_location=device, weights_only=False)
print(f"checkpoint: {ckpt}")
print(f"  saved epoch={sd.get('epoch')} best_acc={sd.get('best_acc')}")
model.load_state_dict(sd["model_state_dict"])
model.to(device).eval()

correct = total = 0
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()

acc = correct / total
print(f"\nMEASURED test accuracy: {acc:.4f} ({correct}/{total})")
print("VERDICT:", "victim is REAL (>=90%)" if acc >= 0.90 else
      f"victim is UNDERTRAINED ({acc*100:.1f}% < 90%)")
