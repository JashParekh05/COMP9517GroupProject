"""Qualitative figures for the report / video.

    # every corruption at severities 1, 3, 5 on one validation image
    python visualize.py corruptions --data <ROOT> --index 3

    # 8 random training samples after a config's augmentation, with the (transformed) GT boxes
    # -> sanity check that boxes follow the geometric transforms
    python visualize.py augment --data <ROOT> --config full

    # GT (green) vs predictions (red, score >= 0.3) of several models on one corrupted image
    python visualize.py predict --data <ROOT> --index 3 --corruption gaussian_blur --severity 4 \
        --ckpt runs/baseline_s0/best.pt runs/full_s0/best.pt
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from augment import build_augment
from corruptions import CORRUPTIONS, corrupt_deterministic
from data import BirdDataset, load_split, read_rgb
from evaluate import predict_images
from experiments import CONFIGS
from metrics import iou_matrix
from model import Net, load_checkpoint

GT_COLOR, PRED_COLOR = (0, 220, 0), (230, 40, 40)


def draw(img, gts=(), preds=(), thick=1):
    out = img.copy()
    for x1, y1, x2, y2 in np.asarray(gts).reshape(-1, 4).astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), GT_COLOR, thick)
    for x1, y1, x2, y2 in np.asarray(preds).reshape(-1, 4).astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), PRED_COLOR, thick)
    return out


def cmd_corruptions(a):
    items = load_split(a.data, a.split)
    img = read_rgb(items[a.index][0])
    sev = [1, 3, 5]
    names = list(CORRUPTIONS)
    fig, axes = plt.subplots(len(sev), len(names) + 1, figsize=(2 * (len(names) + 1), 2 * len(sev)))
    for r, s in enumerate(sev):
        axes[r, 0].imshow(img)
        axes[r, 0].set_title("clean" if r == 0 else "", fontsize=8)
        axes[r, 0].set_ylabel(f"severity {s}", fontsize=8)
        for c, n in enumerate(names, 1):
            axes[r, c].imshow(corrupt_deterministic(img, n, s, a.index))
            if r == 0:
                axes[r, c].set_title(n.replace("_", " "), fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = os.path.join(a.out, f"corruptions_{a.split}{a.index}.png")
    fig.savefig(path, dpi=150)
    print(path)


def cmd_augment(a):
    items = load_split(a.data, "train")
    ds = BirdDataset(items)
    aug = build_augment(CONFIGS[a.config])
    rng = np.random.default_rng(a.seed)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax in axes.ravel():
        i = int(rng.integers(len(ds)))
        img, boxes = ds.load(i)
        if aug:
            img, boxes = aug(img, boxes, rng, ds)
        ax.imshow(draw(img, boxes))
        ax.set_title(f"#{i}  {len(boxes)} birds", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"augmentation config '{a.config}' (green = transformed GT boxes)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(a.out, f"augment_{a.config}.png")
    fig.savefig(path, dpi=150)
    print(path)


def cmd_predict(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_split(a.data, a.split)
    path, gts = items[a.index]
    img = read_rgb(path)
    if a.corruption:
        img = corrupt_deterministic(img, a.corruption, a.severity, a.index)
    n = len(a.ckpt)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.3), squeeze=False)
    for ax, ck in zip(axes[0], a.ckpt):
        model = Net().to(device)
        load_checkpoint(model, ck, device)
        boxes, scores = predict_images(model, [img], device)[0]
        boxes = boxes[scores >= a.thr]
        tp = int((iou_matrix(gts, boxes).max(1) >= 0.5).sum()) if len(gts) and len(boxes) else 0
        ax.imshow(draw(img, gts, boxes))
        ax.set_title(f"{os.path.basename(os.path.dirname(os.path.abspath(ck)))}\n"
                     f"GT {len(gts)}  pred {len(boxes)}  found {tp}", fontsize=9)
        ax.axis("off")
    tag = f"{a.corruption}_s{a.severity}" if a.corruption else "clean"
    fig.suptitle(f"{os.path.basename(path)}  [{tag}]  green = GT, red = prediction (score >= {a.thr})",
                 fontsize=9)
    fig.tight_layout()
    out = os.path.join(a.out, f"pred_{a.split}{a.index}_{tag}.png")
    fig.savefig(out, dpi=150)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["corruptions", "augment", "predict"])
    ap.add_argument("--data", default=os.environ.get("BIRD_DATA"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--config", default="full", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--corruption", choices=list(CORRUPTIONS))
    ap.add_argument("--severity", type=float, default=3)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    if not a.data:
        ap.error("--data is required (or set BIRD_DATA)")
    os.makedirs(a.out, exist_ok=True)
    {"corruptions": cmd_corruptions, "augment": cmd_augment, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
