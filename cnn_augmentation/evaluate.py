"""Test trained models on clean images and on every corruption at levels 1-5.

    python evaluate.py runs/none.pt runs/basic.pt runs/robust.pt runs/robust_seen.pt
    python evaluate.py runs/*.pt --split test              # only once, at the very end!
    python evaluate.py runs/none.pt runs/robust.pt --example 5 --corruption blur --level 4

Prints a table and saves results_<split>.json and results_<split>.png (mAP vs level).
With --example it instead saves a picture of the models' predictions on one image.
"""
import argparse
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_DATA, load_model, load_split, mean_average_precision, predict, read_image
from corruptions import CORRUPTIONS, corrupt
from train import SEEN

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def evaluate(model, images, boxes):
    """Returns {"clean": mAP, "blur": [mAP level1..5], ...} and inference ms per image."""
    start = time.time()
    result = {"clean": mean_average_precision(predict(model, images), boxes)}
    ms = (time.time() - start) / len(images) * 1000
    for name in CORRUPTIONS:
        result[name] = []
        for level in range(1, 6):
            # seed = image index, so every model sees exactly the same corrupted images
            corrupted = [corrupt(im, name, level, seed=i) for i, im in enumerate(images)]
            result[name].append(mean_average_precision(predict(model, corrupted), boxes))
        print(f"   {name:<10} " + "  ".join(f"{v:.3f}" for v in result[name]))
    return result, ms


def show_example(models, item, index, corruption, level):
    path, gt = item
    img = read_image(path)
    if corruption:
        img = corrupt(img, corruption, level, seed=index)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.5), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        boxes, scores = predict(model, [img])[0]
        drawn = img.copy()
        for x1, y1, x2, y2 in gt.astype(int):
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 220, 0), 2)        # green = real bird
        for x1, y1, x2, y2 in boxes[scores > 0.3].astype(int):
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (230, 40, 40), 1)      # red = prediction
        ax.imshow(drawn)
        ax.set_title(f"{name}: {len(gt)} birds, {(scores > 0.3).sum()} predicted")
        ax.axis("off")
    out = f"example_{index}_{corruption or 'clean'}{level if corruption else ''}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data", default=DEFAULT_DATA, help="dataset folder")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--example", type=int, help="only draw predictions for this image index")
    parser.add_argument("--corruption", choices=list(CORRUPTIONS))
    parser.add_argument("--level", type=int, default=3)
    args = parser.parse_args()

    items = load_split(args.data, args.split)
    models = {os.path.splitext(os.path.basename(c))[0]: load_model(c) for c in args.checkpoints}

    if args.example is not None:
        show_example(models, items[args.example], args.example, args.corruption, args.level)
        return

    images = [read_image(p) for p, _ in items]
    boxes = [b for _, b in items]
    results = {}
    for name, model in models.items():
        print(f"\n{name}")
        results[name], ms = evaluate(model, images, boxes)
        results[name]["ms_per_image"] = ms

    # ---- table
    unseen = [c for c in CORRUPTIONS if c not in SEEN]
    print(f"\nmAP@0.5 on {args.split} (corruptions: average over levels 1-5)")
    print(f"{'model':<14}{'clean':>7}" + "".join(f"{c:>10}" for c in CORRUPTIONS)
          + f"{'all':>8}{'unseen':>8}{'ms/img':>8}")
    for name, r in results.items():
        avg = {c: np.mean(r[c]) for c in CORRUPTIONS}
        r["avg_all"] = float(np.mean(list(avg.values())))
        r["avg_unseen"] = float(np.mean([avg[c] for c in unseen]))
        print(f"{name:<14}{r['clean']:>7.3f}" + "".join(f"{avg[c]:>10.3f}" for c in CORRUPTIONS)
              + f"{r['avg_all']:>8.3f}{r['avg_unseen']:>8.3f}{r['ms_per_image']:>8.1f}")
    print(f"'unseen' = average over {unseen} (never used when training robust_seen)")

    with open(f"results_{args.split}.json", "w") as f:
        json.dump(results, f, indent=1)

    # ---- plot: one small chart per corruption, one line per model
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    for ax, c in zip(axes.flat, CORRUPTIONS):
        for colour, (name, r) in zip(COLOURS, results.items()):
            ax.plot(range(6), [r["clean"]] + r[c], "o-", color=colour, lw=2, ms=4, label=name)
        ax.set_title(c + ("" if c in SEEN else " (unseen)"))
        ax.set_xlabel("level (0 = clean)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("mAP@0.5")
    axes[1, 0].set_ylabel("mAP@0.5")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    plt.tight_layout()
    plt.savefig(f"results_{args.split}.png", dpi=150)
    print(f"saved results_{args.split}.json and results_{args.split}.png")


if __name__ == "__main__":
    main()
