"""Tune the Method 1 detectors and test them on clean images and on every corruption at levels 1-5.

    python robustness.py --tune                     # grid search on clean validation images
    python robustness.py --split test               # only once, at the very end!
    python robustness.py --example 0 --corruption dark --level 5 --detectors dog doh

Uses ../cnn/corruptions.py with seed = image index, so the detectors are tested on exactly
the same corrupted images as the CNN. Prints a table and saves results/results_<split>.json
and results/results_<split>.png (mAP vs level). With --example it instead saves a picture
of the predictions on one image.
"""
import argparse
import json
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cnn"))
from corruptions import CORRUPTIONS, corrupt  # noqa: E402  (shared with the CNN)

import detectors as nl  # noqa: E402

PARAMS_FILE = os.path.join(HERE, "best_params.json")
OUT_DIR = os.path.join(HERE, "results")
IMG = 512
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# DoH with box refinement scored best on clean validation images (0.036 vs 0.032), but is
# about 19x slower (2.6 s per image), which is too slow for 36 test conditions, so it is
# left out of the search here.
TOO_SLOW = {"doh": {"refine": True}}


def default_data():
    # same place the CNN code looks, or a "dataset" folder that is the dataset itself
    if os.environ.get("BIRD_DATA"):
        return os.environ["BIRD_DATA"]
    repo = os.path.join(HERE, "..", "..")
    nested = os.path.join(repo, "dataset", "Beach Bird Detection Dataset v1.0")
    return nested if os.path.isdir(nested) else os.path.join(repo, "dataset")


def read_image(path):
    """RGB uint8 image of size 512x512, read the same way as cnn/common.read_image.
    RGB order matters: noise and occlusion colours are drawn per channel."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG, IMG))


def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _detect_one(job):
    detector, params, gray = job
    b, s = nl.DETECTORS[detector](gray, **params)
    return nl.nms(b, s)


def _init_worker():
    cv2.setNumThreads(1)  # one OpenCV thread per process, the processes already use every core


POOL = None


def detect_all(detector, params, grays):
    """Runs one detector + NMS on every image. Returns predictions and seconds per image."""
    jobs = [(detector, params, g) for g in grays]
    start = time.time()
    preds = POOL.map(_detect_one, jobs, chunksize=4) if POOL else [_detect_one(j) for j in jobs]
    return preds, (time.time() - start) / max(len(grays), 1)


def tune(items, detectors):
    """Picks the best setting of each detector on clean validation images."""
    grays = [to_gray(read_image(p)) for p, _ in items]
    gts = [g for _, g in items]
    saved = json.load(open(PARAMS_FILE)) if os.path.exists(PARAMS_FILE) else {}
    for name in detectors:
        best = None
        for params in nl.GRIDS[name]:
            slow = TOO_SLOW.get(name, {})
            if slow and all(params.get(k) == v for k, v in slow.items()):
                continue
            preds, sec = detect_all(name, params, grays)
            ap = nl.average_precision(preds, gts)
            print(f"[{name}] {params} -> val mAP@0.5 = {ap:.3f} ({sec:.2f} s/img)")
            if best is None or ap > best["val_map"]:
                best = {"params": params, "val_map": ap}
        print(f"[{name}] best: {best}\n")
        saved[name] = best
    json.dump(saved, open(PARAMS_FILE, "w"), indent=1)
    print("saved", PARAMS_FILE)


def load_params(detectors):
    if not os.path.exists(PARAMS_FILE):
        raise SystemExit("No best_params.json yet - run `python robustness.py --tune` first.")
    saved = json.load(open(PARAMS_FILE))
    missing = [d for d in detectors if d not in saved]
    if missing:
        raise SystemExit(f"No tuned settings for {missing} - run --tune --detectors {' '.join(missing)}")
    return {d: saved[d]["params"] for d in detectors}


def evaluate(items, detectors, params, split):
    images = [read_image(p) for p, _ in items]
    gts = [g for _, g in items]
    results = {d: {"params": params[d]} for d in detectors}

    print(f"{'':<12}" + "".join(f"{d:>11}" for d in detectors))
    grays = [to_gray(im) for im in images]
    for d in detectors:
        preds, sec = detect_all(d, params[d], grays)
        results[d]["clean"] = nl.average_precision(preds, gts)
        results[d]["ms_per_image"] = sec * 1000
    print(f"{'clean':<12}" + "".join(f"{results[d]['clean']:>11.3f}" for d in detectors))

    for name in CORRUPTIONS:
        for d in detectors:
            results[d][name] = []
        for level in range(1, 6):
            # seed = image index, so every method sees exactly the same corrupted images.
            # Corrupt in RGB (like the CNN), then convert to grey for the detectors.
            grays = [to_gray(corrupt(im, name, level, seed=i)) for i, im in enumerate(images)]
            for d in detectors:
                preds, _ = detect_all(d, params[d], grays)
                results[d][name].append(nl.average_precision(preds, gts))
            print(f"{name + ' ' + str(level):<12}"
                  + "".join(f"{results[d][name][-1]:>11.3f}" for d in detectors))

    print(f"\nmAP@0.5 on {split} ({len(items)} images, corruptions averaged over levels 1-5)")
    print(f"{'detector':<11}{'clean':>7}" + "".join(f"{c:>10}" for c in CORRUPTIONS)
          + f"{'all':>8}{'kept':>7}")
    for d, r in results.items():
        avg = [float(np.mean(r[c])) for c in CORRUPTIONS]
        r["avg_all"] = float(np.mean(avg))
        kept = f"{r['avg_all'] / r['clean']:.0%}" if r["clean"] >= 0.01 else "-"
        print(f"{d:<11}{r['clean']:>7.3f}" + "".join(f"{v:>10.3f}" for v in avg)
              + f"{r['avg_all']:>8.3f}{kept:>7}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"results_{split}")
    json.dump(results, open(out + ".json", "w"), indent=1)

    # one panel per corruption, one line per detector (same layout as the CNN plot)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    for ax, c in zip(axes.flat, CORRUPTIONS):
        for colour, (d, r) in zip(COLOURS, results.items()):
            ax.plot(range(6), [r["clean"]] + r[c], "o-", color=colour, lw=2, ms=4, label=d)
        ax.set_title(c)
        ax.set_xlabel("level (0 = clean)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("mAP@0.5")
    axes[1, 0].set_ylabel("mAP@0.5")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    plt.tight_layout()
    plt.savefig(out + ".png", dpi=150)
    print(f"saved {out}.json and {out}.png")


def show_example(items, index, detectors, params, corruption, level):
    """Green = ground truth, red = the top-scoring predictions."""
    path, gt = items[index]
    img = read_image(path)
    if corruption:
        img = corrupt(img, corruption, level, seed=index)
    gray = to_gray(img)
    fig, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 5.5), squeeze=False)
    for ax, d in zip(axes[0], detectors):
        (boxes, scores), = detect_all(d, params[d], [gray])[0]
        keep = np.argsort(-scores)[:max(len(gt), 10)]
        drawn = img.copy()
        for x1, y1, x2, y2 in gt.astype(int):
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 220, 0), 2)
        for x1, y1, x2, y2 in boxes[keep].astype(int):
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (230, 40, 40), 1)
        ax.imshow(drawn)
        ax.set_title(f"{d}: {len(gt)} birds, top {len(keep)} of {len(boxes)} predictions")
        ax.axis("off")
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"{corruption}{level}" if corruption else "clean"
    out = os.path.join(OUT_DIR, f"example_{index}_{name}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=default_data(), help="dataset folder (contains images/ and labels/)")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--detectors", nargs="+", default=list(nl.DETECTORS), choices=list(nl.DETECTORS))
    parser.add_argument("--tune", action="store_true", help="grid search on clean validation images")
    parser.add_argument("--example", type=int, help="draw the predictions for this image index")
    parser.add_argument("--corruption", choices=list(CORRUPTIONS))
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes (1 = no parallelism)")
    args = parser.parse_args()

    global POOL
    if args.workers > 1 and args.example is None:
        import multiprocessing
        POOL = multiprocessing.Pool(args.workers, initializer=_init_worker)

    if args.tune:
        tune(nl.load_split(args.data, "val"), args.detectors)  # tuning never uses the test set
    elif args.example is not None:
        items = nl.load_split(args.data, args.split)
        show_example(items, args.example, args.detectors, load_params(args.detectors),
                     args.corruption, args.level)
    else:
        items = nl.load_split(args.data, args.split)
        evaluate(items, args.detectors, load_params(args.detectors), args.split)


if __name__ == "__main__":
    main()
