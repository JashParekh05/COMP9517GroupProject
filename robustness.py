"""Robustness of the non-learning detectors to image corruptions.

Uses Jan's cnn_augmentation/corruptions.py, so the non-learning detectors are tested on
exactly the same corrupted pixels as the CNN (same image order, same seed = image index,
same RGB channel order before corrupting).

    python robustness.py --tune                       # 1) pick best params per detector on clean val
    python robustness.py                              # 2) clean + 7 corruptions x 5 levels on val
    python robustness.py --split test                 # 3) final numbers - run ONCE at the very end
    python robustness.py --example 5 --corruption blur --level 4   # picture of predictions

Quick check:  python robustness.py --limit 30 --detectors dog mser
Outputs go to results_nonlearning/.
"""
import argparse
import contextlib
import importlib
import io
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
sys.path.insert(0, os.path.join(HERE, "cnn_augmentation"))
from corruptions import CORRUPTIONS, corrupt  # noqa: E402  (Jan's file)

# "non-learning.py" has a hyphen, so it can't be imported with a normal import statement
nl = importlib.import_module("non-learning")

OUT_DIR = os.path.join(HERE, "results_nonlearning")
PARAMS_FILE = os.path.join(OUT_DIR, "best_params.json")
IMG = 512  # same size as the CNN uses (the dataset images are already 512x512)
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def read_rgb(path):
    """Read exactly like Jan's common.read_image: RGB, 512x512.
    RGB order matters: noise/occlusion colours are drawn per channel, so BGR would give
    different grey images than the ones the CNN sees."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG, IMG))


def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _detect_one(job):
    """Worker: one detector (+ NMS) on one grey image."""
    detector, params, gray = job
    det_params = {k: v for k, v in params.items() if k != "iou_thr"}
    b, s = nl.DETECTORS[detector](gray, **det_params)
    return nl.nms(b, s, params.get("iou_thr", 0.3))


def _init_worker():
    cv2.setNumThreads(1)  # one process per core already; stop OpenCV from also multi-threading


POOL = None  # created in main(); None = run in this process


def detect_all(detector, params, grays):
    """Run one detector (+ NMS) on a list of grey images, spread over all CPU cores.
    Returns preds and wall-clock seconds/image.
    NOTE: ms/image is measured with all cores busy; use --workers 1 for a single-core
    timing to compare against the CNN fairly."""
    jobs = [(detector, params, g) for g in grays]
    start = time.time()
    preds = POOL.map(_detect_one, jobs, chunksize=4) if POOL else [_detect_one(j) for j in jobs]
    return preds, (time.time() - start) / max(len(grays), 1)


def score(preds, gts):
    """mAP@0.5 using Riya's average_precision, with its per-call print silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        return nl.average_precision(preds, gts)


def tune(items, detectors):
    """Grid search (nl.GRIDS) on CLEAN validation images. Never touches test."""
    grays = [to_gray(read_rgb(p)) for p, _ in items]
    gts = [g for _, g in items]
    best = {}
    for name in detectors:
        for params in nl.GRIDS[name]:
            preds, dt = detect_all(name, params, grays)
            ap = score(preds, gts)
            print(f"[{name}] {params} -> val mAP@0.5={ap:.3f} ({dt:.2f}s/img)")
            if name not in best or ap > best[name]["val_map"]:
                best[name] = {"params": params, "val_map": ap}
        print(f"[{name}] BEST: {best[name]}\n")
    os.makedirs(OUT_DIR, exist_ok=True)
    old = json.load(open(PARAMS_FILE)) if os.path.exists(PARAMS_FILE) else {}
    old.update(best)
    json.dump(old, open(PARAMS_FILE, "w"), indent=1)
    print("saved", PARAMS_FILE)


def load_params(detectors):
    if not os.path.exists(PARAMS_FILE):
        raise SystemExit("No best_params.json yet - run `python robustness.py --tune` first.")
    saved = json.load(open(PARAMS_FILE))
    missing = [d for d in detectors if d not in saved]
    if missing:
        raise SystemExit(f"No tuned params for {missing} - run --tune --detectors {' '.join(missing)}")
    return {d: saved[d]["params"] for d in detectors}


def evaluate(items, detectors, params, split):
    rgbs = [read_rgb(p) for p, _ in items]
    gts = [g for _, g in items]
    results = {d: {"params": params[d]} for d in detectors}

    # clean images
    grays = [to_gray(im) for im in rgbs]
    print(f"{'':<11}" + "".join(f"{d:>11}" for d in detectors))
    row = []
    for d in detectors:
        preds, dt = detect_all(d, params[d], grays)
        results[d]["clean"] = score(preds, gts)
        results[d]["ms_per_image"] = dt * 1000
        row.append(results[d]["clean"])
    print(f"{'clean':<11}" + "".join(f"{v:>11.3f}" for v in row))

    # every corruption x level. Corrupt once, then give the same images to every detector.
    for name in CORRUPTIONS:
        for d in detectors:
            results[d][name] = []
        for level in range(1, 6):
            # seed = image index, exactly like Jan's evaluate.py
            grays = [to_gray(corrupt(im, name, level, seed=i)) for i, im in enumerate(rgbs)]
            row = []
            for d in detectors:
                preds, _ = detect_all(d, params[d], grays)
                results[d][name].append(score(preds, gts))
                row.append(results[d][name][-1])
            print(f"{name + ' ' + str(level):<11}" + "".join(f"{v:>11.3f}" for v in row))

    # summary table: clean, each corruption averaged over levels 1-5, overall average
    print(f"\nmAP@0.5 on {split} ({len(items)} images; corruptions averaged over levels 1-5)")
    print(f"{'detector':<11}{'clean':>7}" + "".join(f"{c:>10}" for c in CORRUPTIONS)
          + f"{'all':>8}{'ms/img':>8}")
    for d, r in results.items():
        avg = {c: float(np.mean(r[c])) for c in CORRUPTIONS}
        r["avg_all"] = float(np.mean(list(avg.values())))
        print(f"{d:<11}{r['clean']:>7.3f}" + "".join(f"{avg[c]:>10.3f}" for c in CORRUPTIONS)
              + f"{r['avg_all']:>8.3f}{r['ms_per_image']:>8.1f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"results_{split}")
    json.dump(results, open(out + ".json", "w"), indent=1)

    # same layout as Jan's results plot: one panel per corruption, one line per method
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


def show_example(items, index, detectors, params, corruption, level, top=None):
    """Green = real birds, red = predictions (top-N by score, N = number of real birds)."""
    path, gt = items[index]
    img = read_rgb(path)
    if corruption:
        img = corrupt(img, corruption, level, seed=index)
    gray = to_gray(img)
    fig, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 5.5), squeeze=False)
    for ax, d in zip(axes[0], detectors):
        (boxes, scores), = detect_all(d, params[d], [gray])[0]
        n = top or max(len(gt), 10)
        keep = np.argsort(-scores)[:n]
        drawn = img.copy()
        for x1, y1, x2, y2 in gt.astype(int):
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 220, 0), 2)
        for x1, y1, x2, y2 in boxes[keep].astype(int):
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (230, 40, 40), 1)
        ax.imshow(drawn)
        ax.set_title(f"{d}: {len(gt)} birds, top {len(keep)} of {len(boxes)} preds")
        ax.axis("off")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"example_{index}_{corruption or 'clean'}{level if corruption else ''}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(HERE, "dataset"))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--detectors", nargs="+", default=list(nl.DETECTORS), choices=list(nl.DETECTORS))
    parser.add_argument("--tune", action="store_true", help="grid search on clean val, save best params")
    parser.add_argument("--limit", type=int, help="only use the first N images (quick check)")
    parser.add_argument("--example", type=int, help="draw predictions for this image index")
    parser.add_argument("--corruption", choices=list(CORRUPTIONS))
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="processes to run detectors in (1 = no parallelism)")
    args = parser.parse_args()

    global POOL
    if args.workers > 1 and args.example is None:
        import multiprocessing
        POOL = multiprocessing.Pool(args.workers, initializer=_init_worker)

    if args.tune:
        items = nl.load_split(args.data, "val")[:args.limit]   # tuning always on val
        tune(items, args.detectors)
        return

    items = nl.load_split(args.data, args.split)
    if args.limit:
        items = items[:args.limit]
    params = load_params(args.detectors)
    if args.example is not None:
        show_example(items, args.example, args.detectors, params, args.corruption, args.level)
    else:
        evaluate(items, args.detectors, params, args.split)


if __name__ == "__main__":
    main()
