"""Evaluate one or more checkpoints on clean images and on the corruption benchmark.

Example:
    python evaluate.py --data <DATASET_ROOT> --split val \
        --ckpt runs/baseline_s0/best.pt runs/full_s0/best.pt

For every checkpoint this writes results/<run-name>_<split>.json containing
clean mAP@0.5, mAP@0.5 for every corruption x severity, inference time per image,
and failure-analysis numbers (precision/recall, recall by bird size, count error).

Use --split val while developing; run --split test only once, for the final numbers.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from corruptions import CORRUPTIONS, SEVERITIES, corrupt_deterministic
from data import load_split, read_rgb, to_tensor
from metrics import average_precision, detection_summary
from model import Net, decode, load_checkpoint


@torch.no_grad()
def predict_images(model, images, device, batch=16, thr=0.05, topk=200):
    """images: list of uint8 RGB arrays -> list of (boxes, scores)."""
    model.eval()
    preds = []
    for i in range(0, len(images), batch):
        x = torch.stack([to_tensor(im) for im in images[i:i + batch]]).to(device)
        preds += decode(model(x), thr=thr, topk=topk)
    return preds


def evaluate_model(model, items, device, corruptions=None, severities=SEVERITIES,
                   batch=16, images=None, verbose=True):
    """Clean + corrupted evaluation of one model. `images` may be passed in to avoid re-reading."""
    images = images if images is not None else [read_rgb(p) for p, _ in items]
    gts = [b for _, b in items]

    # warm-up, then time clean inference (per image, includes decoding)
    predict_images(model, images[:batch], device, batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    clean_preds = predict_images(model, images, device, batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) / len(images) * 1000

    res = {"clean": detection_summary(clean_preds, gts), "infer_ms_per_img": infer_ms,
           "corrupted": {}}
    if verbose:
        print(f"  clean                 mAP@0.5 = {res['clean']['ap']:.4f}  ({infer_ms:.1f} ms/img)")

    for name in (corruptions or []):
        res["corrupted"][name] = {}
        for s in severities:
            imgs_c = [corrupt_deterministic(im, name, s, i) for i, im in enumerate(images)]
            preds = predict_images(model, imgs_c, device, batch)
            if s == 3:  # full failure-analysis numbers at the middle severity
                summ = detection_summary(preds, gts)
                res["corrupted"][name][str(s)] = summ["ap"]
                res["corrupted"][name]["summary_s3"] = summ
            else:
                res["corrupted"][name][str(s)] = average_precision(preds, gts)
        if verbose:
            aps = [res["corrupted"][name][str(s)] for s in severities]
            print(f"  {name:<20}  " + "  ".join(f"s{s}={a:.3f}" for s, a in zip(severities, aps)))

    if corruptions:
        per_corr = {c: float(np.mean([res["corrupted"][c][str(s)] for s in severities]))
                    for c in corruptions}
        res["mean_ap_per_corruption"] = per_corr
        res["mean_corrupted_ap"] = float(np.mean(list(per_corr.values())))
        # undefined for a model that does not work on clean images at all
        res["relative_robustness"] = (res["mean_corrupted_ap"] / res["clean"]["ap"]
                                      if res["clean"]["ap"] >= 0.01 else None)
        if verbose:
            rr = res["relative_robustness"]
            print(f"  mean corrupted mAP = {res['mean_corrupted_ap']:.4f}   "
                  f"relative robustness = {'n/a' if rr is None else f'{rr:.3f}'}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("BIRD_DATA"),
                    help="dataset root containing images/ and labels/ (or set BIRD_DATA)")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", help="display names (default: checkpoint's run dir)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS),
                    help="subset of corruptions (pass nothing for clean only)")
    ap.add_argument("--severities", nargs="*", type=int, default=list(SEVERITIES))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (quick check)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.data:
        ap.error("--data is required (or set the BIRD_DATA environment variable)")

    device = torch.device(args.device)
    items = load_split(args.data, args.split)
    if args.limit:
        items = items[:args.limit]
    images = [read_rgb(p) for p, _ in items]
    print(f"{args.split}: {len(items)} images, {sum(len(b) for _, b in items)} birds")
    os.makedirs(args.out, exist_ok=True)

    for k, ck in enumerate(args.ckpt):
        name = (args.names[k] if args.names and k < len(args.names)
                else os.path.basename(os.path.dirname(os.path.abspath(ck))))
        print(f"\n== {name}  ({ck})")
        model = Net().to(device)
        meta = load_checkpoint(model, ck, device)
        res = evaluate_model(model, items, device, args.corruptions, args.severities,
                             args.batch, images=images)
        res.update({"name": name, "ckpt": ck, "split": args.split, "n_images": len(items),
                    "config": meta.get("config") if isinstance(meta, dict) else None,
                    "train_time_s": meta.get("train_time_s") if isinstance(meta, dict) else None})
        path = os.path.join(args.out, f"{name}_{args.split}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
