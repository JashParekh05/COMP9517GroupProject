"""Train the bird CNN with a named augmentation config.

Example:
    python train.py --data <DATASET_ROOT> --config full --seed 0
    python train.py --data <DATASET_ROOT> --config baseline --seed 0 --init tune-512-clean.pt

Writes runs/<config>_s<seed>/best.pt (best clean val mAP@0.5), last.pt and history.json.
Checkpoints use the same {"state": ..., "map": ...} format as the original notebook.
Model selection uses only the *clean validation* set; the test set is never touched here.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from augment import build_augment
from data import BirdDataset, load_split, read_rgb
from evaluate import predict_images
from experiments import CONFIGS
from metrics import average_precision
from model import Net, detection_loss, load_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("BIRD_DATA"),
                    help="dataset root containing images/ and labels/ (or set BIRD_DATA)")
    ap.add_argument("--config", default="baseline", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--init", default=None, help="optional checkpoint to fine-tune from")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="use a random fraction of the training images (data-scarcity study)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None, help="run name (default <config>_s<seed>)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.data:
        ap.error("--data is required (or set the BIRD_DATA environment variable)")

    set_seed(args.seed)
    device = torch.device(args.device)
    name = args.name or f"{args.config}_s{args.seed}" + \
        (f"_frac{args.train_frac:g}" if args.train_frac < 1 else "")
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)

    train_items = load_split(args.data, "train")
    if args.train_frac < 1:
        idx = np.random.default_rng(args.seed).permutation(len(train_items))
        train_items = [train_items[i] for i in sorted(idx[:int(len(idx) * args.train_frac)])]
    val_items = load_split(args.data, "val")
    val_images = [read_rgb(p) for p, _ in val_items]
    val_gts = [b for _, b in val_items]

    cfg = CONFIGS[args.config]
    ds = BirdDataset(train_items, augment=build_augment(cfg))
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, generator=g, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)

    model = Net()
    if args.init:
        load_checkpoint(model, args.init)
        print(f"initialised from {args.init}")
    else:
        model.init_for_training()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(loader)
    warm = min(len(loader), steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm
        else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(steps - warm, 1))))

    print(f"run {name}: config={cfg}  train={len(ds)}  val={len(val_items)}  device={device}")
    history, best_map, t_start = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep, sums, n = time.time(), {"loss": 0, "hm": 0, "wh": 0, "off": 0}, 0
        for x, tgt in loader:
            x = x.to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}
            loss, parts = detection_loss(model(x), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            sums["loss"] += loss.item()
            for k, v in parts.items():
                sums[k] += v
            n += 1
        rec = {"epoch": epoch, **{k: v / max(n, 1) for k, v in sums.items()},
               "epoch_s": time.time() - t_ep}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            preds = predict_images(model, val_images, device)
            rec["val_map"] = average_precision(preds, val_gts)
            if rec["val_map"] > best_map:
                best_map = rec["val_map"]
                torch.save({"state": model.state_dict(), "map": best_map, "epoch": epoch,
                            "config": args.config, "cfg": cfg, "args": vars(args),
                            "train_time_s": time.time() - t_start},
                           os.path.join(run_dir, "best.pt"))
        history.append(rec)
        print(f"ep {epoch:3d}  loss {rec['loss']:.3f} (hm {rec['hm']:.3f} wh {rec['wh']:.3f} "
              f"off {rec['off']:.3f})  {rec['epoch_s']:.0f}s"
              + (f"  val mAP@0.5 {rec['val_map']:.4f} (best {best_map:.4f})" if "val_map" in rec else ""))

    total = time.time() - t_start
    torch.save({"state": model.state_dict(), "map": history[-1].get("val_map"),
                "epoch": args.epochs, "config": args.config, "cfg": cfg,
                "args": vars(args), "train_time_s": total}, os.path.join(run_dir, "last.pt"))
    # record the *total* training time in best.pt as well (the spec asks to compare training times)
    best_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(best_path):
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best["train_time_s"] = total
        torch.save(best, best_path)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"name": name, "config": args.config, "cfg": cfg, "args": vars(args),
                   "best_val_map": best_map, "train_time_s": total, "history": history}, f, indent=1)
    print(f"done in {total / 60:.1f} min, best val mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()
