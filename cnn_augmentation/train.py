"""Train the bird CNN with one of four augmentation settings.

    python train.py --aug none
    python train.py --aug basic
    python train.py --aug robust
    python train.py --aug robust_seen

Saves the model with the best validation mAP to runs/<aug>.pt
"""
import argparse
import os
import random
import time

import numpy as np
import torch

from common import IMG, BirdData, Net, load_split, loss_fn, mean_average_precision, predict, read_image
from corruptions import CORRUPTIONS, corrupt

# "robust_seen" only trains with these; the other corruptions stay unseen,
# so we can test whether robustness carries over to new kinds of damage.
SEEN = ["noise", "dark", "jpeg"]


def flip(img, boxes):
    img = img[:, ::-1].copy()
    boxes[:, [0, 2]] = IMG - boxes[:, [2, 0]]
    return img, boxes


def colour_jitter(img):
    """Random brightness, contrast and saturation (different weather / time of day)."""
    x = img.astype(np.float32)
    x = x * random.uniform(0.7, 1.3)                                  # brightness
    x = (x - x.mean()) * random.uniform(0.7, 1.3) + x.mean()           # contrast
    grey = x.mean(axis=2, keepdims=True)
    x = grey + (x - grey) * random.uniform(0.7, 1.3)                   # saturation
    return np.clip(x, 0, 255).astype(np.uint8)


def make_augment(aug):
    if aug == "none":
        return None
    corruption_list = list(CORRUPTIONS) if aug == "robust" else SEEN

    def augment(img, boxes):
        if random.random() < 0.5:
            img, boxes = flip(img, boxes)
        img = colour_jitter(img)
        # robust settings: half the time, also damage the image with a random corruption
        if aug != "basic" and random.random() < 0.5:
            name = random.choice(corruption_list)
            img = corrupt(img, name, random.randint(1, 5), seed=random.randint(0, 10 ** 9))
        return img, boxes

    return augment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug", default="none", choices=["none", "basic", "robust", "robust_seen"])
    parser.add_argument("--data", default=os.environ.get("BIRD_DATA"), help="dataset folder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--frac", type=float, default=1.0, help="use only this fraction of training images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_items = load_split(args.data, "train")
    random.shuffle(train_items)
    train_items = train_items[:int(len(train_items) * args.frac)]
    val_items = load_split(args.data, "val")
    val_images = [read_image(p) for p, _ in val_items]
    val_boxes = [b for _, b in val_items]

    loader = torch.utils.data.DataLoader(BirdData(train_items, make_augment(args.aug)),
                                         batch_size=args.batch, shuffle=True)
    model = Net()
    torch.nn.init.constant_(model.hm.bias, -2.19)   # start with low bird scores (CenterNet trick)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    os.makedirs("runs", exist_ok=True)
    best_map, start = -1, time.time()
    print(f"training '{args.aug}' on {len(train_items)} images for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for img, *targets in loader:
            loss = loss_fn(model(img), *targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        msg = f"epoch {epoch:2d}  loss {total / len(loader):.3f}  ({(time.time() - start) / 60:.0f} min)"

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val_map = mean_average_precision([predict(model, im) for im in val_images], val_boxes)
            msg += f"  val mAP@0.5 {val_map:.3f}"
            if val_map > best_map:
                best_map = val_map
                torch.save({"state": model.state_dict(), "map": val_map, "aug": args.aug},
                           f"runs/{args.aug}.pt")
        print(msg)

    # store the total training time too (the spec asks us to compare training times)
    ckpt = torch.load(f"runs/{args.aug}.pt", weights_only=False)
    ckpt["train_minutes"] = (time.time() - start) / 60
    torch.save(ckpt, f"runs/{args.aug}.pt")
    print(f"done, best val mAP@0.5 = {best_map:.3f}, saved runs/{args.aug}.pt")


if __name__ == "__main__":
    main()
