"""Simple lecture CNN for beach-bird detection (COMP9517 W7).

Story (not YOLO):
  - A small Conv→BN→ReLU encoder learns features from the beach image.
  - The network outputs a spatial *bird-score map* (heatmap) plus box size
    at each location — the same idea as reading activation maps in class.
  - Peaks on the score map become detections; IoU-NMS is optional cleanup.

Best clean checkpoint: trials/runs/tune-512-clean.pt  (val mAP@0.5 ≈ 0.252)

Usage:
  python detector.py train --epochs 12 --input 512 --scale-aug --save runs/simple-cnn.pt
  python detector.py predict --split val --out preds-val.json --weights runs/tune-512-clean.pt
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "dataset" / "beach-bird" / "Beach Bird Detection Dataset v1.0"
SRC = 512
IN = 256
OUT = 64  # heatmap size; one cell = IN/OUT input pixels
STRIDE = IN // OUT


def load_split(split):
    """[(path, boxes[N,4] x,y,w,h in IN-pixel coords)]"""
    out = []
    scale = IN / SRC
    for lbl in sorted((ROOT / "labels" / split).glob("*.txt")):
        rows = []
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * SRC, float(cy) * SRC, float(w) * SRC, float(h) * SRC
            rows.append([(cx - w / 2) * scale, (cy - h / 2) * scale, w * scale, h * scale])
        out.append((ROOT / "images" / split / (lbl.stem + ".jpg"), np.float32(rows).reshape(-1, 4)))
    return out


def use_input(size):
    global IN, OUT, STRIDE
    if size % 8 != 0:
        raise ValueError(f"input {size} not divisible by 8")
    IN = size
    OUT = size // 4  # 4 stride-2 enc layers → /16, then 2× upsample → /4
    STRIDE = IN // OUT


class Net(nn.Module):
    """Small CNN: encode with Conv-BN-ReLU (stride 2), upsample, score/size/offset heads."""

    def __init__(self, width=1.0):
        super().__init__()
        c1, c2, c3, c4 = (max(8, int(round(c * width))) for c in (16, 32, 64, 128))

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 2, 1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.enc = nn.Sequential(blk(3, c1), blk(c1, c2), blk(c2, c3), blk(c3, c4))
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c4, c3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c3, c2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.hm = nn.Conv2d(c2, 1, 1)
        self.wh = nn.Conv2d(c2, 2, 1)
        self.off = nn.Conv2d(c2, 2, 1)
        nn.init.constant_(self.hm.bias, -2.0)

    def forward(self, x):
        f = self.dec(self.enc(x))
        return self.hm(f), self.wh(f), self.off(f)


def gaussian_radius(det_size, min_overlap=0.7):
    h, w = det_size
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = max(b1 ** 2 - 4 * a1 * c1, 0)
    r1 = (b1 + math.sqrt(sq1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    sq2 = max(b2 ** 2 - 4 * a2 * c2, 0)
    r2 = (b2 + math.sqrt(sq2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    sq3 = max(b3 ** 2 - 4 * a3 * c3, 0)
    r3 = (b3 + math.sqrt(sq3)) / 2
    return max(0, min(r1, r2, r3))


def draw_gaussian(hm, center, radius):
    radius = int(radius)
    diameter = 2 * radius + 1
    if diameter <= 0:
        return
    x, y = int(center[0]), int(center[1])
    h, w = hm.shape
    sigma = diameter / 6
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    gauss = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    l, t = min(x, radius), min(y, radius)
    r, b = min(w - x, radius + 1), min(h - y, radius + 1)
    if r <= 0 or b <= 0 or l < 0 or t < 0:
        return
    masked = hm[y - t:y + b, x - l:x + r]
    g = gauss[radius - t:radius + b, radius - l:radius + r]
    np.maximum(masked, g, out=masked)


def make_targets(boxes, log_wh=True):
    hm = np.zeros((1, OUT, OUT), np.float32)
    wh = np.zeros((2, OUT, OUT), np.float32)
    off = np.zeros((2, OUT, OUT), np.float32)
    mask = np.zeros((1, OUT, OUT), np.float32)
    for x, y, w, h in boxes:
        cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < OUT and 0 <= iy < OUT):
            continue
        radius = max(gaussian_radius((h / STRIDE, w / STRIDE)), 1)
        draw_gaussian(hm[0], (ix, iy), radius)
        mask[0, iy, ix] = 1.0
        if log_wh:
            # fraction of the input image (matches tune-512-clean.pt)
            wh[0, iy, ix] = math.log(max(w / IN, 1e-4))
            wh[1, iy, ix] = math.log(max(h / IN, 1e-4))
        else:
            wh[0, iy, ix] = w
            wh[1, iy, ix] = h
        off[0, iy, ix] = cx - ix
        off[1, iy, ix] = cy - iy
    return hm, wh, off, mask


def read_image(path, flip=False):
    im = np.asarray(Image.open(path).convert("RGB").resize((IN, IN), Image.BILINEAR), np.float32) / 255.0
    x = torch.from_numpy(im).permute(2, 0, 1)
    return torch.flip(x, [-1]) if flip else x


def jitter(x, boxes):
    if np.random.rand() < 0.5:
        x = torch.flip(x, [-1])
        if len(boxes):
            boxes = np.stack([IN - boxes[:, 0] - boxes[:, 2], boxes[:, 1], boxes[:, 2], boxes[:, 3]], 1)
    scale = float(np.random.uniform(0.85, 1.15))
    nh = max(int(round(IN * scale)), 1)
    y = F.interpolate(x[None], size=(nh, nh), mode="bilinear", align_corners=False)[0]
    if len(boxes):
        boxes = np.float32(boxes) * scale
    if nh >= IN:
        top = int(np.random.randint(0, nh - IN + 1))
        left = int(np.random.randint(0, nh - IN + 1))
        y = y[:, top:top + IN, left:left + IN]
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] -= left
            boxes[:, 1] -= top
    else:
        pad = IN - nh
        top = left = pad // 2
        y = F.pad(y, (left, pad - left, top, pad - top), mode="reflect")
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] += left
            boxes[:, 1] += top
    if len(boxes):
        cx, cy = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
        keep = (cx >= 0) & (cx < IN) & (cy >= 0) & (cy < IN) & (boxes[:, 2] > 1) & (boxes[:, 3] > 1)
        boxes = boxes[keep]
    return y, np.float32(boxes).reshape(-1, 4)


def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """RetinaNet / peak-map focal loss (lecture W8 idea for class imbalance)."""
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_w = torch.pow(1 - gt, beta)
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos_l = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_l = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_w * neg
    n = pos.sum().clamp(min=1)
    return (pos_l.sum() + neg_l.sum()) / n


def box_iou(a, b):
    aw, ah, bw, bh = a[2], a[3], b[2], b[3]
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + aw, b[0] + bw), min(a[1] + ah, b[1] + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    den = aw * ah + bw * bh - inter
    return inter / den if den > 0 else 0.0


def score_detections(records, iou_thresh=0.5):
    flat, n_gt, tp = [], 0, 0
    for preds, gts in records:
        n_gt += len(gts)
        used = [False] * len(gts)
        for box, sc in sorted(preds, key=lambda t: -t[1]):
            best, best_iou = -1, iou_thresh
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = box_iou(box, g)
                if v >= best_iou:
                    best, best_iou = j, v
            if best >= 0:
                used[best] = True
                tp += 1
            flat.append((sc, best >= 0))
    if n_gt == 0:
        return 1.0, 1.0
    if not flat:
        return 0.0, 0.0
    flat.sort(key=lambda t: -t[0])
    prec, seen = [], 0
    for i, (_sc, hit) in enumerate(flat):
        seen += hit
        prec.append(seen / (i + 1))
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = sum(p for (_sc, hit), p in zip(flat, prec) if hit) / n_gt
    return ap, tp / n_gt


def decode(hm, wh, off, thr=0.25, topk=100, log_wh=True):
    """Local peaks on the bird-score map → boxes."""
    # 3×3 max-pool peak extract (same idea as NMS on the score map)
    if hm.dim() == 4:
        hm_2d, wh_b, off_b = hm[0, 0], wh[0], off[0]
    else:
        hm_2d, wh_b, off_b = hm[0] if hm.dim() == 3 else hm, wh, off
    pad = F.max_pool2d(hm_2d[None, None], 3, 1, 1)[0, 0]
    peaks = (hm_2d == pad) & (hm_2d > thr)
    ys, xs = torch.where(peaks)
    if len(xs) == 0:
        return torch.zeros(0, 4), torch.zeros(0)
    scores = hm_2d[ys, xs]
    if scores.numel() > topk:
        scores, idx = torch.topk(scores, topk)
        ys, xs = ys[idx], xs[idx]
    ox = off_b[0, ys, xs]
    oy = off_b[1, ys, xs]
    ww = wh_b[0, ys, xs]
    hh = wh_b[1, ys, xs]
    if log_wh:
        ww, hh = ww.exp() * IN, hh.exp() * IN
    cx = (xs.float() + ox) * STRIDE
    cy = (ys.float() + oy) * STRIDE
    boxes = torch.stack([cx - ww / 2, cy - hh / 2, ww, hh], 1)
    return boxes, scores


@torch.no_grad()
def evaluate(model, val, tensors, log_wh=True, thr=0.1):
    model.eval()
    records = []
    for s in range(0, len(val), 4):
        x = torch.stack(tensors[s:s + 4])
        hm, wh, off = model(x)
        hm = hm.sigmoid()
        for i in range(x.shape[0]):
            boxes, scores = decode(hm[i:i + 1], wh[i:i + 1], off[i:i + 1], thr=thr, log_wh=log_wh)
            records.append((list(zip(boxes.tolist(), scores.tolist())), val[s + i][1]))
    return score_detections(records)


def train(epochs, lr, batch, scale_aug, width, in_size, save, seed, init):
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_input(in_size)
    data = load_split("train")
    val = load_split("val")
    print(f"caching images at {IN}px ...", flush=True)
    train_x = [read_image(p) for p, _ in data]
    val_x = [read_image(p) for p, _ in val]
    model = Net(width=width)
    log_wh = True
    best, best_ep = -1.0, 0
    if init:
        init_path = Path(init)
        if not init_path.is_absolute():
            init_path = Path(__file__).resolve().parent / init_path
        blob = torch.load(init_path, weights_only=False, map_location="cpu")
        model.load_state_dict(blob["state"] if isinstance(blob, dict) and "state" in blob else blob)
        if isinstance(blob, dict):
            log_wh = bool(blob.get("log_wh", True))
            if "map" in blob:
                best = float(blob["map"])
                best_ep = int(blob.get("epoch", 0))
        print(f"loaded {init_path}  bar mAP {best:.4f}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    out = Path(save)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train {len(data)}  val {len(val)}  epochs {epochs}  batch {batch}  "
          f"input {IN}  out {OUT}  params {n_params:,}", flush=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(data))
        tot = 0.0
        n = 0
        for s in range(0, len(order), batch):
            xs, hms, whs, offs, masks = [], [], [], [], []
            for i in order[s:s + batch]:
                x, b = train_x[i], data[i][1].copy()
                if scale_aug:
                    x, b = jitter(x, b)
                elif np.random.rand() < 0.5:
                    x = torch.flip(x, [-1])
                    if len(b):
                        b = np.stack([IN - b[:, 0] - b[:, 2], b[:, 1], b[:, 2], b[:, 3]], 1)
                hm, wh, off, m = make_targets(b, log_wh=log_wh)
                xs.append(x)
                hms.append(torch.from_numpy(hm))
                whs.append(torch.from_numpy(wh))
                offs.append(torch.from_numpy(off))
                masks.append(torch.from_numpy(m))
            x = torch.stack(xs)
            hm_t, wh_t, off_t, m_t = map(torch.stack, (hms, whs, offs, masks))
            p_hm, p_wh, p_off = model(x)
            loss_hm = focal_loss(p_hm.sigmoid(), hm_t)
            loss_wh = (F.l1_loss(p_wh, wh_t, reduction="none") * m_t).sum() / m_t.sum().clamp(min=1)
            loss_off = (F.l1_loss(p_off, off_t, reduction="none") * m_t).sum() / m_t.sum().clamp(min=1)
            loss = loss_hm + 0.1 * loss_wh + loss_off
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        ap, rec = evaluate(model, val, val_x, log_wh=log_wh)
        mark = ""
        if ap > best:
            best, best_ep = ap, ep
            torch.save(
                {"state": model.state_dict(), "width": width, "input": IN, "map": ap, "epoch": ep,
                 "log_wh": log_wh, "deep": False, "wh_loss": "log"},
                out,
            )
            mark = "  saved"
        print(f"epoch {ep:2d}  train {tot / max(n, 1):.3f}  val mAP {ap:.4f}  "
              f"recall {rec:.3f}  {time.time() - t0:.0f}s{mark}", flush=True)
    print(f"best val mAP {best:.4f} at epoch {best_ep}  -> {out.name}")


def load_model(path):
    blob = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(blob, dict) and "state" in blob:
        use_input(int(blob.get("input", 256)))
        model = Net(width=float(blob.get("width", 1.0)))
        model.load_state_dict(blob["state"])
        log_wh = bool(blob.get("log_wh", True))
        meta_map = float(blob.get("map", -1))
    else:
        use_input(256)
        model = Net()
        model.load_state_dict(blob)
        log_wh, meta_map = True, -1
    model.eval()
    return model, log_wh, meta_map


def predict(split, out_path, weights, thr=0.3):
    model, log_wh, _ = load_model(weights)
    preds = []
    with torch.no_grad():
        for path, _ in load_split(split):
            hm, wh, off = model(read_image(path).unsqueeze(0))
            boxes, scores = decode(hm.sigmoid(), wh, off, thr=thr, log_wh=log_wh)
            k = SRC / IN
            preds.append({
                "file": path.name,
                "preds": [{"box": [round(float(v) * k, 2) for v in b], "score": round(float(s), 6)}
                          for b, s in zip(boxes, scores)],
            })
    Path(out_path).write_text(json.dumps({"split": split, "predictions": preds}))
    print(f"wrote {out_path}: {sum(len(p['preds']) for p in preds)} preds over {len(preds)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--scale-aug", action="store_true")
    t.add_argument("--width", type=float, default=1.0)
    t.add_argument("--input", type=int, default=256)
    t.add_argument("--save", default="runs/simple-cnn.pt")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init", default="")
    p = sub.add_parser("predict")
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default="runs/tune-512-clean.pt")
    p.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.epochs, a.lr, a.batch, a.scale_aug, a.width, a.input, a.save, a.seed, a.init)
    else:
        predict(a.split, a.out, a.weights, a.thr)
