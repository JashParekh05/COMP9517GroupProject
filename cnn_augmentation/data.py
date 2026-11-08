"""Dataset loading and CenterNet-style target encoding for the bird CNN.

Boxes are handled internally as float32 arrays of shape [N, 4] in
(x1, y1, x2, y2) pixel coordinates -- the same convention as the
non-learning branch, so predictions/metrics are directly comparable.
"""
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 512   # every image in Beach Bird Detection v1.0 is 512x512
STRIDE = 4       # the network outputs a 128x128 map (IMG_SIZE / 4)
OUT_SIZE = IMG_SIZE // STRIDE
MAX_OBJS = 256   # max birds per image in the dataset is 130; mosaic can combine images


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_split(root, split):
    """Return a list of (image_path, boxes[N,4] x1y1x2y2 in pixels).

    Iterates over the *images* so that negative images (no label file or an
    empty one) are included.
    """
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    items = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(name)[0]
        boxes = []
        lbl = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lbl):
            with open(lbl) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, w, h = map(float, parts)
                    # YOLO format is normalised (cx, cy, w, h)
                    boxes.append([(cx - w / 2) * IMG_SIZE, (cy - h / 2) * IMG_SIZE,
                                  (cx + w / 2) * IMG_SIZE, (cy + h / 2) * IMG_SIZE])
        items.append((os.path.join(img_dir, name),
                      np.array(boxes, dtype=np.float32).reshape(-1, 4)))
    return items


def read_rgb(path):
    """Read an image as uint8 RGB, resized to IMG_SIZE if needed."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return img


def to_tensor(img):
    """uint8 HxWx3 RGB -> float 3xHxW in [0, 1] (same as the original notebook)."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


# --------------------------------------------------------------------------
# Target encoding (CenterNet, Zhou et al. 2019 "Objects as Points")
# --------------------------------------------------------------------------
def gaussian_radius(h, w, min_overlap=0.7):
    """Radius such that a box shifted by r still has IoU >= min_overlap.

    Taken from the CenterNet reference implementation (Zhou et al., 2019).
    h, w are in output-map cells.
    """
    a1, b1 = 1, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap, cx, cy, radius):
    """Splat a 2D Gaussian (peak = 1) onto heatmap at integer (cx, cy), in place, using max."""
    d = 2 * radius + 1
    sigma = d / 6
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    H, W = heatmap.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    region = heatmap[cy - t:cy + b, cx - l:cx + r]
    patch = g[radius - t:radius + b, radius - l:radius + r]
    np.maximum(region, patch, out=region)


def encode_targets(boxes):
    """Turn x1y1x2y2 pixel boxes into CenterNet training targets.

    hm   [1, OUT, OUT]  Gaussian heatmap of bird centres
    wh   [MAX_OBJS, 2]  log(box size / IMG_SIZE)   (decoded as exp(wh) * IMG_SIZE,
                                                     matching the original notebook)
    off  [MAX_OBJS, 2]  sub-cell centre offset in [0, 1)
    ind  [MAX_OBJS]     flat index of the centre cell
    mask [MAX_OBJS]     1 for valid entries
    """
    hm = np.zeros((OUT_SIZE, OUT_SIZE), np.float32)
    wh = np.zeros((MAX_OBJS, 2), np.float32)
    off = np.zeros((MAX_OBJS, 2), np.float32)
    ind = np.zeros(MAX_OBJS, np.int64)
    mask = np.zeros(MAX_OBJS, np.float32)
    for k, (x1, y1, x2, y2) in enumerate(boxes[:MAX_OBJS]):
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
        ix = int(np.clip(cx, 0, OUT_SIZE - 1))
        iy = int(np.clip(cy, 0, OUT_SIZE - 1))
        radius = max(0, int(gaussian_radius(h / STRIDE, w / STRIDE)))
        draw_gaussian(hm, ix, iy, radius)
        wh[k] = np.log([w / IMG_SIZE, h / IMG_SIZE])
        off[k] = (cx - ix, cy - iy)
        ind[k] = iy * OUT_SIZE + ix
        mask[k] = 1.0
    return {
        "hm": torch.from_numpy(hm)[None],
        "wh": torch.from_numpy(wh),
        "off": torch.from_numpy(off),
        "ind": torch.from_numpy(ind),
        "mask": torch.from_numpy(mask),
    }


# --------------------------------------------------------------------------
# Torch dataset
# --------------------------------------------------------------------------
class BirdDataset(Dataset):
    """Training dataset. `augment` is a callable (img, boxes, rng, dataset) -> (img, boxes)."""

    def __init__(self, items, augment=None):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def load(self, i):
        path, boxes = self.items[i]
        return read_rgb(path), boxes.copy()

    def __getitem__(self, i):
        img, boxes = self.load(i)
        if self.augment is not None:
            # Draw the per-sample seed from torch's RNG: DataLoader seeds every
            # worker from the main generator, so runs are reproducible given --seed.
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
            img, boxes = self.augment(img, boxes, rng, self)
        return to_tensor(img), encode_targets(boxes)
