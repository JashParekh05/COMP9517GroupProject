"""Training-time data augmentation for the bird detector.

Every op has the signature  op(img uint8 HxWx3, boxes [N,4] x1y1x2y2, rng) -> (img, boxes)
and geometric ops transform the boxes together with the image.

`build_augment(cfg)` turns a config dict (see experiments.py) into one callable
used by data.BirdDataset. Pipeline order:

    mosaic -> horizontal flip -> scale/translate -> cutout -> photometric -> corruption
"""
import cv2
import numpy as np

import corruptions as C
from data import IMG_SIZE

FILL = (114, 114, 114)  # neutral grey for padded regions (as in YOLO)


def _clip_filter(boxes, orig_area, W, H, min_visible=0.4, min_side=2.0):
    """Clip boxes to the image and drop those that are now mostly outside / tiny."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    keep = (w >= min_side) & (h >= min_side) & (w * h >= min_visible * orig_area)
    return b[keep]


# --------------------------------------------------------------------------
# Geometric
# --------------------------------------------------------------------------
def hflip(img, boxes, rng, p=0.5):
    if rng.random() >= p:
        return img, boxes
    W = img.shape[1]
    img = img[:, ::-1].copy()
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
    return img, boxes


def scale_translate(img, boxes, rng, scale=(0.75, 1.5), p=0.8):
    """Random zoom in/out + translation (keeps output size). Simulates birds
    at different distances from the camera. Areas outside the source are grey."""
    if rng.random() >= p:
        return img, boxes
    H, W = img.shape[:2]
    s = rng.uniform(*scale)
    lo_x, hi_x = sorted([0.0, W - W * s])
    lo_y, hi_y = sorted([0.0, H - H * s])
    tx, ty = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
    M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=FILL)
    if len(boxes):
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * s * s
        b = boxes * s
        b[:, [0, 2]] += tx
        b[:, [1, 3]] += ty
        boxes = _clip_filter(b, area, W, H)
    return img, boxes


def mosaic(dataset, rng, tile=IMG_SIZE // 2):
    """4-image mosaic (Bochkovskiy et al., YOLOv4) *without rescaling*: a random
    tile x tile crop of 4 random training images is placed in each quadrant.
    Cropping instead of shrinking keeps bird sizes realistic (they are already tiny),
    while increasing scene variety and bird density per sample."""
    out = np.empty((2 * tile, 2 * tile, 3), np.uint8)
    all_boxes = []
    for q in range(4):
        img, boxes = dataset.load(int(rng.integers(len(dataset))))
        H, W = img.shape[:2]
        x0, y0 = int(rng.integers(0, W - tile + 1)), int(rng.integers(0, H - tile + 1))
        ox, oy = (q % 2) * tile, (q // 2) * tile
        out[oy:oy + tile, ox:ox + tile] = img[y0:y0 + tile, x0:x0 + tile]
        if len(boxes):
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            b = boxes.copy()
            b[:, [0, 2]] -= x0
            b[:, [1, 3]] -= y0
            b = _clip_filter(b, area, tile, tile)
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            all_boxes.append(b)
    boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
    return out, boxes.astype(np.float32)


# --------------------------------------------------------------------------
# Occlusion
# --------------------------------------------------------------------------
def cutout(img, boxes, rng, n=(1, 6), size=(16, 80), p=0.7, drop_covered=0.7):
    """Cutout / Random Erasing (DeVries & Taylor 2017; Zhong et al. 2020).
    Birds that end up >= drop_covered hidden are removed from the targets."""
    if rng.random() >= p:
        return img, boxes
    img = img.copy()
    H, W = img.shape[:2]
    mask = np.zeros((H, W), bool)
    for _ in range(int(rng.integers(n[0], n[1] + 1))):
        h, w = int(rng.integers(*size)), int(rng.integers(*size))
        y, x = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
        if rng.random() < 0.5:
            img[y:y + h, x:x + w] = rng.integers(0, 256, 3)
        else:
            img[y:y + h, x:x + w] = rng.integers(0, 256, (h, w, 3))
        mask[y:y + h, x:x + w] = True
    if len(boxes):
        keep = []
        for x1, y1, x2, y2 in boxes.astype(int):
            region = mask[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
            keep.append(region.size == 0 or region.mean() < drop_covered)
        boxes = boxes[np.array(keep, bool)]
    return img, boxes


# --------------------------------------------------------------------------
# Photometric
# --------------------------------------------------------------------------
def photometric(img, rng, p=0.8, brightness=0.3, contrast=0.3, saturation=0.3,
                hue_deg=8, gamma=(0.7, 1.4)):
    """Colour jitter (brightness / contrast / saturation / hue) + gamma.
    Simulates varying sun, time of day and camera white balance."""
    if rng.random() >= p:
        return img
    x = img.astype(np.float32)
    x = x * rng.uniform(1 - brightness, 1 + brightness)
    m = x.mean()
    x = (x - m) * rng.uniform(1 - contrast, 1 + contrast) + m
    x = x.clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue_deg, hue_deg) / 2) % 180  # OpenCV hue is 0..180
    hsv[..., 1] = (hsv[..., 1] * rng.uniform(1 - saturation, 1 + saturation)).clip(0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    x = x ** rng.uniform(*gamma)
    return (x * 255).round().clip(0, 255).astype(np.uint8)


def random_corruption(img, rng, names, p=0.5, max_severity=5.0, max_ops=2):
    """With probability p apply 1..max_ops corruptions drawn from `names`, each at a
    continuous random severity in (0, max_severity]. Restricting `names` gives the
    held-out-corruption experiment (train on some families, test on the others)."""
    if not names or rng.random() >= p:
        return img
    k = int(rng.integers(1, max_ops + 1))
    for name in rng.choice(names, size=min(k, len(names)), replace=False):
        img = C.CORRUPTIONS[name](img, rng.uniform(0.1, max_severity), rng)
    return img


# --------------------------------------------------------------------------
# Config -> callable
# --------------------------------------------------------------------------
def build_augment(cfg):
    """cfg keys (all optional): mosaic (prob), flip (bool), scale_translate (bool),
    cutout (bool), photo (bool), corrupt (list of corruption names), corrupt_p (float)."""
    if not cfg:
        return None

    def augment(img, boxes, rng, dataset):
        if cfg.get("mosaic", 0) and rng.random() < cfg["mosaic"]:
            img, boxes = mosaic(dataset, rng)
        if cfg.get("flip"):
            img, boxes = hflip(img, boxes, rng)
        if cfg.get("scale_translate"):
            img, boxes = scale_translate(img, boxes, rng)
        if cfg.get("cutout"):
            img, boxes = cutout(img, boxes, rng)
        if cfg.get("photo"):
            img = photometric(img, rng)
        if cfg.get("corrupt"):
            img = random_corruption(img, rng, cfg["corrupt"], p=cfg.get("corrupt_p", 0.5))
        return np.ascontiguousarray(img), boxes.astype(np.float32).reshape(-1, 4)

    return augment
