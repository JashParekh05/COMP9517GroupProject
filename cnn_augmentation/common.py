"""Shared code: loading the dataset, the CNN, the loss, turning outputs into boxes, and mAP.

Boxes are always numpy arrays of shape [N, 4] as (x1, y1, x2, y2) in pixels.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 512          # all images are 512x512
STRIDE = 4         # the network output is 4x smaller than the image
OUT = IMG // STRIDE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if there is one (e.g. Colab)

# Where the dataset is if you don't pass --data: Group Project/dataset/..., next to the repo folder.
# Can also be set with the BIRD_DATA environment variable.
DEFAULT_DATA = os.environ.get("BIRD_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "Beach Bird Detection Dataset v1.0")


# ---------------------------------------------------------------- data
def load_split(root, split):
    """Returns a list of (image_path, boxes) for 'train', 'val' or 'test'."""
    items = []
    img_dir = os.path.join(root, "images", split)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"Dataset not found: {img_dir}\nPass the dataset folder with --data \"...\"")
    for name in sorted(os.listdir(img_dir)):
        label_file = os.path.join(root, "labels", split, os.path.splitext(name)[0] + ".txt")
        boxes = []
        if os.path.exists(label_file):
            for line in open(label_file):
                if len(line.split()) == 5:
                    # YOLO format: class cx cy w h, all relative to the image size
                    _, cx, cy, w, h = map(float, line.split())
                    boxes.append([(cx - w / 2) * IMG, (cy - h / 2) * IMG,
                                  (cx + w / 2) * IMG, (cy + h / 2) * IMG])
        items.append((os.path.join(img_dir, name), np.array(boxes, np.float32).reshape(-1, 4)))
    return items


def read_image(path):
    """RGB uint8 image of size 512x512."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG, IMG))


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255


def make_targets(boxes):
    """What the network should output for these boxes (CenterNet-style):
    - heat:   a Gaussian blob at every bird centre (peak value 1)
    - size:   log(width/512), log(height/512) at the centre cell
    - offset: where exactly inside the centre cell the centre is
    - mask:   1 at the centre cells (only those cells get a size/offset loss)
    """
    heat = np.zeros((OUT, OUT), np.float32)
    size = np.zeros((2, OUT, OUT), np.float32)
    offset = np.zeros((2, OUT, OUT), np.float32)
    mask = np.zeros((OUT, OUT), np.float32)
    ys, xs = np.mgrid[0:OUT, 0:OUT]
    for x1, y1, x2, y2 in boxes:
        w, h = max(x2 - x1, 1), max(y2 - y1, 1)
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
        ix, iy = min(int(cx), OUT - 1), min(int(cy), OUT - 1)
        sigma = max(min(w, h) / STRIDE / 6, 0.5)          # bigger bird -> wider blob
        blob = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2))
        heat = np.maximum(heat, blob)
        size[:, iy, ix] = np.log([w / IMG, h / IMG])
        offset[:, iy, ix] = [cx - ix, cy - iy]
        mask[iy, ix] = 1
    return heat[None], size, offset, mask


class BirdData(torch.utils.data.Dataset):
    def __init__(self, items, augment=None):
        self.items = items
        self.augment = augment      # function (img, boxes) -> (img, boxes), or None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        img = read_image(path)
        if self.augment:
            img, boxes = self.augment(img, boxes.copy())
        targets = [torch.from_numpy(t) for t in make_targets(boxes)]
        return to_tensor(img), *targets


# ---------------------------------------------------------------- model
class Net(nn.Module):
    """Same network as simple_cnn_bird_detector.ipynb (so its weights load here too)."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.enc = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.hm = nn.Conv2d(32, 1, 1)   # bird score
        self.wh = nn.Conv2d(32, 2, 1)   # width, height
        self.off = nn.Conv2d(32, 2, 1)  # centre offset

    def forward(self, x):
        x = self.dec(self.enc(x))
        return self.hm(x), self.wh(x), self.off(x)


def load_model(path):
    model = Net()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"])
    return model.to(DEVICE).eval()


def loss_fn(outputs, heat, size, offset, mask):
    """Focal loss on the heatmap (from CenterNet) + L1 loss on size and offset."""
    heat_pred, size_pred, off_pred = outputs
    p = torch.sigmoid(heat_pred).clamp(1e-4, 1 - 1e-4)
    is_centre = heat.eq(1).float()
    pos = -(torch.log(p) * (1 - p) ** 2 * is_centre).sum()
    neg = -(torch.log(1 - p) * p ** 2 * (1 - heat) ** 4 * (1 - is_centre)).sum()
    heat_loss = (pos + neg) / is_centre.sum().clamp(min=1)

    m = mask.unsqueeze(1)
    n = mask.sum().clamp(min=1)
    size_loss = (torch.abs(size_pred - size) * m).sum() / n
    off_loss = (torch.abs(off_pred - offset) * m).sum() / n
    return heat_loss + size_loss + off_loss


@torch.no_grad()
def predict(model, images, thr=0.05, max_birds=200, batch_size=16):
    """Peaks of the heatmap -> boxes (same idea as `predict` in the notebook).
    Takes a list of images and runs them through the network in batches (faster).
    Returns a list with one (boxes [N,4], scores [N]) per image."""
    results = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([to_tensor(img) for img in images[start:start + batch_size]])
        device = next(model.parameters()).device
        heat, size, off = [t.cpu() for t in model(batch.to(device))]       # run on GPU if there is one
        heat = torch.sigmoid(heat)[:, 0]                                    # B x 128 x 128
        is_peak = (heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]) & (heat > thr)

        # the number of birds differs per image, so the boxes are read out image by image
        for b in range(len(batch)):
            ys, xs = torch.where(is_peak[b])
            scores = heat[b, ys, xs]
            if len(scores) > max_birds:    # notebook used 100, but some images have 130 birds
                scores, keep = scores.topk(max_birds)
                ys, xs = ys[keep], xs[keep]
            w = size[b, 0, ys, xs].clamp(max=2).exp() * IMG
            h = size[b, 1, ys, xs].clamp(max=2).exp() * IMG
            cx = (xs + off[b, 0, ys, xs]) * STRIDE
            cy = (ys + off[b, 1, ys, xs]) * STRIDE
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            results.append((boxes.numpy(), scores.numpy()))
    return results


# ---------------------------------------------------------------- metric
def iou(box, boxes):
    """IoU of one box with each of `boxes`."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def mean_average_precision(preds, gts, iou_thr=0.5):
    """mAP@0.5 (one class, so mAP = AP). Same definition as the non-learning branch:
    go through all predictions from most to least confident, a prediction is correct if it
    overlaps a not-yet-matched bird with IoU >= 0.5, then take the area under the
    precision-recall curve (VOC all-point interpolation).
    preds: list of (boxes, scores) per image, gts: list of boxes per image."""
    all_preds = [(s, i, b) for i, (boxes, scores) in enumerate(preds) for b, s in zip(boxes, scores)]
    all_preds.sort(key=lambda p: -p[0])
    used = [np.zeros(len(g), bool) for g in gts]
    correct = np.zeros(len(all_preds))
    for k, (_, i, box) in enumerate(all_preds):
        if len(gts[i]) == 0:
            continue
        overlaps = iou(box, gts[i])
        j = overlaps.argmax()
        if overlaps[j] >= iou_thr and not used[i][j]:
            correct[k] = 1
            used[i][j] = True

    n_birds = sum(len(g) for g in gts)
    tp = np.cumsum(correct)
    recall = tp / max(n_birds, 1)
    precision = tp / np.arange(1, len(correct) + 1)
    # area under the precision-recall curve, with precision made monotonically decreasing
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
