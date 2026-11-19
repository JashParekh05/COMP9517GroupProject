"""Method 1: four hand-crafted bird detectors (no training), NMS and mAP@0.5.

Every detector takes a grey-scale uint8 image and returns (boxes, scores):
boxes is an (N, 4) array of x1, y1, x2, y2 in pixels, scores is (N,).
"""
import os

import cv2
import numpy as np


def load_split(root, split):
    """Returns a list of (image_path, gt_boxes) for 'train', 'val' or 'test'."""
    items = []
    img_dir = os.path.join(root, "images", split)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"Dataset not found: {img_dir}\nPass the dataset folder with --data \"...\"")
    for name in sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")):
        path = os.path.join(img_dir, name)
        label = os.path.join(root, "labels", split, os.path.splitext(name)[0] + ".txt")
        h, w = cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape
        boxes = []
        if os.path.exists(label):
            for line in open(label):
                parts = line.split()
                if len(parts) != 5:
                    continue
                # YOLO format (class cx cy w h, relative) -> x1, y1, x2, y2 in pixels
                _, cx, cy, bw, bh = map(float, parts)
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
        items.append((path, np.array(boxes, dtype=np.float32).reshape(-1, 4)))
    return items


def _empty():
    return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)


def _scale_space_peaks(R, thresh):
    # 3x3x3 local maxima over (scale, y, x) above the threshold
    dil = np.stack([cv2.dilate(r, np.ones((3, 3), np.uint8)) for r in R])
    dil = np.maximum(dil, np.roll(dil, 1, axis=0))
    dil = np.maximum(dil, np.roll(dil, -1, axis=0))
    peaks = (R == dil) & (R > thresh)
    peaks[0] = peaks[-1] = False  # the first and last scale have only one neighbour
    return np.nonzero(peaks)


def dog_detect(gray, smallest_sigma_blur=2.0, n_scales=12, k=1.4, thresh=0.02,
               polarity="both", box_scale=1.6, refine=False, **_):
    """Difference of Gaussians (SIFT-style scale space)."""
    img = gray.astype(np.float32) / 255.0
    sigmas = [smallest_sigma_blur * k ** i for i in range(n_scales + 1)]
    gauss = [cv2.GaussianBlur(img, (0, 0), s) for s in sigmas]
    dog = np.stack([gauss[i + 1] - gauss[i] for i in range(n_scales)])

    responses = []
    if polarity in ("dark", "both"):    # dark bird on brighter sand/water
        responses.append(dog)
    if polarity in ("bright", "both"):  # bright bird on darker background
        responses.append(-dog)

    boxes, scores = [], []
    for R in responses:
        for s, y, x in zip(*_scale_space_peaks(R, thresh)):
            r = box_scale * sigmas[s]  # a bird extends beyond the blob core
            box = [x - r, y - r, x + r, y + r]
            boxes.append(refine_box(gray, box) if refine else box)
            scores.append(R[s, y, x])
    if not boxes:
        return _empty()
    return np.array(boxes, np.float32), np.array(scores, np.float32)


def doh_detect(gray, smallest_sigma_blur=2.0, n_scales=12, k=1.4, thresh=0.002,
               box_scale=1.6, refine=False, **_):
    """Determinant of the Hessian (SURF-style). Large for blobs, close to zero on edges."""
    img = gray.astype(np.float32) / 255.0
    sigmas = [smallest_sigma_blur * k ** i for i in range(n_scales)]
    R = []
    for s in sigmas:
        g = cv2.GaussianBlur(img, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        R.append((s ** 4) * (dxx * dyy - dxy ** 2))  # sigma^4 makes scales comparable
    R = np.stack(R)

    boxes, scores = [], []
    for s, y, x in zip(*_scale_space_peaks(R, thresh)):
        r = box_scale * sigmas[s]
        box = [x - r, y - r, x + r, y + r]
        boxes.append(refine_box(gray, box) if refine else box)
        scores.append(R[s, y, x])
    if not boxes:
        return _empty()
    return np.array(boxes, np.float32), np.array(scores, np.float32)


def mser_detect(gray, delta=5, min_area=10, max_area=250000, **_):
    """Maximally Stable Extremal Regions. delta = grey levels a region must stay stable over."""
    mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)
    boxes, scores = [], []
    for img in (gray, 255 - gray):  # dark and bright regions
        _, bbs = mser.detectRegions(img)
        for x, y, w, h in bbs:
            boxes.append([x, y, x + w, y + h])
            scores.append(local_contrast(gray, x, y, w, h))
    if not boxes:
        return _empty()
    return np.array(boxes, np.float32), np.array(scores, np.float32)


def threshold_detect(gray, block=51, C=10, open_k=3, min_area=10, max_area=250000, **_):
    """CLAHE + adaptive threshold + opening; connected components become boxes."""
    # CLAHE: contrast equalisation on an 8x8 grid of tiles, clipped so noise is not amplified
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    boxes, scores = [], []
    # pixels darker than their neighbourhood mean by C, then pixels brighter by C
    for mode, c in ((cv2.THRESH_BINARY_INV, C), (cv2.THRESH_BINARY, -C)):
        bw = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, block, c)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        _, _, stats, _ = cv2.connectedComponentsWithStats(bw)
        for x, y, w, h, area in stats[1:]:
            if min_area <= area <= max_area:
                boxes.append([x, y, x + w, y + h])
                scores.append(local_contrast(gray, x, y, w, h))
    if not boxes:
        return _empty()
    return np.array(boxes, np.float32), np.array(scores, np.float32)


def local_contrast(gray, x, y, w, h, pad=None):
    """Score for MSER/threshold boxes: |mean inside - mean of a surrounding ring| / 255."""
    pad = pad or max(w, h)
    H, W = gray.shape
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    outer = gray[y0:y1, x0:x1].astype(np.float32)
    inner = gray[y:y + h, x:x + w].astype(np.float32)
    ring_n = outer.size - inner.size
    if ring_n <= 0 or inner.size == 0:
        return 0.0
    return abs(inner.mean() - (outer.sum() - inner.sum()) / ring_n) / 255.0


def refine_box(gray, box, pad=1.5):
    """Replace a square blob box by the Otsu region around its centre (fits the bird's shape)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = (x2 - x1) / 2 * pad
    H, W = gray.shape
    X1, Y1 = int(max(cx - r, 0)), int(max(cy - r, 0))
    X2, Y2 = int(min(cx + r, W)), int(min(cy + r, H))
    patch = gray[Y1:Y2, X1:X2]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return box
    _, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    px = min(max(int(cx) - X1, 0), patch.shape[1] - 1)
    py = min(max(int(cy) - Y1, 0), patch.shape[0] - 1)
    if bw[py, px] == 0:  # make the bird the white part
        bw = 255 - bw
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    x, y, w, h, area = stats[labels[py, px]]  # region containing the centre
    # keep the square if the region is tiny or fills the whole window (= background)
    if area < 4 or (w >= patch.shape[1] - 1 and h >= patch.shape[0] - 1):
        return box
    return [X1 + x, Y1 + y, X1 + x + w, Y1 + y + h]


DETECTORS = {
    "dog": dog_detect,
    "doh": doh_detect,
    "mser": mser_detect,
    "threshold": threshold_detect,
}

# settings searched on clean validation images (39 in total)
GRIDS = {
    "dog": [dict(smallest_sigma_blur=s, box_scale=b, refine=rf)
            for s in [2.0, 3.0] for b in [2.0, 2.5, 3.0] for rf in [False, True]],
    "doh": [dict(smallest_sigma_blur=s, box_scale=b, refine=rf)
            for s in [2.0, 3.0] for b in [2.0, 2.5, 3.0] for rf in [False, True]],
    "mser": [dict(delta=d, min_area=a) for d in [3, 5, 8] for a in [5, 15]],
    "threshold": [dict(block=bl, C=c) for bl in [31, 51, 101] for c in [5, 10, 20]],
}


def nms(boxes, scores, iou_thr=0.3):
    """Greedy non-maximum suppression (removes duplicate boxes)."""
    if len(boxes) == 0:
        return boxes, scores
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), 0.0, iou_thr)
    keep = np.array(keep).flatten() if len(keep) else np.array([], int)
    return boxes[keep], scores[keep]


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(all_preds, all_gts, iou_thr=0.5):
    """mAP@0.5 (one class, so mAP = AP) with VOC all-point interpolation.

    all_preds: list of (boxes, scores) per image; all_gts: list of gt boxes per image.
    Predictions of all images are ranked together by score; each one matches the
    unmatched ground-truth box it overlaps most if IoU >= iou_thr.
    """
    n_gt = sum(len(g) for g in all_gts)
    if n_gt == 0:
        return 0.0
    records = [(s, i, b) for i, (pb, ps) in enumerate(all_preds) for b, s in zip(pb, ps)]
    records.sort(key=lambda r: -r[0])
    matched = [np.zeros(len(g), bool) for g in all_gts]
    tp = np.zeros(len(records))
    for n, (_, i, b) in enumerate(records):
        g = all_gts[i]
        if len(g) == 0:
            continue
        ious = iou_matrix(b[None], g)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[i][j]:
            tp[n] = 1
            matched[i][j] = True
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
