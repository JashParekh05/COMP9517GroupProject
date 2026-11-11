import os
import time
 
import cv2
import numpy as np

# load the data

def load_split(root, split):
    # returns image_path, gt_boxes[N,4] in x1,y1,x2,y2 pixels).
    items = []
    img_dir = os.path.join(root, "images", split)
    img_exts = (".jpg")
    img_paths = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                       if f.lower().endswith(img_exts))
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        label = os.path.join(root, "labels", split, stem + ".txt")
        h, w = cv2.imread(p, cv2.IMREAD_GRAYSCALE).shape
        boxes = []
        if os.path.exists(label):
            for line in open(label):
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, center_x, center_y, box_width, box_height = map(float, parts)
                #convert from YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)
                boxes.append([(center_x - box_width / 2) * w, (center_y - box_height / 2) * h,
                              (center_x + box_width / 2) * w, (center_y + box_height / 2) * h])
        items.append((p, np.array(boxes, dtype=np.float32).reshape(-1, 4)))
    return items

#could delete this. can be used for blur range. will help for dataset description.
# tells us how big the bird boxes are.

def box_size_stats(items):
    sizes = np.concatenate([np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
                            for _, b in items if len(b)])
    print(f"bird box size (sqrt area, px): min {sizes.min():.1f}  "
          f"p10 {np.percentile(sizes, 10):.1f}  median {np.median(sizes):.1f}  "
          f"p90 {np.percentile(sizes, 90):.1f}  max {sizes.max():.1f}")
    return sizes


# DoG scale-space blob detection

def dog_detect(gray, smallest_sigma_blur = 2.0, n_scales=12, k = 1.4, thresh=0.02,
               polarity="both", box_scale=1.6, refine=False, **_):
    img = gray.astype(np.float32)/255.0
    sigmas = [smallest_sigma_blur * k ** i for i in range(n_scales + 1)]
    gauss = [cv2.GaussianBlur(img, (0, 0), s) for s in sigmas]
    dog = np.stack([(gauss[i + 1] - gauss[i]) for i in range(n_scales)])
 
    responses = []
    if polarity in ("dark", "both"):   # dark bird on bright sand/water
        responses.append(dog)
    if polarity in ("bright", "both"):  # bright on darker background
        responses.append(-dog) #fliped sign so that the bright blobs are positive responses
 
    boxes, scores = [], []
    for R in responses:
        # local maxima across space and scale
        dil = np.stack([cv2.dilate(r, np.ones((3, 3), np.uint8)) for r in R])
        dil = np.maximum(dil, np.roll(dil, 1, axis=0))
        dil = np.maximum(dil, np.roll(dil, -1, axis=0))
        peaks = (R == dil) & (R > thresh)
        peaks[0] = peaks[-1] = False  #ignores boundary scales
        s_idx, ys, xs = np.nonzero(peaks)
        for s, y, x in zip(s_idx, ys, xs):
            r = box_scale * sigmas[s]  # blob radius is equal to sqrt(2)*sigma, so box_scale ~1.4-2
            box = [x - r, y - r, x + r, y + r]
            if refine:
                box = refine_box(gray, box)
            boxes.append(box)
            scores.append(R[s, y, x])
 
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.array(boxes, np.float32), np.array(scores, np.float32)

#determinant of hessian
def doh_detect(gray, smallest_sigma_blur = 2.0, n_scales=12, k = 1.4, thresh=0.002,
               box_scale=1.6, refine=False, **_):
    img = gray.astype(np.float32) / 255.0
    sigmas = [smallest_sigma_blur * k ** i for i in range(n_scales)]
    R = []
    for s in sigmas:
        g = cv2.GaussianBlur(img, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        R.append((s ** 4) * (dxx * dyy - dxy ** 2))  # normalize
    R = np.stack(R)
    dil = np.stack([cv2.dilate(r, np.ones((3, 3), np.uint8)) for r in R])
    dil = np.maximum(dil, np.roll(dil, 1, axis=0))
    dil = np.maximum(dil, np.roll(dil, -1, axis=0))
    peaks = (R == dil) & (R > thresh)
    peaks[0] = peaks[-1] = False
    s_idx, ys, xs = np.nonzero(peaks)
    boxes, scores = [], []
    for s, y, x in zip(s_idx, ys, xs):
        r = box_scale * sigmas[s]
        box = [x - r, y - r, x + r, y + r]
        if refine:
            box = refine_box(gray, box)
        boxes.append(box)
        scores.append(R[s, y, x])
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.array(boxes, np.float32), np.array(scores, np.float32)

# Maximally Stable Extremal Regions
# delta --> how many grey levels a region should stay stable over
#min and max area are rejected regions that are too small or too large
def mser_detect(gray, delta=5, min_area=10, max_area=250000, **_):
    mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)
    boxes, scores = [], []
    for inv in (False, True):
        img = 255 - gray if inv else gray
        regions, bbs = mser.detectRegions(img)
        for (x, y, w, h) in bbs:
            boxes.append([x, y, x + w, y + h])
            scores.append(local_contrast(gray, x, y, w, h))
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.array(boxes, np.float32), np.array(scores, np.float32)
 
# segmentation 
def threshold_detect(gray, block=51, C=10, open_k=3, min_area=10,
                     max_area=250000, **_):
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) #CLAHE splits the image into a 8x8 grid and stretches the contrast. clipLimit stops from overamplifying noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    boxes, scores = [], []
    for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        bw = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   mode, block, C)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw)
        for x, y, w, h, area in stats[1:]:
            if min_area <= area <= max_area:
                boxes.append([x, y, x + w, y + h])
                scores.append(local_contrast(gray, x, y, w, h))
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.array(boxes, np.float32), np.array(scores, np.float32)
 
 
def local_contrast(gray, x, y, w, h, pad=None):
    pad = pad or max(w, h)
    H, W = gray.shape
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    outer = gray[y0:y1, x0:x1].astype(np.float32)
    inner = gray[y:y + h, x:x + w].astype(np.float32)
    ring_sum = outer.sum() - inner.sum()
    ring_n = outer.size - inner.size
    if ring_n <= 0 or inner.size == 0:
        return 0.0
    return abs(inner.mean() - ring_sum / ring_n) / 255.0
 
def refine_box(gray, box, pad=1.5):
    #Fit a box to the bird's real shape around a blob detection
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = (x2 - x1) / 2 * pad
    H, W = gray.shape
    X1, Y1 = int(max(cx - r, 0)), int(max(cy - r, 0))
    X2, Y2 = int(min(cx + r, W)), int(min(cy + r, H))
    patch = gray[Y1:Y2, X1:X2]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return box
    # Otsu picks the threshold that best splits bird vs background
    _, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    px = min(max(int(cx) - X1, 0), patch.shape[1] - 1)
    py = min(max(int(cy) - Y1, 0), patch.shape[0] - 1)
    if bw[py, px] == 0: # make the bird the white part
        bw = 255 - bw
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    lab = labels[py, px] # blob containing the centre
    x, y, w, h, area = stats[lab]
    # reject failures: tiny blob, or blob filling the whole window (= background)
    if area < 4 or (w >= patch.shape[1] - 1 and h >= patch.shape[0] - 1):
        return box
    return [X1 + x, Y1 + y, X1 + x + w, Y1 + y + h]

DETECTORS = {
    "dog": dog_detect, # SIFT scale-space detector
    "doh": doh_detect, # SURF-style Hessian detector
    "mser": mser_detect, # stable intensity regions
    "threshold": threshold_detect, # segmentation/morphology
}
 
# removes duplicates
def nms(boxes, scores, iou_thr=0.3):
    if len(boxes) == 0:
        return boxes, scores
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), 0.0, iou_thr)
    keep = np.array(keep).flatten() if len(keep) else np.array([], int)
    return boxes[keep], scores[keep]
 
 
#############
# Evaluation
#############
def iou_matrix(a, b): #looks at overlap between all pairs of boxes
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter/(area_a[:, None] + area_b[None, :] - inter + 1e-9)
 
# this computes the score
def average_precision(all_preds, all_gts, iou_thr=0.5):
    records, n_gt = [], sum(len(g) for g in all_gts)
    for img_id, ((pb, ps), gb) in enumerate(zip(all_preds, all_gts)):
        for b, s in zip(pb, ps):
            records.append((s, img_id, b))
    records.sort(key=lambda r: -r[0])
    matched = [np.zeros(len(g), bool) for g in all_gts]
    tp = np.zeros(len(records))
    for i, (_, img_id, b) in enumerate(records):
        g = all_gts[img_id]
        if len(g) == 0:
            continue
        ious = iou_matrix(b[None], g)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[img_id][j]:
            tp[i] = 1
            matched[img_id][j] = True
    if n_gt == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    n_pred = len(tp)
    n_tp = int(ctp[-1]) if n_pred else 0
    print(f"    {n_gt} real birds, {n_pred} predictions, {n_tp} correct | "
          f"max recall {n_tp / n_gt:.2f}, "
          f"precision at that point {n_tp / max(n_pred, 1):.3f}")
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
 
 

def run(items, detector="dog", **params):
    detect = DETECTORS[detector]
    preds, gts, t0 = [], [], time.time()
    for p, g in items:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        b, s = detect(gray, **{k: v for k, v in params.items() if k != "iou_thr"})
        b, s = nms(b, s, params.get("iou_thr", 0.3))
        preds.append((b, s))
        gts.append(g)
    dt = (time.time() - t0) / max(len(items), 1)
    return average_precision(preds, gts), dt, preds

def debug_one(item, detector="dog", out="debug.jpg", **params):
    p, g = item
    gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    b, s = DETECTORS[detector](gray, **params)
    b, s = nms(b, s)
    print(f"{os.path.basename(p)}: {len(g)} real birds, {len(b)} predictions")
    if len(b) and len(g):
        best = iou_matrix(g, b).max(axis=1)
        print("best IoU for each real bird:", np.round(best, 2))
    img = cv2.imread(p)
    for x1, y1, x2, y2 in g.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x1, y1, x2, y2 in b[np.argsort(-s)[:50]].astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(out, img)

# parameter grids searched on the validation set (module level so robustness.py can reuse them)
GRIDS = {
    "dog": [dict(smallest_sigma_blur=s, box_scale=b, refine=rf)
            for s in [2.0, 3.0] for b in [2.0, 2.5, 3.0] for rf in [False, True]],
    "doh": [dict(smallest_sigma_blur=s, box_scale=b, refine=rf)
            for s in [2.0, 3.0] for b in [2.0, 2.5, 3.0] for rf in [False, True]],
    "mser": [dict(delta=d, min_area=a) for d in [3, 5, 8] for a in [5, 15]],
    "threshold": [dict(block=bl, C=c) for bl in [31, 51, 101]
                  for c in [5, 10, 20]],
}

if __name__ == "__main__":
    ROOT = "dataset"
 
    val = load_split(ROOT, "val")
    val = val[:20] #testing
    box_size_stats(val)
    """debug_one(next(it for it in val if len(it[1])), detector="doh",
              smallest_sigma_blur=2.0, box_scale=2.5)
    raise SystemExit"""

    best_per_method = {}
    for name, grid in GRIDS.items():
        best = None
        for params in grid:
            ap, dt, _ = run(val, detector=name, **params)
            print(f"[{name}] {params} -> val mAP@0.5={ap:.3f} ({dt:.2f}s/img)")
            if best is None or ap > best[0]:
                best = (ap, params, dt)
        best_per_method[name] = best
        print(f"[{name}] BEST on val: {best}\n")
 
    