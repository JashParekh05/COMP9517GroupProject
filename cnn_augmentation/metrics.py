"""Detection metrics. Pure numpy, framework-independent (shareable with the non-ML methods).

AP definition is identical to `average_precision` on the `nonlearning` branch
(PASCAL VOC all-point interpolation, greedy matching in descending score order,
IoU >= 0.5), so numbers are directly comparable across the group's methods.
With a single class ("bird"), mAP@0.5 == AP@0.5.
"""
import numpy as np


def iou_matrix(a, b):
    """IoU between every box in a [N,4] and b [M,4] (x1y1x2y2) -> [N, M]."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(preds, gts, iou_thr=0.5):
    """Greedy VOC matching over the whole dataset.

    preds: list of (boxes[N,4], scores[N]) per image; gts: list of boxes[M,4].
    Returns (scores sorted desc, tp flags, per-image matched-GT masks, n_gt).
    """
    recs = [(s, i, b) for i, (pb, ps) in enumerate(preds) for b, s in zip(pb, ps)]
    recs.sort(key=lambda r: -r[0])
    matched = [np.zeros(len(g), bool) for g in gts]
    tp = np.zeros(len(recs))
    for k, (_, i, b) in enumerate(recs):
        g = gts[i]
        if len(g) == 0:
            continue
        ious = iou_matrix(b[None], g)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[i][j]:
            tp[k] = 1
            matched[i][j] = True
    scores = np.array([r[0] for r in recs])
    return scores, tp, matched, sum(len(g) for g in gts)


def average_precision(preds, gts, iou_thr=0.5):
    """AP@iou_thr with VOC all-point interpolation."""
    _, tp, _, n_gt = match(preds, gts, iou_thr)
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
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_summary(preds, gts, iou_thr=0.5, score_thr=0.3,
                      size_bins=(0, 8, 16, 32, 1e9)):
    """Extra numbers for failure analysis.

    - ap: AP@iou_thr
    - precision / recall / f1 at a fixed operating point (score >= score_thr)
    - recall_by_size: recall at that operating point, split by GT size
      (sqrt(area) in pixels), to see e.g. whether blur kills the tiny birds first
    - count_mae: mean absolute error of the per-image bird count
    """
    ap = average_precision(preds, gts, iou_thr)
    filt = [(b[s >= score_thr], s[s >= score_thr]) for b, s in preds]
    _, tp, matched, n_gt = match(filt, gts, iou_thr)
    n_pred = len(tp)
    n_tp = int(tp.sum())
    prec = n_tp / max(n_pred, 1)
    rec = n_tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    sizes = np.concatenate([np.sqrt(np.clip((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]), 0, None))
                            for g in gts if len(g)] or [np.zeros(0)])
    hit = np.concatenate([m for m, g in zip(matched, gts) if len(g)] or [np.zeros(0, bool)])
    by_size = {}
    for lo, hi in zip(size_bins[:-1], size_bins[1:]):
        sel = (sizes >= lo) & (sizes < hi)
        label = f"{int(lo)}-{int(hi)}px" if hi < 1e8 else f">={int(lo)}px"
        by_size[label] = {"n": int(sel.sum()),
                          "recall": float(hit[sel].mean()) if sel.any() else None}

    count_err = [abs(len(b) - len(g)) for (b, _), g in zip(filt, gts)]
    return {"ap": ap, "precision": prec, "recall": rec, "f1": f1,
            "score_thr": score_thr, "recall_by_size": by_size,
            "count_mae": float(np.mean(count_err)) if count_err else 0.0}
