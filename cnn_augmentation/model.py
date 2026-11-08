"""The small CNN detector (architecture identical to simple_cnn_bird_detector.ipynb),
its CenterNet-style loss, and box decoding.

Keeping the layer names identical means checkpoints from the CNN teammate
(e.g. tune-512-clean.pt, {"state": state_dict, ...}) load directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import IMG_SIZE, STRIDE


class Net(nn.Module):
    """4x strided Conv-BN-ReLU encoder, 2x upsampling decoder, 3 heads at stride 4."""

    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hm = nn.Conv2d(32, 1, 1)   # bird centre score (logits)
        self.wh = nn.Conv2d(32, 2, 1)   # log(width / IMG_SIZE), log(height / IMG_SIZE)
        self.off = nn.Conv2d(32, 2, 1)  # sub-cell centre offset

    def init_for_training(self):
        """CenterNet trick: start the heatmap at p ~= 0.1 so focal loss is stable early on."""
        nn.init.constant_(self.hm.bias, -2.19)

    def forward(self, x):
        x = self.dec(self.enc(x))
        return self.hm(x), self.wh(x), self.off(x)


def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def focal_loss(logits, gt):
    """Penalty-reduced pixel-wise focal loss (CenterNet / CornerNet, alpha=2, beta=4)."""
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - gt) ** 4 * neg
    num_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def _gather(feat, ind):
    """feat [B, C, H, W], ind [B, K] -> [B, K, C]."""
    B, C = feat.shape[:2]
    feat = feat.view(B, C, -1).permute(0, 2, 1)
    return feat.gather(1, ind.unsqueeze(2).expand(-1, -1, C))


def reg_l1(pred, target, ind, mask):
    p = _gather(pred, ind)
    m = mask.unsqueeze(2)
    return (torch.abs(p - target) * m).sum() / (m.sum() * p.shape[2]).clamp(min=1)


def detection_loss(outputs, targets, w_wh=1.0, w_off=1.0):
    hm, wh, off = outputs
    l_hm = focal_loss(hm, targets["hm"])
    l_wh = reg_l1(wh, targets["wh"], targets["ind"], targets["mask"])
    l_off = reg_l1(off, targets["off"], targets["ind"], targets["mask"])
    total = l_hm + w_wh * l_wh + w_off * l_off
    return total, {"hm": l_hm.item(), "wh": l_wh.item(), "off": l_off.item()}


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
@torch.no_grad()
def decode(outputs, thr=0.05, topk=200):
    """Heatmap peaks -> boxes. Returns a list (one per image) of
    (boxes[N,4] x1y1x2y2 numpy float32, scores[N] numpy float32).

    Same logic as `predict` in the notebook, but batched and with topk=200
    instead of 100 (images contain up to 130 birds, so 100 caps recall).
    A low threshold is used for mAP so the full precision/recall curve is covered.
    """
    hm, wh, off = outputs
    hm = torch.sigmoid(hm)
    peaks = hm == F.max_pool2d(hm, 3, 1, 1)
    hm = hm * peaks
    B, _, H, W = hm.shape
    k = min(topk, H * W)
    scores, inds = torch.topk(hm.view(B, -1), k)
    ys = torch.div(inds, W, rounding_mode="floor").float()
    xs = (inds % W).float()
    whs = _gather(wh, inds).clamp(-10, 2).exp() * IMG_SIZE   # [B, k, 2]
    offs = _gather(off, inds)                                 # [B, k, 2]
    cx = (xs + offs[..., 0]) * STRIDE
    cy = (ys + offs[..., 1]) * STRIDE
    boxes = torch.stack([cx - whs[..., 0] / 2, cy - whs[..., 1] / 2,
                         cx + whs[..., 0] / 2, cy + whs[..., 1] / 2], dim=-1)
    out = []
    for b in range(B):
        keep = scores[b] > thr
        out.append((boxes[b][keep].cpu().numpy().astype("float32"),
                    scores[b][keep].cpu().numpy().astype("float32")))
    return out
