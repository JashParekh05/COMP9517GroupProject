"""Realistic image corruptions for robustness evaluation (and corruption-based augmentation).

Pure numpy + OpenCV, no torch. The non-ML robustness experiments can import this file
directly so that both methods are evaluated on *identical* corrupted images:

    from corruptions import CORRUPTIONS, corrupt_deterministic
    img_c = corrupt_deterministic(img_rgb_uint8, "gaussian_blur", severity=3, image_index=i)

Design follows ImageNet-C (Hendrycks & Dietterich, ICLR 2019): each corruption has
5 severity levels. Every function accepts a *continuous* severity in [0, 5]
(0 = clean) by interpolating its parameter table, which is what training-time
corruption augmentation uses; evaluation uses the integer levels 1..5.

All corruptions are photometric or occluding only -- box annotations stay unchanged.
Images are uint8 HxWx3 (RGB or BGR; channel order does not matter for any of these).
"""
import zlib

import cv2
import numpy as np

# parameter at severity 0, 1, 2, 3, 4, 5
PARAMS = {
    "gaussian_noise": [0.0, 0.02, 0.04, 0.06, 0.09, 0.12],   # std, image in [0,1]
    "gaussian_blur":  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],         # sigma in px
    "motion_blur":    [1, 3, 5, 7, 9, 13],                    # kernel length in px
    "jpeg":           [100, 60, 40, 25, 15, 8],               # JPEG quality
    "low_brightness": [1.0, 0.7, 0.5, 0.35, 0.25, 0.15],      # intensity scale
    "low_contrast":   [1.0, 0.7, 0.5, 0.35, 0.25, 0.15],      # contrast factor
    "fog":            [0.0, 0.15, 0.3, 0.4, 0.5, 0.6],        # max haze opacity
    "occlusion":      [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],        # fraction of image covered
}


def _param(name, severity):
    return float(np.interp(severity, np.arange(6), PARAMS[name]))


def gaussian_noise(img, severity, rng):
    std = _param("gaussian_noise", severity)
    x = img.astype(np.float32) / 255.0 + rng.normal(0, std, img.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def gaussian_blur(img, severity, rng):
    sigma = _param("gaussian_blur", severity)
    if sigma < 0.1:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigma)


def motion_blur(img, severity, rng):
    """Linear motion blur with a random direction (camera shake / flying birds)."""
    length = int(round(_param("motion_blur", severity)))
    if length <= 1:
        return img.copy()
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    angle = rng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def jpeg(img, severity, rng):
    q = int(round(_param("jpeg", severity)))
    if q >= 100:
        return img.copy()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def low_brightness(img, severity, rng):
    """Underexposure / dusk: scale intensities down."""
    f = _param("low_brightness", severity)
    return (img.astype(np.float32) * f).round().clip(0, 255).astype(np.uint8)


def low_contrast(img, severity, rng):
    """Flat lighting / overcast: blend towards the mean grey value."""
    c = _param("low_contrast", severity)
    x = img.astype(np.float32)
    m = x.mean()
    return ((x - m) * c + m).round().clip(0, 255).astype(np.uint8)


def fog(img, severity, rng):
    """Sea mist / haze: blend with a bright airlight using a smooth random opacity map."""
    a_max = _param("fog", severity)
    if a_max <= 0:
        return img.copy()
    H, W = img.shape[:2]
    coarse = rng.uniform(0.5, 1.0, (4, 4)).astype(np.float32)
    alpha = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1) * a_max
    airlight = 0.85 * 255
    x = img.astype(np.float32) * (1 - alpha[..., None]) + airlight * alpha[..., None]
    return x.round().clip(0, 255).astype(np.uint8)


def occlusion(img, severity, rng):
    """Partial occlusion: random grey/noise rectangles until a target area fraction is covered.

    Note: GT boxes are *not* removed, so fully occluded birds count as misses.
    This measures how much detection degrades when parts of the scene are hidden.
    """
    frac = _param("occlusion", severity)
    out = img.copy()
    H, W = img.shape[:2]
    covered = np.zeros((H, W), bool)
    target = frac * H * W
    tries = 0
    while covered.sum() < target and tries < 200:
        tries += 1
        h = int(rng.uniform(0.06, 0.2) * H)
        w = int(rng.uniform(0.06, 0.2) * W)
        y, x = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
        if rng.random() < 0.5:
            out[y:y + h, x:x + w] = rng.integers(0, 256, 3)
        else:
            out[y:y + h, x:x + w] = rng.integers(0, 256, (h, w, 3))
        covered[y:y + h, x:x + w] = True
    return out


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg": jpeg,
    "low_brightness": low_brightness,
    "low_contrast": low_contrast,
    "fog": fog,
    "occlusion": occlusion,
}
SEVERITIES = (1, 2, 3, 4, 5)


def corrupt(img, name, severity, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    return CORRUPTIONS[name](img, severity, rng)


def corrupt_deterministic(img, name, severity, image_index):
    """Same (image, corruption, severity) -> always the same output, for any method.

    The seed depends only on the corruption name, severity and image index,
    so every model/method is evaluated on exactly the same corrupted pixels.
    """
    seed = [zlib.crc32(name.encode()), int(round(severity * 10)), int(image_index)]
    return CORRUPTIONS[name](img, severity, np.random.default_rng(seed))
