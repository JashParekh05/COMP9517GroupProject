"""Image corruptions with 5 levels each (idea from ImageNet-C, Hendrycks & Dietterich 2019).

Only uses numpy + OpenCV, so the non-learning method can use exactly the same images:
    from corruptions import corrupt, CORRUPTIONS
    bad_img = corrupt(img, "blur", level=3, seed=image_index)

Run `python corruptions.py some_image.jpg` to save a picture of all corruptions.
"""
import sys

import cv2
import numpy as np

# value used at level 1, 2, 3, 4, 5
CORRUPTIONS = {
    "noise":     [0.02, 0.04, 0.06, 0.09, 0.12],   # std of Gaussian noise (image in 0..1)
    "blur":      [0.5, 1.0, 1.5, 2.0, 3.0],        # Gaussian blur sigma in pixels
    "jpeg":      [60, 40, 25, 15, 8],              # JPEG quality
    "dark":      [0.7, 0.5, 0.35, 0.25, 0.15],     # brightness multiplier
    "contrast":  [0.7, 0.5, 0.35, 0.25, 0.15],     # contrast multiplier
    "fog":       [0.15, 0.3, 0.4, 0.5, 0.6],       # how much white haze is blended in
    "occlusion": [0.05, 0.1, 0.2, 0.3, 0.4],       # fraction of the image covered by blocks
}


def corrupt(img, name, level, seed=0):
    """img: uint8 HxWx3. level: 1..5. Same seed -> same result (for fair comparisons)."""
    rng = np.random.default_rng(seed)
    v = CORRUPTIONS[name][level - 1]
    x = img.astype(np.float32)

    if name == "noise":
        x = x + rng.normal(0, v * 255, x.shape)
    elif name == "blur":
        x = cv2.GaussianBlur(x, (0, 0), v)
    elif name == "jpeg":
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, v])
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    elif name == "dark":
        x = x * v
    elif name == "contrast":
        x = (x - x.mean()) * v + x.mean()
    elif name == "fog":
        x = x * (1 - v) + 220 * v
    elif name == "occlusion":
        # random coloured rectangles until the target area is covered
        h, w = img.shape[:2]
        covered = np.zeros((h, w), bool)
        while covered.mean() < v:
            bh, bw = rng.integers(30, 100, 2)
            y, x0 = rng.integers(0, h - bh), rng.integers(0, w - bw)
            x[y:y + bh, x0:x0 + bw] = rng.integers(0, 256, 3)
            covered[y:y + bh, x0:x0 + bw] = True
    return np.clip(x, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(3, len(CORRUPTIONS), figsize=(2.2 * len(CORRUPTIONS), 7))
    for col, name in enumerate(CORRUPTIONS):
        for row, level in enumerate([1, 3, 5]):
            axes[row, col].imshow(corrupt(img, name, level))
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{name} {level}", fontsize=9)
    plt.tight_layout()
    plt.savefig("corruptions.png", dpi=120)
    print("saved corruptions.png")
