# Method 2: CenterNet-style CNN + data augmentation

Small from-scratch CNN (same architecture for every run). We ask:

1. How badly does it break on noisy / blurry / dark / low-contrast / foggy / JPEG / occluded images?
2. Does data augmentation fix that? If we only train on some damage types, does it help on unseen ones?

## Files

| File | Role |
|------|------|
| `common.py` | dataset loading, `Net`, loss, decode boxes from heatmap, mAP@0.5 |
| `corruptions.py` | 7 corruptions × 5 levels (NumPy/OpenCV only; Method 1 reuses this) |
| `train.py` | train with one `--aug` setting |
| `evaluate.py` | clean + corrupted eval table / plot / example image |
| `colab.ipynb` | optional Colab training notebook |

## Augmentation settings

| `--aug` | Training images get… |
|---------|----------------------|
| `none` | nothing (clean baseline) |
| `basic` | horizontal flip + brightness/contrast/saturation |
| `robust` | basic + 50% chance of a random corruption (any of the 7) |
| `robust_seen` | basic + 50% chance of noise / dark / JPEG only |

## How to run

Dataset is **not** in the ZIP. Pass `--data` to the Beach Bird Detection Dataset v1.0 folder
(must contain `images/` and `labels/`), or set `BIRD_DATA`.

```bash
pip install -r ../requirements.txt
cd cnn

python train.py --aug none --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug basic --data "PATH/TO/..."
python train.py --aug robust --epochs 80 --data "PATH/TO/..."
python train.py --aug robust_seen --epochs 80 --data "PATH/TO/..."

# do not include runs/*.pt in the course ZIP
python evaluate.py runs/none.pt runs/basic.pt runs/robust.pt runs/robust_seen.pt --data "PATH/TO/..."
python evaluate.py runs/none.pt runs/robust.pt --example 5 --corruption blur --level 4 --data "PATH/TO/..."
```

Useful flags:

- `--frac 0.5` — train on half the images if time is short
- `--epochs` — default 30; use 80 for robust settings in the report
- `--split test` on `evaluate.py` — only once, for final numbers

## Notes

- mAP@0.5 matches Method 1 (VOC all-point), so scores are comparable.
- Up to 200 boxes per image (some scenes have up to 130 birds).
- Corruption seed = image index (same pixels for every model and for Method 1).
- Do **not** ship trained weights or result images in the submission ZIP.

## References

- Zhou et al., *Objects as Points* (CenterNet), 2019
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness…*, 2019
- Michaelis et al., *Benchmarking Robustness in Object Detection*, 2019
