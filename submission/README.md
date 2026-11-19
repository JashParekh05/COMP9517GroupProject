# Beach Bird Detection — COMP9517 code

Ready-to-run code for the group project (Method 1 + Method 2).

**Do not put the dataset, trained `.pt` weights, or result images in the submission ZIP** (spec ≤ 25 MB).

## Folder layout

```
submission/
  requirements.txt
  README.md                 ← you are here
  nonlearning/              ← Method 1 (classical detectors, no training)
  cnn/                      ← Method 2 (CenterNet-style CNN + augmentation)
```

## Setup

1. Install Python 3.10+ and dependencies:

```bash
pip install -r requirements.txt
```

2. Download the **Beach Bird Detection Dataset v1.0** separately (not in this ZIP).
   The folder must contain `images/{train,val,test}` and `labels/{train,val,test}`.

3. Either:
   - pass `--data "path/to/Beach Bird Detection Dataset v1.0"` to every command, or
   - set `BIRD_DATA` to that path, or
   - place the dataset at `../dataset/Beach Bird Detection Dataset v1.0` relative to this `submission/` folder
     (the default search path used by the CNN scripts).

## Method 1 — non-learning (`nonlearning/`)

Four handcrafted detectors: DoG, DoH, MSER, adaptive thresholding.

```bash
cd nonlearning

# 1) pick best settings on clean validation (~20 min)
python robustness.py --tune --data "PATH/TO/Beach Bird Detection Dataset v1.0"

# 2) evaluate clean + 7 corruptions × 5 levels (run once on test)
python robustness.py --split test --data "PATH/TO/Beach Bird Detection Dataset v1.0"

# optional: draw one example (green = labels, red = predictions)
python robustness.py --split test --example 0 --corruption dark --level 5 --detectors dog doh --data "PATH/TO/..."
```

Details: see `nonlearning/README.md`.

## Method 2 — CNN (`cnn/`)

Small CenterNet-style network (~190k params), trained from scratch. Same architecture for all augmentation settings.

| `--aug` | Training |
|---------|----------|
| `none` | no augmentation (clean baseline) |
| `basic` | flip + colour jitter |
| `robust` | basic + random corruption (all 7 types) |
| `robust_seen` | basic + only noise / dark / JPEG |

```bash
cd cnn

# train one setting (saves runs/<aug>.pt — keep these local, do not zip)
python train.py --aug none --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug basic --data "PATH/TO/..."
python train.py --aug robust --epochs 80 --data "PATH/TO/..."
python train.py --aug robust_seen --epochs 80 --data "PATH/TO/..."

# evaluate on val (or --split test once at the end)
python evaluate.py runs/none.pt runs/basic.pt runs/robust.pt runs/robust_seen.pt --data "PATH/TO/..."
python evaluate.py runs/none.pt runs/robust.pt --example 5 --corruption blur --level 4 --data "PATH/TO/..."
```

Optional Colab notebook: `cnn/colab.ipynb`.

Details: see `cnn/README.md`.

## Metric

Both methods report **mAP@0.5** (VOC-style all-point AP, IoU ≥ 0.5), so Method 1 and Method 2 numbers are comparable.

Corruptions are shared (`cnn/corruptions.py`) with `seed = image index`, so every model sees the same corrupted pixels.

## Third-party libraries (not our code)

| Library | Used for |
|---------|----------|
| NumPy / OpenCV | images, classical detectors, corruptions |
| PyTorch | Method 2 CNN |
| Matplotlib | plots / example figures |

Dataset: Kai Liu, *Beach Bird Detection Dataset v1.0*, Kaggle, 2026.
