# Beach Bird Detection (COMP9517)

This package contains the code for two bird detectors on the Beach Bird Detection Dataset v1.0:

1. **Method 1** (`nonlearning/`): classical detectors with no training (DoG, DoH, MSER, adaptive thresholding)
2. **Method 2** (`cnn/`): a small CenterNet-style CNN trained from scratch, with optional data augmentation

Trained weights and the dataset are not included. Download the dataset yourself, then train or run the scripts as shown below.

## Requirements

- Python 3.10 or newer
- Beach Bird Detection Dataset v1.0, with this layout:

```
Beach Bird Detection Dataset v1.0/
  images/train|val|test/
  labels/train|val|test/
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Pointing to the dataset

Every script accepts `--data` with the path to the dataset root (the folder that contains `images/` and `labels/`).

Example:

```bash
--data "C:/data/Beach Bird Detection Dataset v1.0"
```

Alternatives:

- set environment variable `BIRD_DATA` to that path, or
- place the dataset at `../dataset/Beach Bird Detection Dataset v1.0` relative to this folder (default path used by the CNN scripts)

## Quick start: Method 1

```bash
cd nonlearning

python robustness.py --tune --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python robustness.py --split test --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

`--tune` chooses detector settings on the validation set and writes `best_params.json`. The second command evaluates clean images and seven corruptions (five severity levels each) on the test set.

Optional example image (green = ground truth, red = predictions):

```bash
python robustness.py --split test --example 0 --corruption dark --level 5 --detectors dog doh --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

More detail: `nonlearning/README.md`.

## Quick start: Method 2

Train one augmentation setting at a time. Checkpoints are saved under `cnn/runs/` (create that folder locally; it is not part of the ZIP).

| `--aug` | What training does |
|---------|--------------------|
| `none` | no augmentation |
| `basic` | horizontal flip and colour jitter |
| `robust` | basic, plus a random corruption from all seven types |
| `robust_seen` | basic, plus a random corruption from noise, dark, or JPEG only |

```bash
cd cnn

python train.py --aug none --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug basic --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug robust --epochs 80 --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug robust_seen --epochs 80 --data "PATH/TO/Beach Bird Detection Dataset v1.0"

python evaluate.py runs/none.pt runs/basic.pt runs/robust.pt runs/robust_seen.pt --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

Optional single-image check:

```bash
python evaluate.py runs/none.pt runs/robust.pt --example 5 --corruption blur --level 4 --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

Use `--split test` on `evaluate.py` only for the final test numbers. For a shorter training run, add `--frac 0.5`.

There is also an optional notebook: `cnn/colab.ipynb`.

More detail: `cnn/README.md`.

## Evaluation metric

Both methods report **mAP@0.5** (VOC-style average precision at IoU 0.5), so the scores can be compared directly.

Corruptions are implemented in `cnn/corruptions.py` and reused by Method 1. The random seed for each corrupted image is the image index, so every method sees the same corrupted pixels.

## External libraries

| Library | Used for |
|---------|----------|
| NumPy, OpenCV | image I/O, classical detectors, corruptions |
| PyTorch | Method 2 CNN |
| Matplotlib | plots and example figures |

## Dataset citation

Kai Liu. Beach Bird Detection Dataset v1.0. Kaggle, 2026.  
https://www.kaggle.com/datasets/liugalhigh/beach-bird
