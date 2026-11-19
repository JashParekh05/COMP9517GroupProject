# Beach Bird Detection (COMP9517)

Thank you for trying our code. This package has two methods for detecting birds in beach and wetland images:

1. **Method 1** (`nonlearning/`): classical detectors with no training (DoG, DoH, MSER, adaptive thresholding)
2. **Method 2** (`cnn/`): a small CenterNet-style CNN trained from scratch, with optional data augmentation

## Requirements

- Python 3.10 or newer
- Beach Bird Detection Dataset v1.0 in this layout:

```
Beach Bird Detection Dataset v1.0/
  images/train|val|test/
  labels/train|val|test/
```

Install the Python packages:

```bash
pip install -r requirements.txt
```

## Dataset path

Pass the dataset root (the folder that contains `images/` and `labels/`) with `--data`:

```bash
--data "C:/data/Beach Bird Detection Dataset v1.0"
```

You can also set `BIRD_DATA` to that path, or place the dataset at
`../dataset/Beach Bird Detection Dataset v1.0` relative to this folder.

## Quick start: Method 1

```bash
cd nonlearning

python robustness.py --tune --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python robustness.py --split test --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

`--tune` chooses detector settings on the validation set and writes `best_params.json`.
The next command evaluates clean images and seven corruptions (five severity levels each)
on the test set.

Optional example image (green = ground truth, red = predictions):

```bash
python robustness.py --split test --example 0 --corruption dark --level 5 --detectors dog doh --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

See also `nonlearning/README.md`.

## Quick start: Method 2

Train one augmentation setting at a time. Checkpoints are saved under `cnn/runs/`.

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

Tips:

- add `--frac 0.5` for a shorter training run
- use `--split test` on `evaluate.py` for the final test numbers
- optional notebook: `cnn/colab.ipynb`

See also `cnn/README.md`.

## Evaluation metric

Both methods report **mAP@0.5** (VOC-style average precision at IoU 0.5), so the scores can be compared directly.

Corruptions live in `cnn/corruptions.py` and are shared with Method 1. The seed for each
corrupted image is the image index, so every method sees the same corrupted pixels.

## External libraries

| Library | Used for |
|---------|----------|
| NumPy, OpenCV | image I/O, classical detectors, corruptions |
| PyTorch | Method 2 CNN |
| Matplotlib | plots and example figures |

## Dataset citation

Kai Liu. Beach Bird Detection Dataset v1.0. Kaggle, 2026.  
https://www.kaggle.com/datasets/liugalhigh/beach-bird
