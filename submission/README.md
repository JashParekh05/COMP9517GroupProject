# Beach Bird Detection (COMP9517)

Code for two bird detectors on the Beach Bird Detection Dataset v1.0:

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

## Method 1

```bash
cd nonlearning

python robustness.py --tune --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python robustness.py --split test --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

`--tune` selects detector settings on the validation set and writes `best_params.json`.
The second command evaluates clean images and seven corruptions (five severity levels each)
on the test set.

Example visualisation (green = ground truth, red = predictions):

```bash
python robustness.py --split test --example 0 --corruption dark --level 5 --detectors dog doh --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

See `nonlearning/README.md` for more detail.

## Method 2

Train one augmentation setting at a time. Checkpoints are saved under `cnn/runs/`.

| `--aug` | Training |
|---------|----------|
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

Example visualisation:

```bash
python evaluate.py runs/none.pt runs/robust.pt --example 5 --corruption blur --level 4 --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

- `--frac 0.5` trains on half of the training images
- `--split test` on `evaluate.py` for final test numbers
- optional notebook: `cnn/colab.ipynb`

See `cnn/README.md` for more detail.

## Metric

Both methods report **mAP@0.5** (VOC-style average precision at IoU 0.5).

Corruptions are defined in `cnn/corruptions.py` and shared with Method 1. The seed for each
corrupted image is the image index, so both methods see the same corrupted pixels.

## Libraries

| Library | Used for |
|---------|----------|
| NumPy, OpenCV | image I/O, classical detectors, corruptions |
| PyTorch | Method 2 CNN |
| Matplotlib | plots and example figures |

## Dataset

Kai Liu. Beach Bird Detection Dataset v1.0. Kaggle, 2026.  
https://www.kaggle.com/datasets/liugalhigh/beach-bird
