# Method 2: CNN detector

Small CenterNet-style network trained from scratch (~190k parameters). The same architecture is used for every augmentation setting.

## Files

| File | Role |
|------|------|
| `common.py` | dataset loading, network, loss, box decode, mAP@0.5 |
| `corruptions.py` | seven corruptions at five severity levels |
| `train.py` | train with one `--aug` setting |
| `evaluate.py` | evaluate on clean and corrupted images |
| `colab.ipynb` | optional Colab notebook |

## Augmentation settings

| `--aug` | Training |
|---------|----------|
| `none` | no augmentation |
| `basic` | horizontal flip and colour jitter |
| `robust` | basic, plus 50% chance of any of the seven corruptions |
| `robust_seen` | basic, plus 50% chance of noise, dark, or JPEG only |

## Run

Install dependencies from the parent folder, then:

```bash
pip install -r ../requirements.txt
cd cnn

python train.py --aug none --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug basic --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug robust --epochs 80 --data "PATH/TO/Beach Bird Detection Dataset v1.0"
python train.py --aug robust_seen --epochs 80 --data "PATH/TO/Beach Bird Detection Dataset v1.0"

python evaluate.py runs/none.pt runs/basic.pt runs/robust.pt runs/robust_seen.pt --data "PATH/TO/Beach Bird Detection Dataset v1.0"
```

Useful flags:

- `--frac 0.5`: train on half of the training images
- `--epochs`: default is 30; use 80 for the robust settings
- `--split test` on `evaluate.py`: use once for final test numbers

Checkpoints are written to `runs/<aug>.pt`. That folder is created locally and is not part of the submitted ZIP.

## Notes

- mAP@0.5 matches Method 1, so scores are comparable.
- Up to 200 detections per image (some images contain up to 130 birds).
- Corruption seed equals the image index, so evaluations stay aligned across models.

## References

- Zhou et al., Objects as Points (CenterNet), 2019
- Hendrycks and Dietterich, Benchmarking Neural Network Robustness to Common Corruptions and Perturbations, 2019
- Michaelis et al., Benchmarking Robustness in Object Detection, 2019
