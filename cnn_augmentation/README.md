# CNN data augmentation & robustness study

This folder is the **"augmentation on the CNN"** part of the project. It takes the group's small
CenterNet-style CNN from `simple_cnn_bird_detector.ipynb` (same architecture, same layer names,
so the teammate's `tune-512-clean.pt` loads unchanged) and studies:

1. **Robustness benchmark.** Eight realistic image corruptions, each at five severities
   (ImageNet-C style): Gaussian noise, Gaussian blur, motion blur, JPEG compression,
   low brightness, low contrast, fog/haze and partial occlusion.
2. **Augmentation ablation.** The same training recipe with different augmentations, so any
   change in clean or corrupted mAP@0.5 can be attributed to the augmentation.
3. **Generalisation to unseen corruptions.** Models trained with only "sensor" corruptions
   (noise, low light, JPEG) are tested on "optical/atmospheric" ones (blur, fog, contrast,
   occlusion) that they never saw during training.

## Files

| file | purpose |
|---|---|
| `data.py` | load YOLO labels into `x1y1x2y2` boxes, encode CenterNet targets (Gaussian heatmap, log-size, offset) |
| `model.py` | `Net` (identical to the notebook), focal + L1 loss, batched peak decoding |
| `augment.py` | training augmentations: flip, scale/translate, mosaic, cutout, colour jitter, random corruptions. Boxes are transformed with the image |
| `corruptions.py` | **shared benchmark**: the 8 corruptions × 5 severities, deterministic per image. Pure numpy/OpenCV |
| `metrics.py` | mAP@0.5 (VOC all-point, same definition as the `nonlearning` branch), plus P/R, recall by bird size and count error |
| `experiments.py` | the named augmentation configs (see below) |
| `train.py` | train one config and seed. Model selection uses **clean val** only |
| `evaluate.py` | clean + corrupted evaluation of one or more checkpoints → `results/<run>_<split>.json` |
| `summarize.py` | tables (`.md`/`.csv`) and figures (mAP vs severity, model × corruption heatmap) |
| `visualize.py` | qualitative figures: corruption gallery, augmentation sanity check, prediction comparison |
| `run_all.py` | train everything → evaluate → summarise (resumable) |

## Augmentation configs (`experiments.py`)

| config | what is on |
|---|---|
| `baseline` | nothing (same recipe, no augmentation) |
| `geo` | horizontal flip + random zoom (0.75–1.5×) / translate |
| `photo` | brightness / contrast / saturation / hue jitter + gamma |
| `cutout` | 1–6 random erased rectangles (birds ≥70 % hidden are dropped from the targets) |
| `mosaic` | 4-image mosaic made of **crops, not downscaled images**, so the already tiny birds keep their size |
| `corrupt` | with p = 0.5, 1–2 random corruptions from all 8 families at a random severity |
| `geo_photo` | "standard" detection augmentation |
| `full` | geo + photo + cutout + mosaic + all corruptions |
| `corrupt_seen` | corruptions from `SEEN = {gaussian_noise, low_brightness, jpeg}` only |
| `full_seen` | geo + photo + mosaic + SEEN corruptions only (no cutout, since cutout ≈ occlusion) |

Compare `corrupt_seen` / `full_seen` against `corrupt` / `full` on the **unseen** corruptions
(the `unseen-only mAP` column of the summary table). This shows whether robustness generalises
to corruptions that weren't seen in training or only covers the ones trained on.

## Setup

```bash
pip install torch numpy opencv-python matplotlib
```

Download the dataset from Kaggle (<https://www.kaggle.com/datasets/liugalhigh/beach-bird>). `--data`
must point to the folder that contains `images/{train,val,test}` and `labels/{train,val,test}`.
You can also set it once:

```bash
export BIRD_DATA="/path/to/Beach Bird Detection Dataset v1.0"      # bash
$env:BIRD_DATA = "C:\path\to\Beach Bird Detection Dataset v1.0"   # PowerShell
```

Run every command from inside `cnn_augmentation/`.

## Usage

```bash
# 0. look at the benchmark and check that the augmentations keep boxes aligned
python visualize.py corruptions --index 3
python visualize.py augment --config full

# 1. quick sanity run (a few minutes on CPU)
python train.py --config baseline --epochs 2 --eval-every 1 --train-frac 0.1

# 2. the full study on the validation set (resumable)
python run_all.py --epochs 60
#    ... and with 3 seeds for the configs you put in the report
python run_all.py --configs baseline geo_photo full full_seen --seeds 0 1 2

# 3. also evaluate the CNN teammate's original weights on the benchmark
python evaluate.py --ckpt ../path/to/tune-512-clean.pt --names teammate_clean

# 4. FINAL numbers, once everything is fixed: test split, only once
python run_all.py --skip-train --split test --configs baseline geo_photo full full_seen --seeds 0 1 2
python summarize.py --split test --group-seeds

# 5. qualitative failure analysis
python visualize.py predict --index 3 --corruption gaussian_blur --severity 4 \
    --ckpt runs/baseline_s0/best.pt runs/full_s0/best.pt
```

### Compute

There is no GPU in our local setup. The network is small (190k params), but 60 epochs on
1,369 images of size 512² still take a while on CPU. Running on **Google Colab (free T4 GPU)** is
recommended: upload this folder and the dataset to Drive, then

```python
%cd /content/drive/MyDrive/COMP9517GroupProject/cnn_augmentation
!python run_all.py --data "/content/drive/MyDrive/beach-bird/Beach Bird Detection Dataset v1.0"
```

`--train-frac 0.5` is allowed by the spec if compute is tight (report it).

## Notes for the report

- **Metric.** mAP@0.5 (single class, so mAP = AP), VOC all-point interpolation, identical to the
  non-learning branch. Robustness summary: *mean corrupted mAP* (over 8 corruptions × 5
  severities) and *relative robustness* = mean corrupted / clean.
- **Decoding change vs the notebook.** We keep the top **200** heatmap peaks instead of 100.
  The dataset has images with up to 130 birds, so a cap of 100 limits recall. The score threshold
  for mAP is 0.05, so the whole PR curve is covered.
- **No leakage.** Training uses `train`, model selection and all development use `val`, and `test`
  is only used for the final table. Corruption severities were fixed in advance (see `PARAMS` in
  `corruptions.py`) and were not tuned on test.
- **Train/test corruption overlap.** `corrupt` and `full` see the same corruption *families*
  in training as in evaluation (at continuous random severities). This is why the held-out
  `*_seen` configs exist. Be explicit about this in the report.
- **Occlusion** keeps all GT boxes, so a fully covered bird counts as a miss. Cutout during
  training drops birds that are ≥70 % hidden.
- **Times.** `train_time_s` (in the checkpoint) and `infer_ms_per_img` (in the results JSON) go
  in the spec's required timing comparison.

## Sharing the benchmark with the non-ML method

`corruptions.py` and `metrics.py` don't depend on torch. To evaluate the non-learning detector on
exactly the same corrupted images:

```python
import sys; sys.path.append("cnn_augmentation")
from corruptions import CORRUPTIONS, SEVERITIES, corrupt_deterministic
from metrics import average_precision

for name in CORRUPTIONS:
    for s in SEVERITIES:
        preds = []
        for i, (path, gt) in enumerate(items):          # items = load_split(root, "val")
            img = cv2.imread(path)                       # BGR is fine
            img_c = corrupt_deterministic(img, name, s, image_index=i)
            gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
            preds.append(nms(*detect(gray, **params)))
        print(name, s, average_precision(preds, [g for _, g in items]))
```

The image index `i` must be the position in the sorted split list (as returned by
`load_split`), so that both methods get identical noise and occluder patterns.

## References

- X. Zhou, D. Wang, P. Krähenbühl. *Objects as Points* (CenterNet), arXiv:1904.07850, 2019 (target encoding, focal loss, `gaussian_radius`).
- D. Hendrycks, T. Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019 (corruption benchmark design).
- C. Michaelis et al. *Benchmarking Robustness in Object Detection*, arXiv:1907.07484, 2019 (corruptions for detection).
- T. DeVries, G. Taylor. *Improved Regularization of CNNs with Cutout*, arXiv:1708.04552, 2017. Z. Zhong et al. *Random Erasing Data Augmentation*, AAAI 2020.
- A. Bochkovskiy et al. *YOLOv4*, arXiv:2004.10934, 2020 (mosaic).
- D. Hendrycks et al. *AugMix*, ICLR 2020 (related work: mixing augmentation chains).
- Libraries: PyTorch, OpenCV, NumPy, Matplotlib.
