# Beach Bird Detection: Non-Learning Detectors and Robustness

COMP9517 group project. This branch contains the **non-learning (hand-crafted) bird
detectors** and the **robustness evaluation** that tests them under image corruptions.
The CNN method and its augmentation study are on the `cnn-augmentation` branch; both
use the same corruption code (`cnn_augmentation/corruptions.py`), so their results are
directly comparable.

## Repository structure

```
non-learning.py                  4 blob detectors (DoG, DoH, MSER, adaptive threshold),
                                 non-maximum suppression, mAP@0.5 and the parameter grids
robustness.py                    tunes the detectors on clean validation images and evaluates
                                 them on clean + 7 corruptions x 5 levels
robustness_nonlearning.ipynb     walkthrough of the robustness study with explanations,
                                 figures and the checks behind each finding
cnn_augmentation/corruptions.py  the 7 image corruptions (shared with the CNN pipeline)
results_nonlearning/             saved results (best parameters, mAP tables, plots, examples)
requirements.txt                 Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Download the **Beach Bird Detection Dataset v1.0** from Kaggle
(https://www.kaggle.com/datasets/liugalhigh/beach-bird) and place it (or a symlink to it)
at `dataset/` in the repository root:

```
dataset/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt     YOLO format: class cx cy w h (normalised)
```

The dataset is not included in the repository.

## How to run

### 1. Tune the detectors (clean validation images)
```bash
python robustness.py --tune
```
Grid search over 39 settings (`GRIDS` in `non-learning.py`) on all 291 validation
images. The best setting per detector is saved to `results_nonlearning/best_params.json`.
Takes roughly 20-30 minutes on an 8-core laptop.

### 2. Robustness evaluation
```bash
python robustness.py                  # validation set
python robustness.py --split test     # test set: final numbers, run once at the end
```
Runs every detector on clean images and on 7 corruptions (noise, blur, JPEG, dark,
contrast, fog, occlusion) at 5 severity levels. Prints a table and saves
`results_nonlearning/results_<split>.json` and `results_<split>.png`.
Takes roughly 1 hour on an 8-core laptop.

### 3. Example pictures
```bash
python robustness.py --example 0 --corruption dark --level 5 --detectors dog doh
```
Draws ground truth (green) and the top predictions (red) for one image. Omit
`--corruption` for the clean image.

### Useful options
| Option | Meaning |
|---|---|
| `--detectors dog doh` | only run some detectors (`dog`, `doh`, `mser`, `threshold`) |
| `--limit 30` | only use the first N images (quick check) |
| `--workers 1` | disable parallelism (single-core timing) |
| `--data PATH` | dataset folder if it is not at `dataset/` |

### Notebook
Open `robustness_nonlearning.ipynb` in Jupyter. It loads the saved results, so it does
not repeat the one-hour evaluation, and it re-runs the smaller checks that explain the
results (e.g. why DoG fails on dark images).

### Quick check of the detectors
```bash
python non-learning.py
```
Runs the parameter grid on the first 20 validation images and prints the mAP of each
setting.

## Evaluation protocol

- **Metric:** mAP@0.5 with VOC all-point interpolation. A prediction is correct if it
  overlaps an unmatched ground-truth box with IoU >= 0.5. There is one class, so mAP = AP.
- **No test-set tuning:** detector settings are chosen on clean validation images only.
  The test set is evaluated once with those settings.
- **Identical corrupted images for both methods:** each image is read in RGB at 512x512,
  corrupted with `seed = image index`, then converted to grey-scale for the detectors.
  The CNN evaluation uses the same order, seed and channel order.
- **DoH box refinement** is disabled in the robustness study (about 19x slower for
  +0.004 validation mAP); this is recorded in `best_params.json`.
- Timings are wall-clock per image with 8 worker processes (use `--workers 1` for
  single-core timing).

## Results (mAP@0.5)

**Test set** (240 images, evaluated once with the validation-tuned settings):

| Detector | Clean | Noise | Blur | JPEG | Dark | Contrast | Fog | Occlusion |
|---|---|---|---|---|---|---|---|---|
| DoG | 0.034 | 0.034 | 0.033 | 0.032 | 0.024 | 0.024 | 0.033 | 0.015 |
| DoH | 0.032 | 0.032 | 0.035 | 0.032 | 0.030 | 0.030 | 0.032 | 0.014 |
| MSER | 0.002 | 0.002 | 0.005 | 0.002 | 0.001 | 0.001 | 0.002 | 0.001 |
| Threshold | 0.002 | 0.003 | 0.005 | 0.002 | 0.002 | 0.002 | 0.003 | 0.001 |

**Validation set** (291 images):

| Detector | Clean | Noise | Blur | JPEG | Dark | Contrast | Fog | Occlusion |
|---|---|---|---|---|---|---|---|---|
| DoG | 0.033 | 0.036 | 0.035 | 0.033 | 0.024 | 0.024 | 0.033 | 0.014 |
| DoH | 0.032 | 0.032 | 0.038 | 0.032 | 0.031 | 0.032 | 0.032 | 0.013 |
| MSER | 0.002 | 0.002 | 0.005 | 0.002 | 0.001 | 0.001 | 0.002 | 0.001 |
| Threshold | 0.005 | 0.005 | 0.008 | 0.004 | 0.005 | 0.005 | 0.006 | 0.002 |

Corruption columns are averaged over levels 1-5. Per-level results are in
`results_nonlearning/results_{val,test}.json`.

**Main findings**
- Occlusion causes the largest drop for every detector (DoG 0.034 to 0.004 at level 5).
- DoG collapses on dark and low-contrast images (0.003 at level 5) because its fixed
  response threshold (0.02) removes every candidate. Scaling the threshold by the same
  factor as the image (to 0.003) restores test mAP to 0.033.
- Noise, JPEG and fog change mAP by at most 0.003; mild blur slightly improves DoH.

## Libraries and sources

**Libraries:** NumPy, OpenCV (`opencv-python`), Matplotlib, Jupyter.

**Methods implemented, with the papers they follow:**
- Difference of Gaussians scale-space detector: D. G. Lowe, "Distinctive Image Features
  from Scale-Invariant Keypoints", IJCV 2004; T. Lindeberg, "Feature Detection with
  Automatic Scale Selection", IJCV 1998.
- Determinant of Hessian detector: H. Bay et al., "Speeded-Up Robust Features (SURF)",
  CVIU 2008.
- MSER (`cv2.MSER_create`): J. Matas et al., "Robust Wide Baseline Stereo from Maximally
  Stable Extremal Regions", BMVC 2002.
- CLAHE (`cv2.createCLAHE`): K. Zuiderveld, "Contrast Limited Adaptive Histogram
  Equalization", Graphics Gems IV, 1994.
- Otsu thresholding (`cv2.threshold`): N. Otsu, "A Threshold Selection Method from
  Gray-Level Histograms", IEEE Trans. SMC, 1979.
- Corruption benchmark design: D. Hendrycks and T. Dietterich, "Benchmarking Neural
  Network Robustness to Common Corruptions and Perturbations", ICLR 2019.
- mAP definition: M. Everingham et al., "The PASCAL Visual Object Classes (VOC)
  Challenge", IJCV 2010.

Non-maximum suppression uses `cv2.dnn.NMSBoxes`. All other code in this branch was
written by the group.
