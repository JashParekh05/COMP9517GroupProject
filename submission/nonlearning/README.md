# Method 1: non-learning detectors and their robustness

Four hand-crafted bird detectors that need no training, tested on clean images and on the
same seven corruptions (five levels each) as the CNN.

| File | What it does |
|---|---|
| `detectors.py` | the four detectors (DoG, DoH, MSER, adaptive threshold), NMS, mAP@0.5 and the parameter grids |
| `robustness.py` | tunes the detectors on clean validation images, tests them under every corruption, draws example predictions |
| `best_params.json` | the settings chosen by `--tune` (used by the other commands) |

The corruptions come from `../cnn/corruptions.py`, so both methods are tested on exactly
the same corrupted pixels (same images, same order, seed = image index, corrupted in RGB).

## Run

Requirements: `numpy`, `opencv-python`, `matplotlib`. The dataset is found automatically in
the same place as for the CNN; otherwise pass `--data "path/to/Beach Bird Detection Dataset v1.0"`.

```bash
python robustness.py --tune            # 1) pick the best setting per detector on clean val (~20 min)
python robustness.py --split test      # 2) clean + 7 corruptions x 5 levels on test (~1 h, run once)
python robustness.py --split test --example 0 --corruption dark --level 5 --detectors dog doh
```

Step 2 prints a table and saves `results/results_test.json` and `results/results_test.png`.
The example command saves a picture with ground truth (green) and predictions (red).
`--detectors dog doh` runs only some detectors, `--workers 1` turns off parallelism.

Notes:
- Settings are chosen on clean validation images only; the test set is evaluated once.
- DoH with box refinement is left out of the search: it scored 0.036 instead of 0.032 on
  clean validation images but is about 19x slower.
- Times are per image with all CPU cores in use.

## Test results (mAP@0.5, corruptions averaged over levels 1-5)

| Detector | Clean | Noise | Blur | JPEG | Dark | Contrast | Fog | Occlusion |
|---|---|---|---|---|---|---|---|---|
| DoG | 0.034 | 0.034 | 0.033 | 0.032 | 0.024 | 0.024 | 0.033 | 0.015 |
| DoH | 0.032 | 0.032 | 0.035 | 0.032 | 0.030 | 0.030 | 0.032 | 0.014 |
| MSER | 0.002 | 0.002 | 0.005 | 0.002 | 0.001 | 0.001 | 0.002 | 0.001 |
| Threshold | 0.003 | 0.003 | 0.008 | 0.003 | 0.003 | 0.003 | 0.003 | 0.001 |

## Sources

- DoG: D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004;
  T. Lindeberg, "Feature Detection with Automatic Scale Selection", IJCV 1998.
- DoH: H. Bay et al., "Speeded-Up Robust Features (SURF)", CVIU 2008.
- MSER (`cv2.MSER_create`): J. Matas et al., BMVC 2002.
- CLAHE (`cv2.createCLAHE`): K. Zuiderveld, Graphics Gems IV, 1994.
- Otsu thresholding (`cv2.threshold`): N. Otsu, IEEE Trans. SMC, 1979.
- NMS uses `cv2.dnn.NMSBoxes`.
