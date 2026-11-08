# CNN augmentation: making the bird CNN robust to bad images

We take the group's CNN (from `simple_cnn_bird_detector.ipynb`) and ask two questions:

1. **How badly does it break** on noisy, blurry, dark, low-contrast, foggy, JPEG-compressed
   or partly covered images?
2. **Does data augmentation fix that?** And if we only train with some kinds of damage,
   does the model also get better at kinds it never saw?

## Files

| file | what it does |
|---|---|
| `common.py` | loading data, the CNN (same as the notebook), loss, boxes from the heatmap, mAP@0.5 |
| `corruptions.py` | 7 corruptions × 5 levels. Only numpy/OpenCV, so the non-ML method can use it too |
| `train.py` | trains the CNN with one augmentation setting |
| `evaluate.py` | tests models on clean + corrupted images and prints a table and a plot |

## The four augmentation settings

| `--aug` | training images get... |
|---|---|
| `none` | nothing |
| `basic` | random horizontal flip + brightness/contrast/saturation changes |
| `robust` | basic + 50 % chance of a random corruption (any of the 7) |
| `robust_seen` | basic + 50 % chance of a corruption, but only noise, dark or jpeg |

Comparing `robust_seen` and `robust` on blur, contrast, fog and occlusion (the "unseen"
column) shows whether robustness carries over to kinds of damage the model never saw.

## How to run

```powershell
pip install torch numpy opencv-python matplotlib
cd cnn_augmentation
$env:BIRD_DATA = "G:\My Drive\ETH\Semester 5 (UNSW)\CV\assignments\Group Project\dataset\Beach Bird Detection Dataset v1.0"

python corruptions.py "$env:BIRD_DATA\images\val\<some image>.jpg"   # picture of all corruptions

python train.py --aug none          # each takes about 1.5-2 h on a laptop CPU (30 epochs)
python train.py --aug basic
python train.py --aug robust
python train.py --aug robust_seen

python evaluate.py runs/none.pt runs/basic.pt runs/robust.pt runs/robust_seen.pt
python evaluate.py runs/none.pt runs/robust.pt --example 5 --corruption blur --level 4
```

Add `--frac 0.5` to `train.py` to train on half the images if time is short.
Only run `evaluate.py ... --split test` once, at the very end, to get the final numbers.

## Notes for the report

- **mAP@0.5** uses the same definition as the non-learning branch (VOC all-point), so the
  numbers can be compared directly.
- Up to **200** boxes per image instead of the notebook's 100, because some images have
  up to 130 birds.
- Each corrupted image is generated with `seed = image index`, so every model (and the non-ML
  method) is tested on exactly the same pixels.
- `robust` trains with the same corruption types it is tested on. That's why `robust_seen`
  exists: it's the fair test.

## References

- Zhou et al., *Objects as Points* (CenterNet), 2019: heatmap targets and focal loss.
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions*, ICLR 2019.
- Michaelis et al., *Benchmarking Robustness in Object Detection*, 2019.
