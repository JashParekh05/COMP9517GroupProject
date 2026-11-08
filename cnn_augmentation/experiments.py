"""Named augmentation configurations (the ablation study).

Each config is trained with the *same* recipe (epochs, optimiser, seed), so the
only difference between runs is the augmentation -> differences in clean and
corrupted mAP can be attributed to the augmentation.
"""
from corruptions import CORRUPTIONS

ALL = list(CORRUPTIONS)

# Held-out-corruption split: train with "sensor" degradations only, then test on
# "optical / atmospheric / occlusion" degradations the model never saw.
SEEN = ["gaussian_noise", "low_brightness", "jpeg"]
UNSEEN = [c for c in ALL if c not in SEEN]

GEO = {"flip": True, "scale_translate": True}

CONFIGS = {
    # --- single-technique ablation ---
    "baseline": {},                                         # no augmentation at all
    "geo":      dict(GEO),                                  # flip + zoom/translate
    "photo":    {"photo": True},                            # colour jitter + gamma
    "cutout":   {"cutout": True},                           # random erasing (occlusion)
    "mosaic":   {"mosaic": 0.5},                            # 4-image mosaic, no rescale
    "corrupt":  {"corrupt": ALL, "corrupt_p": 0.5},         # all corruption families

    # --- combinations ---
    "geo_photo": {**GEO, "photo": True},                    # "standard" detection aug
    "full":      {**GEO, "photo": True, "cutout": True, "mosaic": 0.5,
                  "corrupt": ALL, "corrupt_p": 0.5},

    # --- generalisation to unseen corruptions ---
    "corrupt_seen": {"corrupt": SEEN, "corrupt_p": 0.5},
    "full_seen":    {**GEO, "photo": True, "cutout": False, "mosaic": 0.5,
                     "corrupt": SEEN, "corrupt_p": 0.5},
}
