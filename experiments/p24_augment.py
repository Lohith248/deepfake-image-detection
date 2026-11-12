"""P24 training view: cascaded corruption augmentation (replaces e1_transforms.training_view for P24 fits only; e1_transforms.py is NOT edited).
Same stable_seed RNG keying as training_view with key prefix 'train-p24:'.
  1. horizontal flip with p=.5;
  2. with p=.80 a cascade: k in {1,2,3} with probs .5/.3/.2; k DISTINCT families drawn from e1_transforms.TRAIN_FAMILIES (jpeg, resolution, photometric, tone,
     occlusion, noise; convolution and geometric stay held out); for each, in drawn order: operator = rng.choice(FAMILIES[family]), severity in {0,1,2} with
     weights e1_transforms.SEVERITY_WEIGHTS (.5/.35/.15), result = e1_transforms.transform(result, operator, severity, rng.randrange(2**63));
  3. otherwise unchanged. Returns uint8 HxWx3 of the input shape.
"""
import random

import numpy as np

from e1_transforms import FAMILIES, SEVERITY_WEIGHTS, TRAIN_FAMILIES, stable_seed, transform

AUGMENT_ID = 'cascade80_k123_sev012_train6'
KEY_PREFIX = 'train-p24:'
FLIP_P = .5
APPLY_P = .80
K_VALUES = (1, 2, 3)
K_PROBS = (.5, .3, .2)


def cascade_view(array, image_id, epoch, seed, fold):
    rng = random.Random(stable_seed(f'{KEY_PREFIX}{seed}:{fold}:{epoch}:{image_id}'))
    result = array[:, ::-1].copy() if rng.random() < FLIP_P else array.copy()
    if rng.random() < APPLY_P:
        k = rng.choices(K_VALUES, weights=K_PROBS)[0]
        for family in rng.sample(TRAIN_FAMILIES, k):
            operator = rng.choice(FAMILIES[family])
            severity = rng.choices((0, 1, 2), weights=SEVERITY_WEIGHTS)[0]
            result = transform(result, operator, severity, rng.randrange(2**63))
    return np.asarray(result, dtype=np.uint8)
