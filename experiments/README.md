# Follow-up experiments (not adopted)

After the final-v3 pair (ConvNeXt-Tiny + Swin-S, ImageNet-1K initial weights) was chosen, three further experiments were run. Each had its acceptance gate fixed before any result was seen, and the gates were not changed afterwards. None passed, so the shipped model is unchanged.

All numbers are differences in the local J04 proxy (original AUC + 0.4 x weighted corruption AUC; see the main README and `docs/REPORT.md` section 6) against the shipped pair, measured on the same cross-validation folds used for every earlier design decision. Fold-to-fold retraining variance is about +/-0.004 in dJ04, as large as the effects below.

| Experiment | Change | Result | Outcome |
|---|---|---|---|
| P22 | ImageNet-21k/22k initial weights instead of ImageNet-1K; five backbones screened on folds 0 and 1 (10 fits) | dJ04 was negative on fold 1 for every member. Pair means: ConvNeXt-T -0.0012, ConvNeXt-S -0.0012, Swin-S -0.0023, Swin-B -0.0023, ViT-B/16 -0.0078 | No member promoted |
| P23 | Sharpness-Aware Minimization (rho 0.05, AdamW base) for both members, 5 folds (10 fits) | Mean dJ04 +0.0009 (bar +0.004); 2/5 folds positive (bar 4/5); worst fold -0.0026 (bar -0.002); mean change in original AUC +0.0005 | Failed |
| P24 | Cascaded corruption augmentation: with probability 0.8, 1 to 3 distinct corruption families (probabilities 0.5/0.3/0.2) at severities 0 to 2, 5 folds (10 fits) | Mean dJ04 +0.0023 (bar +0.004); 3/5 folds positive; worst fold -0.0035; mean change in original AUC +0.0013. An independent corruption registry improved by +0.0022 and held-out corruption families were non-negative, but the primary gates failed | Failed |

A post hoc, report-only check found that a 6-member logit ensemble of all these fine-tunes reached only +0.0036 mean dJ04 at three times the inference cost, so weight averaging was not pursued.

## Files

- `p24_augment.py`: the P24 training view (`cascade_view`). It imports `FAMILIES`, `SEVERITY_WEIGHTS`, `TRAIN_FAMILIES`, `stable_seed` and `transform` from `training/e1_transforms.py`, so run it with that folder on `PYTHONPATH`. It replaces the standard training view for P24 fits only.

## P23 (SAM) in brief

At each step, the gradient is computed at the current weights, the weights are moved by `rho * g / ||g||` along that gradient, a second forward and backward pass is run at the perturbed weights on the same batch, the weights are restored exactly, and the optimizer steps with the second gradient. Steps with non-finite gradients are skipped and the mixed-precision scale is reduced. Everything else (seed 17, GCE q=0.7, EMA, schedule, ImageNet-1K starts) matched the final-v3 recipe. Training time roughly doubles. The P23 training scripts and the host packaging scripts are not included in this repository.
