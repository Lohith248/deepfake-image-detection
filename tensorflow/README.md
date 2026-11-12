# TensorFlow export of the final-v3 model

The PyTorch `inference_script.py` in the repository root is the primary deliverable. The TensorFlow SavedModel is an equivalent alternative, not a replacement.

## Where to get the SavedModel

The SavedModel (`sm_ensemble/`, about 298 MB) is not stored in this repository. It is available as a release asset named `sm_ensemble.zip` on the v1.0 release. Unzip it so that the `sm_ensemble/` folder is next to `inference_tf.py`.

## What it is

- Source: the final-v3 bundle (`model_weights.pt`): ConvNeXt-Tiny + Swin-S, equal-logit-mean ensemble.
- Path: PyTorch -> ONNX (opset 17, static 1x3x224x224, simplified with onnxsim) -> onnx2tf SavedModel (NHWC), one per member, then combined into `sm_ensemble`.
- Input: normalised NHWC float32 `[N,224,224,3]`. Outputs: per-member `logits` and `prob` = sigmoid of the float64 mean of the two logits.
- Preprocessing matches the PyTorch script (`decode_rgb`, /255, ImageNet mean/std) except for HWC layout.

## Scripts

- `export_tf.py` (`--stage onnx|convert|combine|all`): builds the ONNX files, calls `convert_tf.py` per member, combines the SavedModels.
- `convert_tf.py <name>`: onnx2tf conversion. Replaces onnx2tf's sample-image download with random data.
- `check_equivalence.py`: compares the SavedModel with the PyTorch bundle. `common.py`: image list and preprocessing helpers.
- `inference_tf.py`: command-line inference with the same output format as the PyTorch script.

Usage:

    PYTHONPATH=.. python inference_tf.py --test_dir DIR --weights sm_ensemble --output predictions.csv

It needs tensorflow, numpy, Pillow and torch. Torch is imported by `deepfake_core_x1` (in the repository root, hence `PYTHONPATH=..`) but is not used for compute.

The export and check scripts were run from the original working tree. `common.py` resolves `ROOT` as the parent of this folder and expects the model bundle and a 64-image training-pool manifest at paths in that tree (`BUNDLE`, `MANIFEST`); adjust those constants to your own locations before running `export_tf.py` or `check_equivalence.py`. Training images are not distributed with this repository.

## Equivalence results

Checked on 64 training-pool images. No test images were used.

| Check | Result |
|---|---|
| ONNX vs torch, max abs logit diff | ConvNeXt-T 1.81e-5, Swin-S 1.79e-5 |
| TF vs PyTorch, max abs logit diff | ConvNeXt-T 1.34e-5, Swin-S 2.00e-5 |
| TF vs PyTorch, max abs probability diff | 5.5e-7 |
| Labels identical | yes (28/64 positive) |
| CPU time, 64 images | TF 11.7 s, PyTorch 11.0 s |

Pass rule in `check_equivalence.py`: max probability diff <= 1e-4 and identical labels. Result: pass. The CPU timing is one run on one machine and is not a T4 measurement.

## SHA-256 of `sm_ensemble`

    2dc56fba1fa36c76ef5b16fb32bfaff65c933f69cbdd02c7ad445546bdd60495  fingerprint.pb
    d512ca1410037e470026fc8e3130194e16a89dfa8a5a538fde4b656ba450d643  saved_model.pb
    611e01ee5e06af5a20a27b9a6c1a1e01a0cb4239867df9dae84f61423c275025  variables/variables.data-00000-of-00001
    ad2f553522afdfe3bbe162330072d51aacc3b7f997169de3c68eac7c673a64f4  variables/variables.index

Library versions used for the conversion were not recorded.
