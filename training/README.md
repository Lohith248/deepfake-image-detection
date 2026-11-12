# Training code: final-v3 deepfake classifier (ConvNeXt-Tiny + Swin-S ensemble)

This folder holds the training, screening and release-check code. The shared core `deepfake_core_x1.py`, `inference_script.py` and `requirements.txt` live in the repository root; run training with the repository root on `PYTHONPATH` (e.g. `PYTHONPATH=.. python scripts/train_final_v3.py ...` from this folder).

Binary classifier (0 = real, 1 = AI-manipulated) for 224x224 face images. Two ImageNet-1K-initialised
backbones (ConvNeXt-Tiny, Swin-S) are fine-tuned on the provided training images only; the final score is
the mean of the two logits passed through a sigmoid, thresholded at 0.5. Python 3.10, PyTorch 2.7.1.
Contains no data, no weights and no credentials. The scripts under `scripts/` were written for the original training machine (Linux); some default paths and the frozen protocol files expect that layout.

## Inference (from the repository root)
    pip install -r requirements.txt
    python inference_script.py --test_dir <flat folder of .jpg> --weights model_weights.pt --output predictions.csv
Put `model_weights.pt` (and `release_policy.json`, which enables the verified CPU fallback) next to the weights;
`deepfake_core_x1.py` must be next to `inference_script.py`. `--device cpu` forces CPU.
Output header: `image_id,label,confidence`.

## Files
| File | Role |
|---|---|
| `../deepfake_core_x1.py` (root) | Model construction, preprocessing, bundle loading, prediction, CSV writing (used by inference and training) |
| `deepfake_core.py` | Earlier version of the shared core (hash-pinned in provenance) |
| `e1_transforms.py` | Fixed label-independent train/validation transforms and corruptions |
| `e1_metrics.py` | Exact weighted AUC with ties |
| `scripts/x1_setup_weights.py` | One-time download of torchvision ImageNet-1K starting weights |
| `scripts/train_e1.py`, `scripts/train_p1.py` | Earlier screening experiments (fit recipe the final run reuses) |
| `scripts/analyze_p1.py` | Audit of the P1/P2 screening experiments |
| `scripts/train_final_v3.py` | Final training: pool fit, one holdout readout, all-data refit for both members |
| `scripts/final_release_v3.py` | Packs the two checkpoints into `model_weights.pt` and runs release gates |
| `scripts/e1_checks.py`, `contract_checks.py`, `final_runner_checks_v3.py`, `final_interface_checks_v3.py`, `final_cli_runner_v3.py` | CPU checks of transforms, the score contract and the command line |
| `scripts/remote_final_bootstrap_v3.sh` | Order in which the steps above were run on the training machine |
| `experiments/` | Frozen protocol files (e1, p1, l1, x1, final-v3); the last two are redacted copies |
| `scripts/release_p1.py` | Prediction-file reader imported by `final_release_v3.py` (redacted copy) |
| `REDACTIONS.md` | Lists the redacted files with original and redacted hashes |

## Training (how final-v3 was produced)
1. Install the pinned environment (Python 3.10, torch 2.7.1+cu128, torchvision 0.22.1, numpy 2.2.6, Pillow 11.2.1), GPU with CUDA 12.8.
2. `python scripts/x1_setup_weights.py --archs convnext_tiny swin_s` (fetches initial weights).
3. Run the check scripts, then `python scripts/train_final_v3.py --dry-run`, then `--execute-frozen`.
   See `scripts/remote_final_bootstrap_v3.sh` for the exact sequence.
4. Training data is the organiser's training set (7000 images, real and fake); it is not included. The scripts expect
   the manifests and the frozen final protocol file next to the repository root.

## Redacted files
Four files (`experiments/final_protocol_v3.json`, `experiments/x1_protocol_v2.json`, `experiments/l1_protocol.json`,
`scripts/release_p1.py`) are included as redacted copies: the training machine's address, username and home
path were replaced by placeholders (`<GPU_HOST>`, `<USER>`, `<HOME>`, `<private access>`). Nothing else changed.
Because of this their file hashes differ from the originals; see `REDACTIONS.md`. To re-run training end to end, set
your own host and paths in `final_protocol_v3.json` and the default paths in `release_p1.py`.
