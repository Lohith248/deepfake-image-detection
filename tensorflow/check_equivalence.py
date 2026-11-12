"""Self-check: TF SavedModel vs PyTorch bundle on the first 64 pool_manifest images (train/ only).
PASS iff max |prob diff| <= 1e-4 and labels identical.  Run: python check_equivalence.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from deepfake_core_x1 import load_bundle, decode_rgb, to_tensor
from common import BUNDLE, pool_images
import inference_tf
paths = pool_images(64); assert len(paths) == 64 and all('test' not in p.parts[-3:-2] for p in paths)
models, meta = load_bundle(BUNDLE, torch.device('cpu'))
t = time.perf_counter()
with torch.inference_mode():
    lg = np.stack([np.concatenate([m(torch.stack([to_tensor(decode_rgb(p)) for p in paths[i:i+16]])).reshape(-1).numpy()
                                   for i in range(0, 64, 16)]) for m in models], 1)  # [64,2] FP32 member logits
t_torch = time.perf_counter() - t
p_torch = torch.sigmoid(torch.from_numpy(lg).double().mean(1)).numpy()
fn = inference_tf.load(HERE := Path(__file__).resolve().parent / 'sm_ensemble')
inference_tf.score(fn, paths[:2], 2)  # warm-up
t = time.perf_counter(); ids, p_tf, lg_tf = inference_tf.score(fn, paths, 16); t_tf = time.perf_counter() - t
p_tf = np.array(p_tf)
dp = np.abs(p_tf - p_torch).max(); dl = np.abs(lg_tf - lg).max(0)
same = bool(((p_tf >= .5) == (p_torch >= .5)).all())
print(f'images: {len(paths)}  (train pool, first 64 manifest rows)')
print(f'max |logit diff| convnext_tiny: {dl[0]:.3e}   swin_s: {dl[1]:.3e}')
print(f'max |prob diff|: {dp:.3e}   labels identical: {same}   positives: {int((p_torch>=.5).sum())}/64')
print(f'CPU time 64 imgs (excl. decode warm-up): TF {t_tf:.1f}s   PyTorch {t_torch:.1f}s')
ok = dp <= 1e-4 and same
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
