"""PyTorch bundle -> ONNX (per member, opset 17, static 1x3x224x224) -> onnxsim -> onnx2tf SavedModel (NHWC)
-> combined SavedModel `sm_ensemble` (input: normalized NHWC float32 [N,224,224,3]; outputs: per-member
logits and prob = sigmoid(mean of the two logits in float64)).  Run from this folder with the venv python.
Stages: --stage onnx | convert | combine | all (default)."""
import argparse, subprocess, sys, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
NAMES = ['convnext_tiny', 'swin_s']

def stage_onnx():
    import torch, numpy as np, onnxruntime as ort
    from deepfake_core_x1 import load_bundle, decode_rgb
    from common import BUNDLE, pool_images, normalize_nchw
    models, meta = load_bundle(BUNDLE, torch.device('cpu'))
    assert meta['combination'] == 'logit_mean' and len(models) == 2
    x = normalize_nchw([decode_rgb(p) for p in pool_images()])
    for m, n in zip(models, NAMES):
        torch.onnx.export(m, torch.zeros(1, 3, 224, 224), HERE / f'{n}.onnx', opset_version=17, dynamo=False,
                          input_names=['input'], output_names=['logit'])
        subprocess.run([str(Path(sys.executable).parent / 'onnxsim'), str(HERE / f'{n}.onnx'), str(HERE / f'{n}_s.onnx')], check=True)
        sess = ort.InferenceSession(str(HERE / f'{n}_s.onnx'), providers=['CPUExecutionProvider'])
        with torch.inference_mode():
            ref = np.concatenate([m(torch.from_numpy(x[i:i+1])).numpy() for i in range(len(x))]).ravel()
        got = np.concatenate([sess.run(None, {'input': x[i:i+1]})[0] for i in range(len(x))]).ravel()
        print(f'ONNX vs torch {n}: max|logit diff| = {np.abs(ref - got).max():.3e}')

def stage_convert():
    for n in NAMES:
        shutil.rmtree(HERE / f'sm_{n}', ignore_errors=True)
        subprocess.run([sys.executable, str(HERE / 'convert_tf.py'), n], check=True, cwd=HERE)

def stage_combine():
    import tensorflow as tf
    loaded = [tf.saved_model.load(str(HERE / f'sm_{n}')) for n in NAMES]  # kept as attribute so variables stay tracked
    subs = [l.signatures['serving_default'] for l in loaded]
    class Ens(tf.Module):
        def __init__(self):
            super().__init__(); self.loaded = loaded; self.subs = subs
        @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')])
        def __call__(self, x):
            def one(row):  # members are static batch 1
                return tf.stack([tf.reshape(s(input=row[None])['output_0'], []) for s in self.subs])
            lg = tf.map_fn(one, x, fn_output_signature=tf.float32)          # [N,2] float32 member logits
            p = tf.sigmoid(tf.reduce_mean(tf.cast(lg, tf.float64), axis=1)) # FP64 mean then sigmoid
            return {'logits': lg, 'prob': p}
    m = Ens()
    shutil.rmtree(HERE / 'sm_ensemble', ignore_errors=True)
    tf.saved_model.save(m, str(HERE / 'sm_ensemble'), signatures={'serving_default': m.__call__})
    print('saved sm_ensemble')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='all', choices=['onnx', 'convert', 'combine', 'all'])
    a = ap.parse_args()
    if a.stage in ('onnx', 'all'): stage_onnx()
    if a.stage in ('convert', 'all'): stage_convert()
    if a.stage in ('combine', 'all'): stage_combine()
