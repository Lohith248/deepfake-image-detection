"""TensorFlow inference, same CLI/output as the PyTorch inference_script.py.
  python inference_tf.py --test_dir DIR --weights sm_ensemble --output predictions.csv
Preprocessing is identical (decode_rgb from the copied deepfake_core_x1.py, /255, ImageNet mean/std);
the SavedModel is NHWC, so the only difference is HWC (not CHW) layout. Needs only tensorflow, numpy, Pillow, torch
(torch is imported by deepfake_core_x1 but never used for compute here)."""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import tensorflow as tf
from deepfake_core_x1 import decode_rgb, enumerate_images
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)

def load(weights):
    return tf.saved_model.load(str(weights)).signatures['serving_default']

def score(fn, paths, batch=16):
    ids, probs, logits = [], [], []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = (np.stack([decode_rgb(p) for p in chunk]).astype(np.float32) / 255 - MEAN) / STD  # NHWC
        out = fn(input=tf.constant(x, tf.float32))
        probs.extend(out['prob'].numpy().tolist()); logits.append(out['logits'].numpy())
        ids.extend(p.stem for p in chunk)
    return ids, probs, np.concatenate(logits)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir', required=True); p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('predictions.csv')); p.add_argument('--batch_size', type=int, default=16)
    a = p.parse_args()
    ids, probs, _ = score(load(a.weights), enumerate_images(a.test_dir), a.batch_size)
    assert all(np.isfinite(probs)) and len(set(ids)) == len(ids)
    with a.output.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['image_id', 'label', 'confidence'])
        for i, q in zip(ids, probs): w.writerow([i, int(q >= .5), format(q, '.17g')])
    print(f'wrote {len(ids)} rows to {a.output}')

if __name__ == '__main__':
    main()
