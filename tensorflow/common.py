"""Shared helpers: the 64 pool images (train/ only) and the preprocessing."""
import csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'research/q2-step2-2026-11-10/pkg-60/pool_manifest.csv'
BUNDLE = ROOT / 'runs/final-primary-v3/delivery/model_weights.pt'
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)

def pool_images(n=64):
    with MANIFEST.open(newline='') as f:
        rows = list(csv.DictReader(f))[:n]
    out = []
    for r in rows:
        sub = 'real' if r['label'] == '0' else 'fake'
        p = ROOT / 'hackathon_dataset/train' / sub / (r['image_id'] + '.jpg')
        assert 'test' not in p.relative_to(ROOT / 'hackathon_dataset').parts[:1]
        out.append(p)
    return out

def normalize_nchw(arrs):
    """uint8 HWC list -> float32 NCHW, identical math to deepfake_core_x1.to_tensor."""
    x = np.stack(arrs).astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(0, 3, 1, 2).copy()
