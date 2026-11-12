"""X1 versioned copy of deepfake_core.py (pre-existing file left unchanged). Adds ConvNeXt-S/B and Swin-S; everything else is byte-for-byte the same logic."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ARCHITECTURES = {
    'efficientnet_b0': ('EfficientNet_B0_Weights', 'IMAGENET1K_V1'),
    'efficientnet_v2_s': ('EfficientNet_V2_S_Weights', 'IMAGENET1K_V1'),
    'convnext_tiny': ('ConvNeXt_Tiny_Weights', 'IMAGENET1K_V1'),
    'swin_t': ('Swin_T_Weights', 'IMAGENET1K_V1'),
    'convnext_small': ('ConvNeXt_Small_Weights', 'IMAGENET1K_V1'),
    'convnext_base': ('ConvNeXt_Base_Weights', 'IMAGENET1K_V1'),
    'swin_s': ('Swin_S_Weights', 'IMAGENET1K_V1'),
}
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def decode_rgb(path):
    """One deterministic path for training fixtures and valid input variants."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                image = image.convert('RGBA')
                background = Image.new('RGBA', image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            image = image.convert('RGB')
            if image.size != (224, 224): image = image.resize((224, 224), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8).copy()
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f'Cannot decode image {Path(path).name}: {exc}') from exc


def to_tensor(array):
    x = torch.from_numpy(np.asarray(array).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def enumerate_images(directory):
    directory = Path(directory)
    if not directory.is_dir(): raise ValueError(f'Image directory missing: {directory}')
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name)
    if not paths: raise ValueError('No supported images')
    ids = [p.stem for p in paths]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate image stems across extensions')
    return paths


class ImageFolder(Dataset):
    def __init__(self, directory): self.paths = enumerate_images(directory)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        return to_tensor(decode_rgb(p)), p.stem


def build_model(architecture, pretrained=False, weights_dir=None):
    if architecture not in ARCHITECTURES: raise ValueError(f'Unsupported architecture: {architecture}')
    from torchvision import models
    enum_name, enum_value = ARCHITECTURES[architecture]
    weight = getattr(getattr(models, enum_name), enum_value)
    model = getattr(models, architecture)(weights=None)
    provenance = None
    if pretrained:
        if weights_dir is None: raise ValueError('An explicit project weight cache is required')
        cache = Path(weights_dir); cache.mkdir(parents=True, exist_ok=True)
        state = torch.hub.load_state_dict_from_url(weight.url, model_dir=str(cache), check_hash=True, map_location='cpu')
        model.load_state_dict(state)
        path = cache / weight.url.rsplit('/', 1)[1]
        provenance = {'architecture': architecture, 'enum': enum_name + '.' + enum_value,
                      'url': weight.url, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                      'pretraining': 'torchvision documented ImageNet-1K recipe',
                      'code_license': 'torchvision BSD-3-Clause; weight/data terms separate',
                      'documentation': f'https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.{architecture}.html'}
    if architecture.startswith('efficientnet'): model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif architecture.startswith('convnext_'): model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    else: model.head = nn.Linear(model.head.in_features, 1)
    return model, provenance


def save_bundle(path, architecture, model, provenance=None, purpose='E0 smoke only; not a selected detector', training_metadata=None):
    metadata = {'schema_version': 1, 'input_size': 224, 'normalization': 'imagenet',
                'interpolation': 'PIL-bilinear', 'combination': 'probability_mean', 'threshold': .5,
                'temperature': 1., 'provenance': provenance, 'purpose': purpose, 'training': training_metadata}
    torch.save({'metadata_json': json.dumps(metadata), 'members': [{'architecture': architecture,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}]}, path)


def load_bundle(path, device):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if set(bundle) != {'metadata_json', 'members'}: raise ValueError('Unexpected bundle fields')
    meta = json.loads(bundle['metadata_json'])
    if meta.get('schema_version') != 1 or meta.get('input_size') != 224 or meta.get('normalization') != 'imagenet':
        raise ValueError('Unsupported bundle input contract')
    if meta.get('combination') not in ('probability_mean', 'logit_mean'): raise ValueError('Unsupported combination')
    if not 0 <= meta['threshold'] <= 1 or not math.isfinite(meta['temperature']) or meta['temperature'] <= 0:
        raise ValueError('Invalid scoring metadata')
    if not 1 <= len(bundle['members']) <= 4: raise ValueError('Invalid ensemble member count')
    models = []
    for member in bundle['members']:
        model, _ = build_model(member['architecture'])
        model.load_state_dict(member['state_dict'], strict=True)
        models.append(model.to(device).float().eval())
    return models, meta


@torch.inference_mode()
def predict(models, loader, device, meta, precision='fp32'):
    ids, probabilities = [], []
    for x, names in loader:
        x = x.to(device)
        member_logits = []
        for model in models:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(precision == 'fp16' and device.type == 'cuda')):
                logits = model(x).reshape(-1)
            member_logits.append(logits.double() / meta['temperature'])
        scores = torch.stack(member_logits)
        p = torch.sigmoid(scores.mean(0)) if meta['combination'] == 'logit_mean' else torch.sigmoid(scores).mean(0)
        if not torch.isfinite(p).all(): raise ValueError('Non-finite model scores')
        ids.extend(names); probabilities.extend(p.cpu().tolist())
    if len(set(ids)) != len(ids): raise ValueError('Duplicate output IDs')
    return ids, probabilities


def write_predictions(path, ids, probabilities, threshold=.5):
    if len(ids) != len(probabilities) or len(set(ids)) != len(ids): raise ValueError('Invalid output ID mapping')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities): raise ValueError('Invalid probability')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temp.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['image_id', 'label', 'confidence'])
            for i, p in zip(ids, probabilities): w.writerow([i, int(p >= threshold), format(p, '.17g')])
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()
