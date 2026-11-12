"""Fixed, label-independent training/validation transformations for E1."""
import hashlib
import io
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SEVERITY_WEIGHTS = [.5, .35, .15]
FAMILIES = {
    'jpeg': ['jpeg'],
    'resolution': ['downsample', 'pixelate'],
    'photometric': ['gamma', 'saturation', 'tint'],
    'tone': ['posterize', 'solarize', 'inversion'],
    'occlusion': ['neutral_patch', 'color_patch'],
    'noise': ['gaussian_noise', 'impulse_noise'],
    'convolution': ['gaussian_blur', 'motion_blur'],
    'geometric': ['rotation', 'resize_jpeg'],
}
TRAIN_FAMILIES = list(FAMILIES)[:6]


def stable_seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def conditions():
    result = [{'name': 'original', 'family': 'original', 'operator': 'original', 'severity': 0, 'weight': 0.}]
    for family, operators in FAMILIES.items():
        for operator in operators:
            for severity in range(3):
                result.append({'name': f'{operator}_s{severity+1}', 'family': family, 'operator': operator,
                               'severity': severity, 'weight': SEVERITY_WEIGHTS[severity] / len(operators) / len(FAMILIES)})
    for operator in ['centre_only', 'border_only']:
        result.append({'name': operator, 'family': 'diagnostic', 'operator': operator, 'severity': 0, 'weight': 0.})
    return result


def jpeg(im, quality):
    buf = io.BytesIO(); im.save(buf, format='JPEG', quality=quality, subsampling=2); buf.seek(0)
    with Image.open(buf) as source: return source.convert('RGB').copy()


def transform(array, operator, severity, seed):
    rng = np.random.default_rng(seed)
    im = Image.fromarray(np.asarray(array, dtype=np.uint8), 'RGB')
    s = severity
    if operator == 'original': return array.copy()
    if operator == 'jpeg': im = jpeg(im, [85, 65, 40][s])
    elif operator == 'downsample':
        side = [160, 112, 64][s]; im = im.resize((side, side), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
    elif operator == 'pixelate':
        side = 224 // [2, 4, 8][s]; im = im.resize((side, side), Image.Resampling.BOX).resize((224, 224), Image.Resampling.NEAREST)
    elif operator == 'gamma':
        gamma = [.9, .75, .6][s]
        if rng.integers(2): gamma = 1 / gamma
        im = Image.fromarray(np.uint8(np.clip((np.asarray(im) / 255.) ** gamma * 255, 0, 255)))
    elif operator == 'saturation': im = ImageEnhance.Color(im).enhance([.7, .3, 0.][s])
    elif operator == 'tint':
        color = [(220, 50, 160), (50, 210, 70), (40, 90, 230)][rng.integers(3)]
        im = Image.blend(im, Image.new('RGB', im.size, color), [.1, .2, .35][s])
    elif operator == 'posterize': im = ImageOps.posterize(im, [7, 6, 4][s])
    elif operator == 'solarize': im = ImageOps.solarize(im, [240, 192, 128][s])
    elif operator == 'inversion': im = Image.blend(im, ImageOps.invert(im), [.1, .25, 1.][s])
    elif operator in ('neutral_patch', 'color_patch'):
        area = 224 * 224 * [.01, .04, .08][s]; aspect = float(rng.uniform(.5, 2))
        width = min(224, max(1, round(math.sqrt(area * aspect)))); height = min(224, max(1, round(area / width)))
        x = int(rng.integers(225-width)); y = int(rng.integers(225-height))
        color = (int(rng.choice([0,128,255])),) * 3 if operator == 'neutral_patch' else tuple(int(x) for x in rng.integers(0,256,3))
        im.paste(color, (x,y,x+width,y+height))
    elif operator == 'gaussian_noise':
        a = np.asarray(im,dtype=np.float32) + rng.normal(0,[.01,.02,.04][s]*255,(224,224,3))
        im = Image.fromarray(np.uint8(a.clip(0,255)))
    elif operator == 'impulse_noise':
        a = np.asarray(im).copy(); mask = rng.random((224,224)); p = [.002,.01,.03][s]
        a[mask < p/2] = 0; a[mask > 1-p/2] = 255; im = Image.fromarray(a)
    elif operator == 'gaussian_blur': im = im.filter(ImageFilter.GaussianBlur([.4,.8,1.5][s]))
    elif operator == 'motion_blur':
        k = [3,5,7][s]; a = np.asarray(im,dtype=np.float32); padded = np.pad(a,((0,0),(k//2,k//2),(0,0)),mode='edge')
        a = sum(padded[:,offset:offset+224] for offset in range(k)) / k; im = Image.fromarray(np.uint8(a.clip(0,255)))
    elif operator == 'rotation':
        angle = [2,5,10][s] * (-1 if rng.integers(2) else 1)
        im = im.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128))
    elif operator == 'resize_jpeg':
        side = [160,112,64][s]; im = im.resize((side,side),Image.Resampling.BILINEAR).resize((224,224),Image.Resampling.BILINEAR)
        im = jpeg(im,[85,65,40][s])
    elif operator in ('centre_only', 'border_only'):
        a = np.asarray(im).copy(); fill = np.uint8([124,116,104])
        if operator == 'border_only': a[32:192,32:192] = fill
        else:
            a[:32] = fill; a[192:] = fill; a[:, :32] = fill; a[:,192:] = fill
        im = Image.fromarray(a)
    else: raise ValueError('Unknown operator: '+operator)
    return np.asarray(im,dtype=np.uint8).copy()


def training_view(array, image_id, epoch, seed, fold):
    rng = random.Random(stable_seed(f'train:{seed}:{fold}:{epoch}:{image_id}'))
    result = array[:, ::-1].copy() if rng.random() < .5 else array.copy()
    if rng.random() < .35:
        family = rng.choice(TRAIN_FAMILIES); operator = rng.choice(FAMILIES[family]); severity = 0 if rng.random() < .6 else 1
        result = transform(result,operator,severity,rng.randrange(2**63))
    return result
