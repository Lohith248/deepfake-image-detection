"""X1 one-time weight setup on a GPU host (user-approved downloads of named torchvision ImageNet-1K weights only).
Downloads with torchvision's hash-prefix check into ~/x1/weights, computes the full SHA-256, checks the two previously pinned
Tiny hashes exactly, and records every hash for cross-host comparison. Training later refuses any network download."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from deepfake_core_x1 import ARCHITECTURES, sha256

FILES = {'convnext_tiny': 'convnext_tiny-983f1562.pth', 'swin_t': 'swin_t-704ceda3.pth', 'convnext_small': 'convnext_small-0c510722.pth',
         'convnext_base': 'convnext_base-6075fbad.pth', 'swin_s': 'swin_s-5e29d889.pth'}
PINNED = {'convnext_tiny': '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d',
          'swin_t': '704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archs', nargs='+', choices=sorted(FILES), required=True)
    args = parser.parse_args()
    from torchvision import models
    cache = Path.home() / 'x1/weights'
    cache.mkdir(parents=True, exist_ok=True)
    record_path = cache / 'x1-weights-record.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for architecture in args.archs:
        enum_name, enum_value = ARCHITECTURES[architecture]
        weight = getattr(getattr(models, enum_name), enum_value)
        filename = weight.url.rsplit('/', 1)[1]
        if filename != FILES[architecture]:
            raise ValueError('Unexpected torchvision weight file: ' + filename)
        state = torch.hub.load_state_dict_from_url(weight.url, model_dir=str(cache), check_hash=True, map_location='cpu')
        getattr(models, architecture)(weights=None).load_state_dict(state, strict=True)
        path = cache / filename
        digest = sha256(path)
        if not digest.startswith(filename.rsplit('-', 1)[1][:8]):
            raise ValueError('Hash prefix mismatch: ' + architecture)
        if architecture in PINNED and digest != PINNED[architecture]:
            raise ValueError('Pinned hash mismatch: ' + architecture)
        if architecture in record and record[architecture]['sha256'] != digest:
            raise ValueError('Weight file changed since recorded: ' + architecture)
        record[architecture] = {'file': filename, 'url': weight.url, 'enum': enum_name + '.' + enum_value, 'sha256': digest,
                                'bytes': path.stat().st_size, 'pinned_before_download': architecture in PINNED}
        print('X1_WEIGHTS ' + json.dumps({'architecture': architecture, **record[architecture]}), flush=True)
    temp = record_path.with_name(record_path.name + '.part')
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    os.replace(temp, record_path)


if __name__ == '__main__':
    main()
