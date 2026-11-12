"""Final-v3 inference: CT_SS (ConvNeXt-T + Swin-S). Interface identical to final-v2; only the member set and versioned core differ.
CPU fallback requires a measured, hash-bound policy."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from deepfake_core_x1 import ImageFolder, load_bundle, predict, sha256, write_predictions


def choose_device(requested, cuda_available, policy, bundle_hash):
    if requested == 'cpu':
        return 'cpu'
    if cuda_available:
        return 'cuda'
    if policy.get('cpu_fallback_enabled') is True and policy.get('bundle_sha256') == bundle_hash:
        return 'cpu'
    raise ValueError('CUDA unavailable and this bundle has no verified CPU fallback')


def validate_metadata(models, meta):
    if len(models) != 2 or meta.get('candidate') != 'CT_SS' or meta.get('combination') != 'logit_mean':
        raise ValueError('Expected the frozen two-member CT_SS bundle')
    if [item.get('architecture') for item in meta.get('provenance') or []] != ['convnext_tiny', 'swin_s']:
        raise ValueError('Expected ConvNeXt-T and Swin-S members in that order')
    if meta.get('threshold') != .5 or meta.get('temperature') != 1. or meta.get('precision') != 'fp32':
        raise ValueError('Frozen inference settings changed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir', required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('predictions.csv'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--precision', choices=['fp32'], default='fp32')
    p.add_argument('--threshold', type=float, default=.5)
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        p.error('Batch size must be positive and workers nonnegative')
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        p.error('Threshold must be finite and in [0,1]')
    if args.batch_size != 64 or args.workers != 0:
        print('Compatibility flags accepted; frozen inference uses batch size 64 and workers 0', file=sys.stderr)
    if args.threshold != .5:
        print('Compatibility threshold accepted; frozen labels use threshold 0.5', file=sys.stderr)
    if args.output.exists():
        print('Existing output will be replaced atomically only after successful inference; on failure previous CSV remains unchanged', file=sys.stderr)
    started = time.perf_counter()
    policy_path = args.weights.parent / 'release_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    device_name = choose_device(args.device, torch.cuda.is_available(), policy, sha256(args.weights))
    if device_name == 'cpu' and args.device == 'cuda':
        print('CUDA unavailable; using verified CPU fallback for this exact bundle', file=sys.stderr)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    dataset = ImageFolder(args.test_dir)
    ignored = sum(1 for item in Path(args.test_dir).iterdir() if item not in dataset.paths)
    if ignored:
        print(f'Ignored {ignored} sidecars/subdirectories; flat-directory inference only', file=sys.stderr)
    models, meta = load_bundle(args.weights, device)
    validate_metadata(models, meta)
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    ids, probabilities = predict(models, loader, device, meta, 'fp32')
    if ids != [item.stem for item in dataset.paths]:
        raise ValueError('Incomplete or reordered output IDs')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(json.dumps({'rows': len(ids), 'device': str(device), 'precision': 'fp32',
                      'inside_process_seconds': time.perf_counter() - started,
                      'status': 'inference complete; committing output'}))
    write_predictions(args.output, ids, probabilities, .5)


if __name__ == '__main__':
    main()
