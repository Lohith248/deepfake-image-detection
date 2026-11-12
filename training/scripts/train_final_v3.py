"""Final-v3 (CT_SS): frozen full-pool fit, one sealed readout, and all-data refit. No adaptive choices.
Derived from train_final_v2.py by scripts/prepare_final_v3.py; only members, core and weight cache differ."""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import csv
import gc
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_p1 as T
import deepfake_core_x1 as C
from analyze_p1 import bootstrap_metrics

PROTOCOL = 'experiments/final_protocol_v3.json'
ARCHITECTURES = ('convnext_tiny', 'swin_s')
WEIGHTS = {'convnext_tiny': ('convnext_tiny-983f1562.pth', '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'), 'swin_s': ('swin_s-5e29d889.pth', '5e29d889356a51648611f542380eb4f3c4b4c84dc61933cdfbd2afc14a8235b2')}


def weight_cache():
    return Path.home() / 'x1/weights'


def verify_weights(cache):
    for name, (filename, digest) in WEIGHTS.items():
        path = cache / filename
        if not path.is_file() or T.sha256(path) != digest:
            raise ValueError('Missing or changed ImageNet weights: ' + name)


def immutable_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Immutable record changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if sys.platform != 'win32':
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def source_identity(root):
    hashes = json.loads((root / 'source-hashes.json').read_text())
    for name, digest in hashes.items():
        path = root / name
        if name.startswith(('holdout-images/', 'pool-images/')):
            raise ValueError('Image bytes must not be in the source-hash traversal')
        if not path.resolve().is_relative_to(root) or T.sha256(path) != digest:
            raise ValueError('Source changed: ' + name)
    if PROTOCOL not in hashes or 'scripts/train_final_v3.py' not in hashes:
        raise ValueError('Final sources absent from identity')
    return {'source_hashes': hashes, 'source_hashes_sha256': T.sha256(root / 'source-hashes.json'),
            'protocol_sha256': T.sha256(root / PROTOCOL),
            'full_manifest_sha256': T.sha256(root / 'full_manifest.csv')}


def check_protocol(protocol):
    expected = {'epochs': 20, 'train_batch_size': 32, 'eval_batch_size': 64,
                'backbone_lr': .0001, 'head_lr': .001, 'weight_decay': .01,
                'warmup_epochs': 2, 'ema_decay': .999, 'input_size': 224}
    if any(protocol['recipe'].get(key) != value for key, value in expected.items()):
        raise ValueError('Frozen recipe changed')
    if protocol['members'] != [{'architecture': a, 'seed': 17} for a in ARCHITECTURES]:
        raise ValueError('Frozen members changed')
    if protocol['holdout']['bootstrap_draws'] != 2000 or protocol['holdout']['bootstrap_seed'] != 20261102:
        raise ValueError('Frozen bootstrap changed')
    if protocol['x1_selection']['chosen'] != 'CT_SS' or protocol['x1_selection']['timing_gate_pass'] is not True:
        raise ValueError('X1 selection evidence changed')
    if (protocol['selected'], protocol['loss'], protocol['gce_q'], protocol['rng_namespace_fold'],
            protocol['member_seed_count']) != ('CT_SS', 'gce_q07', .7, 0, 1):
        raise ValueError('Frozen procedure changed')
    inference = {'combination': 'logit_mean', 'member_weights': [.5, .5], 'precision': 'fp32',
                 'sigmoid': 'float64', 'TF32': False, 'threshold': .5, 'temperature': 1.,
                 'batch_size': 64, 'workers': 0}
    if protocol['inference'] != inference:
        raise ValueError('Frozen inference changed')
    release = {'fixture_images': 3000, 'cold_gpu_processes': 3, 'gpu_timeout_seconds': 600,
               'cpu_processes': 1, 'cpu_timeout_seconds': 1800, 'cpu_fallback_max_seconds': 1200,
               'cpu_max_probability_difference': 1e-5, 'cpu_max_label_flips': 0}
    if any(protocol['release'].get(key) != value for key, value in release.items()):
        raise ValueError('Frozen release gates changed')
    if protocol['fit_stages'] != [{'stage': 'full_pool', 'images': 5950}, {'stage': 'all_data', 'images': 7000}]:
        raise ValueError('Frozen fit schedule changed')


def read_metadata(root, pool):
    with (root / 'full_manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 7000 or len({r['image_id'] for r in rows}) != 7000:
        raise ValueError('Full manifest must have 7000 unique rows')
    pool_map = {r['image_id']: r for r in pool}
    full_pool = [r for r in rows if r['partition'] == 'pool']
    holdout = [r for r in rows if r['partition'] != 'pool']
    if len(full_pool) != 5950 or len(holdout) != 1050:
        raise ValueError('Frozen partition counts changed')
    for row in rows:
        if row['label'] not in ('0', '1') or not row['group_id'] or len(row['sha256']) != 64:
            raise ValueError('Invalid full manifest metadata')
        expected_prefix = 'pool-images/' if row['partition'] == 'pool' else 'holdout-images/'
        if row['relative_path'] != expected_prefix + row['image_id'] + '.jpg':
            raise ValueError('Unsafe or noncanonical image path')
        if row['partition'] == 'pool':
            if any(row[key] != pool_map[row['image_id']][key] for key in pool_map[row['image_id']]):
                raise ValueError('Pool metadata changed in full manifest')
    if {r['group_id'] for r in full_pool} & {r['group_id'] for r in holdout}:
        raise ValueError('Source group crosses sealed partition')
    return rows, holdout


def dry_run(root):
    root = root.resolve()
    protocol = json.loads((root / PROTOCOL).read_text())
    check_protocol(protocol)
    identity = source_identity(root)
    if (identity['full_manifest_sha256'] != protocol['manifest_hashes']['full']
            or T.sha256(root / 'pool_manifest.csv') != protocol['manifest_hashes']['pool']
            or protocol['manifest_hashes']['frozen_split'] != protocol['recipe']['split_manifest_sha256']):
        raise ValueError('Manifests differ from written freeze')
    pool = T.verify_pool(root)
    full, holdout = read_metadata(root, pool)
    result = {'status': 'passed', 'pool_images_verified': len(pool), 'full_manifest_rows': len(full),
              'sealed_metadata_rows': len(holdout), 'holdout_bytes_opened': 0,
              'gpu_probed': False, 'fits': 4, 'identity': identity}
    print('FINAL_DRY_RUN ' + json.dumps(result), flush=True)
    return protocol, identity, pool, full, holdout


def validate_stage(directory, identity):
    receipt = json.loads((directory / 'receipt.json').read_text())
    if receipt['identity'] != identity or receipt.get('stage_name') != directory.name:
        raise ValueError('Stage identity changed')
    if {p.name for p in directory.iterdir()} != set(receipt['files']) | {'receipt.json'}:
        raise ValueError('Stage file set changed')
    for name, digest in receipt['files'].items():
        if T.sha256(directory / name) != digest:
            raise ValueError('Completed stage artifact changed: ' + name)
    return receipt


def commit_stage(staging, final, identity):
    if final.exists():
        raise ValueError('Completed stage already exists')
    immutable_json(staging / 'receipt.json', {'identity': identity, 'stage_name': final.name,
                   'files': {p.name: T.sha256(p) for p in staging.iterdir()}})
    for path in staging.iterdir():
        with path.open('r+b') as stream:
            os.fsync(stream.fileno())
    staging.rename(final)
    if sys.platform != 'win32':
        fd = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    validate_stage(final, identity)


def publish(directory):
    for path in sorted(directory.iterdir()):
        if path.suffix in ('.json', '.npz'):
            T.emit_artifact('final_' + directory.name + '_' + path.name, path.read_bytes())


def train_member(root, out, stage, architecture, rows, arrays, config, identity):
    name = stage + '_' + architecture
    final = out / name
    if final.exists():
        validate_stage(final, identity)
        publish(final)
        return final / 'checkpoint.pt'
    if stage == 'full_pool' and (out / 'holdout_started.json').exists():
        raise RuntimeError('Full-pool refit is forbidden after H was opened')
    T.readiness()
    cache = weight_cache()
    verify_weights(cache)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seed = 17
    fold_namespace = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    staging = out / ('.attempt-' + name + '-' + uuid.uuid4().hex)
    staging.mkdir()
    start = time.perf_counter()
    def reject_download(*args, **kwargs):
        raise RuntimeError('Checkpoint downloads are forbidden')
    original_download = torch.hub.download_url_to_file
    torch.hub.download_url_to_file = reject_download
    try:
        model, provenance = C.build_model(architecture, pretrained=True, weights_dir=cache)
    finally:
        torch.hub.download_url_to_file = original_download
    if provenance['sha256'] != WEIGHTS[architecture][1]:
        raise ValueError('Loaded initialization changed')
    model = model.cuda()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    # train_p1.optimizer_for puts classifier.2.* (ConvNeXt-T) or head.* (Swin-S) in the head group, as in X1.
    optimizer = T.optimizer_for(model, architecture, config)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    dataset = T.FitSet(arrays, rows, np.arange(len(rows)), fold_namespace, seed)
    curves = []
    global_step = 0
    print('FINAL_FIT_START ' + name, flush=True)
    for epoch in range(config['epochs']):
        T.readiness(check_free=False)
        tick = time.perf_counter()
        dataset.epoch = epoch
        model.train()
        generator = torch.Generator().manual_seed(seed + fold_namespace * 1000 + epoch)
        loader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True,
                            num_workers=0, pin_memory=True, generator=generator)
        epoch_loss = 0.
        seen = skipped = 0
        for batch, (x, y) in enumerate(loader):
            progress = epoch + (batch + 1) / len(loader)
            factor = .1 + .9 * progress / config['warmup_epochs'] if progress <= config['warmup_epochs'] else .5 * (1 + math.cos(math.pi * (progress - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            x = x.cuda(non_blocking=True)
            y = y.float().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(x).reshape(-1)
                loss = T.training_loss(z, y, 'gce_q07')
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
            else:
                with torch.no_grad():
                    for q, p0 in zip(ema.parameters(), model.parameters()):
                        q.mul_(config['ema_decay']).add_(p0, alpha=1-config['ema_decay'])
                    for q, b0 in zip(ema.buffers(), model.buffers()):
                        q.copy_(b0)
            epoch_loss += float(loss.detach()) * len(y)
            seen += len(y)
            global_step += 1
        torch.cuda.synchronize()
        if skipped > max(2, .05 * len(loader)):
            raise ValueError('Too many non-finite AMP steps')
        row = {'epoch': epoch + 1, 'loss': epoch_loss / seen,
               'train_seconds': time.perf_counter() - tick, 'optimizer_steps': global_step,
               'amp_skipped_steps': skipped, 'last_lr': optimizer.param_groups[0]['lr']}
        curves.append(row)
        print('FINAL_EPOCH ' + json.dumps({'fit': name, **row}), flush=True)
    if any(not torch.isfinite(value).all() for value in ema.state_dict().values()):
        raise ValueError('Non-finite final state')
    checkpoint = staging / 'checkpoint.pt'
    C.save_bundle(checkpoint, architecture, ema, provenance, purpose='Frozen final-v3 ' + stage,
                  training_metadata={'stage': stage, 'seed': 17, 'loss': 'gce_q07', 'epochs': 20,
                                     'fold_namespace': 0, 'protocol_sha256': identity['protocol_sha256']})
    from deepfake_core_x1 import load_bundle
    probe_indices = [i for i, row in enumerate(rows) if row['partition'] == 'pool'][:16]
    if len(probe_indices) != 16:
        raise ValueError('Pool-only reload fixture is incomplete')
    probe = arrays[probe_indices]
    before = T.logits_for(ema, probe, 64)
    reloaded, _ = load_bundle(checkpoint, torch.device('cuda'))
    after = T.logits_for(reloaded[0], probe, 64)
    reload_difference = float(np.max(np.abs(T.sigmoid64(before) - T.sigmoid64(after))))
    if reload_difference > 1e-5:
        raise ValueError('Checkpoint reload differs on pool-only fixture')
    del reloaded, probe
    immutable_json(staging / 'curves.json', curves)
    immutable_json(staging / 'summary.json', {'stage': stage, 'architecture': architecture,
        'seed': 17, 'loss': 'gce_q07', 'epochs': 20, 'fit_images': len(rows),
        'train_seconds': sum(r['train_seconds'] for r in curves),
        'wall_seconds': time.perf_counter()-start, 'checkpoint_sha256': T.sha256(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size, 'provenance': provenance,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'amp_skipped_steps': sum(r['amp_skipped_steps'] for r in curves),
        'pool_reload_max_probability_difference': reload_difference})
    commit_stage(staging, final, identity)
    publish(final)
    del optimizer, scaler, model, ema, x, y, z, loss
    gc.collect()
    torch.cuda.empty_cache()
    return final / 'checkpoint.pt'


def holdout_readout(root, out, rows, checkpoints, protocol, identity):
    final = out / 'holdout'
    if final.exists():
        validate_stage(final, identity)
        publish(final)
        return
    marker = out / 'holdout_started.json'
    if marker.exists():
        raise RuntimeError('H readout was interrupted. STOP for functional-failure review; never rerun or refit automatically')
    immutable_json(marker, {'identity': identity,
        'checkpoint_hashes': {a: T.sha256(p) for a, p in checkpoints.items()},
        'policy': 'One readout only. Performance never changes the frozen procedure.'})
    staging = out / ('.attempt-holdout-' + uuid.uuid4().hex)
    staging.mkdir()
    for row in rows:
        path = root / row['relative_path']
        if path.is_symlink() or not path.resolve().is_relative_to(root / 'holdout-images') or T.sha256(path) != row['sha256']:
            raise ValueError('Sealed image integrity failed')
    arrays = np.stack([T.decode_rgb(root / r['relative_path']) for r in rows])
    cs = T.conditions()
    cache = T.build_validation_cache(arrays, rows, np.arange(len(rows)), cs, 8)
    all_logits = []
    from deepfake_core_x1 import load_bundle
    for architecture in ARCHITECTURES:
        models, meta = load_bundle(checkpoints[architecture], torch.device('cuda'))
        logits = np.stack([T.logits_for(models[0], condition, 64) for condition in cache])
        all_logits.append(logits)
        del models
        gc.collect()
        torch.cuda.empty_cache()
    del cache, arrays
    logits = np.stack(all_logits)
    y = np.array([int(row['label']) for row in rows])
    groups = np.array([row['group_id'] for row in rows])
    np.savez_compressed(staging / 'predictions.npz', member_logits=logits,
        ensemble_logits=logits.mean(0), probabilities=T.sigmoid64(logits.mean(0)), y=y,
        image_ids=np.array([r['image_id'] for r in rows]), group_ids=groups,
        condition_names=np.array([c['name'] for c in cs]),
        condition_weights=np.array([c['weight'] for c in cs]))
    fit = SimpleNamespace(y=y, group_ids=groups)
    scores = {a: T.sigmoid64(logits[i]) for i, a in enumerate(ARCHITECTURES)}
    scores['CT_SS'] = T.sigmoid64(logits.mean(0))
    boot = {name: bootstrap_metrics(fit, value, cs, 2000, 20261102) for name, value in scores.items()}
    metrics = {name: T.summarize(y, value, cs) for name, value in scores.items()}
    intervals = {name: {key: np.nanpercentile(values[key], [2.5, 97.5]).tolist()
                       for key in ('original_auc', 'robust_auc_proxy', 'J04')}
                 for name, values in boot.items()}
    paired = {name: {key: np.nanpercentile(boot['CT_SS'][key]-boot[name][key], [2.5, 97.5]).tolist()
                    for key in ('original_auc', 'robust_auc_proxy', 'J04')} for name in ARCHITECTURES}
    immutable_json(staging / 'analysis.json', {'metrics': metrics, 'ci95': intervals,
        'paired_CT_SS_minus_member_ci95': paired, 'bootstrap_draws': 2000, 'bootstrap_seed': 20261102,
        'invalid_bootstrap_draws': {name: int((~np.isfinite(value['J04'])).sum()) for name, value in boot.items()},
        'scope': 'One sealed readout of full-pool fits. Conditional source-group uncertainty; no adaptive choice. J is a local proxy.',
        'decision': 'Continue unchanged to all-data refit regardless of these scores.'})
    commit_stage(staging, final, identity)
    publish(final)


def pool_checkpoint_sanity(root, out, pool, checkpoints, identity):
    from deepfake_core_x1 import load_bundle
    record = out / 'full_pool_sanity.json'
    hashes = {a: T.sha256(p) for a, p in checkpoints.items()}
    if record.exists():
        previous = json.loads(record.read_text())
        if previous['identity'] != identity or previous['checkpoint_hashes'] != hashes:
            raise ValueError('Full-pool functional gate identity changed')
        return
    if (out / 'holdout_started.json').exists():
        raise RuntimeError('Missing pre-H functional receipt after H opening')
    chosen = sorted(pool, key=lambda row: row['image_id'])[:16]
    arrays = np.stack([T.decode_rgb(T.pool_path(root, row['relative_path'])) for row in chosen])
    logits = []
    for architecture in ARCHITECTURES:
        models, meta = load_bundle(checkpoints[architecture], torch.device('cuda'))
        if len(models) != 1 or meta['training']['stage'] != 'full_pool':
            raise ValueError('Wrong full-pool checkpoint')
        logits.append(T.logits_for(models[0], arrays, 64))
        del models
        gc.collect()
        torch.cuda.empty_cache()
    probabilities = T.sigmoid64(np.mean(logits, axis=0))
    if len(set(probabilities.tolist())) < 2:
        raise ValueError('Full-pool functional sanity failed: constant fixture scores')
    immutable_json(record, {'identity': identity, 'checkpoint_hashes': hashes,
        'image_ids': [r['image_id'] for r in chosen], 'probabilities': probabilities.tolist(),
        'scope': 'Pool-only functional gate: checkpoint loading, finite and nonconstant scores. No performance claim.'})
    T.emit_artifact('final_full_pool_sanity.json', record.read_bytes())


def remote_environment():
    import torchvision
    import PIL
    expected = (sys.version_info[:3] == (3, 10, 22) and torch.__version__ == '2.7.1+cu128'
                and torchvision.__version__ == '0.22.1+cu128' and np.__version__ == '2.2.6'
                and PIL.__version__ == '11.2.1')
    if not expected:
        raise RuntimeError('Pinned remote environment required')
    T.readiness()
    if not torch.cuda.is_available() or '5060 Ti' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Verified primary GPU required')
    return {'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__, 'Pillow': PIL.__version__, 'cuda': torch.version.cuda,
            'gpu_uuid': 'GPU-4f7dd3a1-55b5-1d57-d0ad-00186da5363e'}


def execute(root):
    env = remote_environment()
    protocol, identity, pool, full, holdout = dry_run(root)
    identity['environment'] = env
    identity = json.loads(json.dumps(identity))
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = root / 'final-results'
    out.mkdir(exist_ok=True)
    immutable_json(out / 'environment.json', identity)
    T.emit_artifact('final_environment.json', (out / 'environment.json').read_bytes())
    arrays = None
    checkpoints = {}
    for architecture in ARCHITECTURES:
        if not (out / ('full_pool_' + architecture)).exists() and arrays is None:
            if (out / 'holdout_started.json').exists():
                raise RuntimeError('A full-pool fit is missing after H opening; stop')
            arrays = np.stack([T.decode_rgb(T.pool_path(root, r['relative_path'])) for r in pool])
        checkpoints[architecture] = train_member(root, out, 'full_pool', architecture, pool, arrays, protocol['recipe'], identity)
    del arrays
    gc.collect()
    pool_checkpoint_sanity(root, out, pool, checkpoints, identity)
    holdout_readout(root, out, holdout, checkpoints, protocol, identity)
    arrays = None
    all_checkpoints = {}
    for architecture in ARCHITECTURES:
        if not (out / ('all_data_' + architecture)).exists() and arrays is None:
            for row in full:
                if T.sha256(root / row['relative_path']) != row['sha256']:
                    raise ValueError('All-data source integrity failed')
            arrays = np.stack([T.decode_rgb(root / r['relative_path']) for r in full])
        all_checkpoints[architecture] = train_member(root, out, 'all_data', architecture, full, arrays, protocol['recipe'], identity)
    del arrays
    gc.collect()
    from final_release_v3 import release
    release(root, all_checkpoints, pool, protocol, identity)
    immutable_json(out / 'complete.json', {'identity': identity, 'status': 'complete',
        'stage_receipts': {name: T.sha256(out/name/'receipt.json') for name in
            ['full_pool_convnext_tiny', 'full_pool_swin_s', 'holdout', 'all_data_convnext_tiny', 'all_data_swin_s', 'release']}})
    T.emit_artifact('final_complete.json', (out / 'complete.json').read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute-frozen', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.dry_run:
        dry_run(root)
        return
    import fcntl
    with (root / '.final-v3.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(root)
        except BaseException as exc:
            failure = root / 'final-results' / ('functional-failure-' + uuid.uuid4().hex + '.json')
            immutable_json(failure, {'status': 'stopped', 'error': repr(exc),
                'holdout_started': (root / 'final-results/holdout_started.json').exists(),
                'policy': 'No score-driven changes. Investigate functional fault; preserve all prior artifacts.'})
            T.emit_artifact('final_' + failure.name, failure.read_bytes())
            raise


if __name__ == '__main__':
    main()
