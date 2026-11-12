"""Release gates on final weights and a fixed pool-only fixture."""
import sys
sys.dont_write_bytecode = True
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import train_p1 as T
from release_p1 import read_predictions
from final_cli_runner_v3 import run_cli


def fallback_eligible(cpu, probabilities, reference):
    if cpu['timed_out'] or cpu['returncode'] != 0 or probabilities is None:
        return False, None, None
    difference = max(abs(a-b) for a, b in zip(probabilities, reference))
    flips = sum((a >= .5) != (b >= .5) for a, b in zip(probabilities, reference))
    return cpu['wall_seconds'] <= 1200 and difference <= 1e-5 and flips == 0, difference, flips


def make_bundle(checkpoints, output, identity):
    members = []
    sources = []
    for architecture in ('convnext_tiny', 'swin_s'):
        path = checkpoints[architecture]
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        if set(bundle) != {'metadata_json', 'members'} or len(bundle['members']) != 1:
            raise ValueError('Invalid member bundle')
        member = bundle['members'][0]
        if member['architecture'] != architecture or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in member['state_dict'].values()):
            raise ValueError('Invalid member state')
        training = json.loads(bundle['metadata_json'])['training']
        if training != {'stage': 'all_data', 'seed': 17, 'loss': 'gce_q07', 'epochs': 20,
                        'fold_namespace': 0, 'protocol_sha256': identity['protocol_sha256']}:
            raise ValueError('Wrong member training provenance')
        members.append(member)
        sources.append({'architecture': architecture, 'seed': 17, 'loss': 'gce_q07',
                        'path': str(path), 'sha256': T.sha256(path)})
    metadata = {'schema_version': 1, 'input_size': 224, 'normalization': 'imagenet',
        'interpolation': 'PIL-bilinear', 'combination': 'logit_mean', 'threshold': .5,
        'temperature': 1., 'precision': 'fp32', 'candidate': 'CT_SS',
        'purpose': 'Final-v3 CT_SS all-7000 training refit. Holdout scored only by earlier full-pool checkpoints.',
        'provenance': sources, 'protocol_sha256': identity['protocol_sha256']}
    torch.save({'metadata_json': json.dumps(metadata, sort_keys=True), 'members': members}, output)
    return metadata


def export_delivery(root, release_dir):
    from train_final_v3 import immutable_json
    names = ('model_weights.pt', 'inference_script.py', 'deepfake_core_x1.py', 'requirements.txt',
             'inferencescript.py', 'release_policy.json', 'provenance.json', 'release_results.json',
             'release_bundle.json')
    manifest = {name: {'sha256': T.sha256(release_dir / name), 'bytes': (release_dir / name).stat().st_size}
                for name in names}
    delivery = root / 'delivery'
    if delivery.exists():
        if json.loads((delivery / 'delivery-manifest.json').read_text()) != manifest:
            raise ValueError('Existing delivery manifest changed')
        if {path.name for path in delivery.iterdir()} != set(names) | {'delivery-manifest.json'}:
            raise ValueError('Unexpected delivery files')
        for name, item in manifest.items():
            if T.sha256(delivery / name) != item['sha256']:
                raise ValueError('Existing delivery file changed')
        return
    staging = root / ('.delivery-' + uuid.uuid4().hex)
    staging.mkdir()
    for name in names:
        shutil.copyfile(release_dir / name, staging / name)
    immutable_json(staging / 'delivery-manifest.json', manifest)
    staging.rename(delivery)


def release(root, checkpoints, pool, protocol, identity):
    from train_final_v3 import immutable_json, validate_stage, commit_stage, publish
    out = root / 'final-results'
    final = out / 'release'
    if final.exists():
        validate_stage(final, identity)
        publish(final)
        export_delivery(root, final)
        return
    staging = out / ('.attempt-release-' + uuid.uuid4().hex)
    staging.mkdir()
    bundle = staging / 'model_weights.pt'
    metadata = make_bundle(checkpoints, bundle, identity)
    if bundle.stat().st_size > 5_000_000_000:
        raise ValueError('Final bundle exceeds organizer model size limit')
    bundle_hash = T.sha256(bundle)
    chosen = sorted(pool, key=lambda row: row['image_id'])[:3000]
    if len(chosen) != 3000 or any(row['partition'] != 'pool' for row in chosen):
        raise ValueError('Invalid release fixture')
    ids = [row['image_id'] for row in chosen]
    hardware = __import__('subprocess').run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
        '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    runs = []
    predictions = []
    raw = []
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='final-fixture-', dir=out) as folder:
        fixture = Path(folder)
        for row in chosen:
            path = T.pool_path(root, row['relative_path'])
            (fixture / path.name).symlink_to(path)
        for index in range(3):
            T.readiness()
            output = staging / ('gpu' + str(index) + '.csv')
            invocation = {}
            if index > 0:
                output.write_bytes(b'stale CSV fixture; must be replaced by current inference')
                invocation = {'batch_size': 32 if index == 1 else 16, 'workers': 2 if index == 1 else 1,
                              'threshold': .5 if index == 1 else .9,
                              'cli_name': 'inference_script.py' if index == 1 else 'inferencescript.py'}
            result = run_cli(root, fixture, bundle, output, 'cuda', 600, **invocation)
            runs.append(result)
            if result['timed_out'] or result['returncode']:
                raise RuntimeError(json.dumps(result))
            if index > 0 and 'replaced atomically' not in result['stderr']:
                raise ValueError('Existing-output compatibility path was not exercised')
            values = read_predictions(output, ids)
            if len(set(values)) < 2:
                raise ValueError('Functional sanity failure: constant fixture scores')
            predictions.append(values)
            raw.append(output.read_bytes())
        if not raw[0] == raw[1] == raw[2]:
            raise ValueError('GPU CLI replay bytes differ')
        cpu_output = staging / 'cpu.csv'
        cpu = run_cli(root, fixture, bundle, cpu_output, 'cpu', 1800)
        cpu_values = None
        if not cpu['timed_out'] and cpu['returncode'] == 0:
            try:
                cpu_values = read_predictions(cpu_output, ids)
            except (ValueError, OSError) as exc:
                cpu['validation_error'] = str(exc)
        enabled, difference, flips = fallback_eligible(cpu, cpu_values, predictions[0])
        policy = {'schema_version': 1, 'bundle_sha256': bundle_hash,
                  'cpu_fallback_enabled': enabled,
                  'reason': 'Measured CPU correctness, agreement, zero label flips and <=1200s' if enabled else 'CPU fallback did not pass all fixed gates'}
        immutable_json(staging / 'release_policy.json', policy)
        fallback = None
        if enabled:
            previous = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            try:
                fallback_output = staging / 'fallback.csv'
                fallback = run_cli(root, fixture, bundle, fallback_output, 'cuda', 1800)
            finally:
                if previous is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = previous
            if fallback['timed_out'] or fallback['returncode'] or fallback_output.read_bytes() != cpu_output.read_bytes():
                raise ValueError('Exact CUDA-unavailable fallback path failed; release stopped')
            if 'verified CPU fallback' not in fallback['stderr']:
                raise ValueError('Fallback warning missing')
            read_predictions(fallback_output, ids)
    immutable_json(staging / 'release_results.json', {'status': 'completed', 'candidate': 'CT_SS',
        'bundle_sha256': bundle_hash, 'bundle_bytes': bundle.stat().st_size,
        'fixture_count': 3000, 'fixture_ids_sha256': hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
        'gpu_inventory': hardware, 'gpu_runs': runs, 'gpu_replay_bytes_equal': True,
        'gpu_csv_sha256': hashlib.sha256(raw[0]).hexdigest(), 'cpu_run': cpu,
        'cpu_probability_difference': difference, 'cpu_label_flips': flips,
        'cpu_fallback_enabled': enabled, 'cuda_unavailable_path_run': fallback,
        'sanity': 'Finite parameters and probabilities, nonconstant outputs, exact row/ID mapping. No performance claim.',
        'cache_state': 'Fresh processes; OS disk cache not cleared. Not a cold-disk benchmark.',
        'scope': 'Final all-data weights; pool-only operational fixture. RTX5060Ti measurement, not T4.'})
    immutable_json(staging / 'provenance.json', {'metadata': metadata, 'identity': identity,
        'bundle_sha256': bundle_hash, 'inference_source_sha256': T.sha256(root / 'inference_script.py'),
        'holdout_readout_receipt_sha256': T.sha256(out / 'holdout/receipt.json')})
    for name in ('inference_script.py', 'deepfake_core_x1.py', 'requirements.txt'):
        shutil.copyfile(root / name, staging / name)
    shutil.copyfile(root / 'inferencescript.py', staging / 'inferencescript.py')
    immutable_json(staging / 'release_bundle.json', {'path': str(final / 'model_weights.pt'),
        'sha256': bundle_hash, 'bytes': bundle.stat().st_size, 'policy': policy,
        'protocol_sha256': identity['protocol_sha256']})
    commit_stage(staging, final, identity)
    publish(final)
    export_delivery(root, final)
    gc.collect()

