import argparse
import base64
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

MANIFEST_HASH = '30f7bb575793da6ab0b119abdaa96b8377bc44645de134035517a09bc0f75c85'
E1_ROOT = Path('<HOME>/deepfake_hackathon/e1-20261031-v1/results')
FIT = 'p1_f0_convnext_tiny_s29_bce'
MEMBERS = {'C': [('convnext_tiny', 17)], 'A': [('convnext_tiny', 17), ('swin_t', 17)],
           'B': [('convnext_tiny', 17), ('convnext_tiny', 29)]}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gate(p1, p2):
    if p1.get('complete') is not True or p2.get('complete') is not True:
        return None
    selected = p1.get('selected')
    if selected not in (None, 'A', 'B', 'C') or type(p1.get('review_required')) is not bool:
        raise ValueError('Invalid P1 decision')
    if type(p2.get('followup_required')) is not bool or 'promoted_loss' not in p2:
        raise ValueError('Invalid P2 decision')
    if p2['promoted_loss'] is not None and not isinstance(p2['promoted_loss'], str):
        raise ValueError('Invalid promoted loss')
    return selected if selected and not p1['review_required'] and p2['promoted_loss'] is None and not p2['followup_required'] else None


def verify_pool(root, expected_count=5950, expected_hash=MANIFEST_HASH):
    manifest = root / 'pool_manifest.csv'
    if expected_hash and sha256(manifest) != expected_hash:
        raise ValueError('Pool manifest changed')
    with manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_count or any(r['partition'] != 'pool' for r in rows):
        raise ValueError('Invalid pool membership')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate pool IDs')
    base = (root / 'pool-images').resolve(strict=True)
    if (root / 'pool-images').is_symlink() or not base.is_relative_to(root.resolve()):
        raise ValueError('Unsafe pool root')
    for r in rows:
        path = (root / r['relative_path']).resolve(strict=True)
        if not path.is_relative_to(base) or path == base or not path.is_file():
            raise ValueError('Image outside pool')
        if path.stem != r['image_id'] or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'):
            raise ValueError('Image ID or extension mismatch')
        if sha256(path) != r['sha256']:
            raise ValueError('Pool image hash mismatch')
    return sorted(rows, key=lambda r: r['image_id'])


def read_predictions(path, expected_ids):
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['image_id', 'label', 'confidence']:
            raise ValueError('Invalid prediction header')
        rows = list(reader)
    if [r['image_id'] for r in rows] != expected_ids:
        raise ValueError('Prediction IDs or order mismatch')
    values = []
    for r in rows:
        p = float(r['confidence'])
        if not math.isfinite(p) or not 0 <= p <= 1 or r['label'] != str(int(p >= .5)):
            raise ValueError('Invalid prediction')
        values.append(p)
    return values


def make_bundle(root, selected, output):
    import torch
    with (root / 'e1-baselines/checkpoint-register.csv').open(newline='', encoding='utf-8') as f:
        register = list(csv.DictReader(f))
    members, sources = [], []
    for architecture, seed in MEMBERS[selected]:
        if seed == 17:
            matches = [r for r in register if r['fold'] == '0' and r['architecture'] == architecture and r['seed'] == '17']
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous E1 checkpoint register entry')
            record = matches[0]
            path = E1_ROOT / ('f0_' + architecture + '_final_ema.pt')
            if record['checkpoint_remote_path'] != str(path):
                raise ValueError('E1 path mismatch')
        else:
            directory = root / 'results' / FIT
            record = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
            path = directory / 'checkpoint.pt'
            if (record.get('fit_id'), record.get('fold'), record.get('architecture'), record.get('seed'), record.get('loss')) != (FIT, 0, architecture, 29, 'bce'):
                raise ValueError('P1 summary identity mismatch')
        digest = sha256(path)
        if digest != record['checkpoint_sha256'] or path.stat().st_size != int(record['checkpoint_bytes']):
            raise ValueError('Checkpoint identity mismatch')
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        if set(bundle) != {'metadata_json', 'members'} or len(bundle['members']) != 1 or bundle['members'][0]['architecture'] != architecture:
            raise ValueError('Unexpected checkpoint bundle')
        member = bundle['members'][0]
        if not all(isinstance(v, torch.Tensor) for v in member['state_dict'].values()):
            raise ValueError('Checkpoint state is not tensor-only')
        members.append(member)
        sources.append({'path': str(path), 'sha256': digest, 'architecture': architecture, 'seed': seed, 'fold': 0})
    metadata = {'schema_version': 1, 'input_size': 224, 'normalization': 'imagenet', 'interpolation': 'PIL-bilinear',
                'combination': 'logit_mean', 'threshold': .5, 'temperature': 1.0, 'precision': 'fp32',
                'purpose': 'Deployment feasibility only; fold 0 development checkpoints, not final frozen, full-P, or all-data weights',
                'candidate': selected, 'provenance': sources}
    torch.save({'metadata_json': json.dumps(metadata, sort_keys=True), 'members': members}, output)
    return metadata


def run_cli(root, fixture, bundle, output, device, timeout):
    # runpy executes the unchanged CLI, while the exit hook records PyTorch allocator peaks.
    hook = ("import atexit,runpy,sys,torch; "
            "atexit.register(lambda: print('P1_CUDA_PEAK '+str(torch.cuda.max_memory_allocated())+' '+str(torch.cuda.max_memory_reserved())) if torch.cuda.is_initialized() else None); "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [sys.executable, '-c', hook, str(root / 'inference_script.py'), '--test_dir', str(fixture),
               '--weights', str(bundle), '--output', str(output), '--device', device, '--batch_size', '64', '--workers', '0', '--precision', 'fp32']
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'})
    start = time.monotonic()
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout, tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr:
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        rss_peak = gpu_peak = None
        timed_out = False
        while proc.poll() is None:
            if time.monotonic() - start > timeout:
                timed_out = True
                proc.kill()
                break
            try:
                status = Path('/proc') / str(proc.pid) / 'status'
                for line in status.read_text().splitlines():
                    if line.startswith('VmHWM:'):
                        rss_peak = max(rss_peak or 0, int(line.split()[1]) * 1024)
            except (OSError, ValueError):
                pass
            if device == 'cuda':
                try:
                    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
                    for line in q.stdout.splitlines():
                        pid, memory = [x.strip() for x in line.split(',')]
                        if int(pid) == proc.pid:
                            gpu_peak = max(gpu_peak or 0, int(memory) * 1024 * 1024)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            time.sleep(.2)
        proc.wait()
        elapsed = time.monotonic() - start
        stdout.seek(0); stderr.seek(0)
        out, err = stdout.read(), stderr.read()
    result = {'device': device, 'wall_seconds': elapsed, 'timeout_seconds': timeout, 'timed_out': timed_out,
              'returncode': proc.returncode, 'stdout': out, 'stderr': err, 'peak_rss_bytes_observed': rss_peak,
              'peak_gpu_bytes_nvidia_smi_observed': gpu_peak}
    return result


def emit(name, data):
    print('P1_ARTIFACT ' + name + ' ' + base64.b64encode(data).decode(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('<HOME>/deepfake_hackathon/p1-20261031-v1'))
    args = parser.parse_args(); root = args.root.resolve()
    p1 = json.loads((root / 'analysis/p1_decision.json').read_text(encoding='utf-8'))
    p2 = json.loads((root / 'analysis/p2_decision.json').read_text(encoding='utf-8'))
    selected = gate(p1, p2)
    if selected is None:
        result = {'status': 'deferred', 'p1_decision': p1, 'p2_decision': p2}
        emit('release_results.json', json.dumps(result, indent=2).encode())
        return
    hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    from train_p1 import readiness
    readiness()
    rows = verify_pool(root); chosen = rows[:3000]; ids = [r['image_id'] for r in chosen]
    outdir = root / 'release'; outdir.mkdir(exist_ok=True)
    bundle = outdir / 'development_feasibility.pt'
    metadata = make_bundle(root, selected, bundle)
    with tempfile.TemporaryDirectory(prefix='p1-release-', dir=outdir) as tmp:
        fixture = Path(tmp) / 'fixture'; fixture.mkdir()
        for r in chosen:
            source = (root / r['relative_path']).resolve(strict=True)
            (fixture / source.name).symlink_to(source)
        runs, predictions, raw = [], [], []
        for i in range(3):
            readiness()
            output = Path(tmp) / ('gpu' + str(i) + '.csv')
            runs.append(run_cli(root, fixture, bundle, output, 'cuda', 300))
            if runs[-1]['timed_out'] or runs[-1]['returncode']:
                raise RuntimeError(json.dumps(runs[-1]))
            predictions.append(read_predictions(output, ids)); raw.append(output.read_bytes())
        if not (raw[0] == raw[1] == raw[2]):
            raise ValueError('GPU replay bytes differ')
        output = Path(tmp) / 'cpu.csv'
        cpu = run_cli(root, fixture, bundle, output, 'cpu', 1800)
        cpu_correct = False; maxdiff = flips = None
        if not cpu['timed_out'] and cpu['returncode'] == 0:
            try:
                cpu_values = read_predictions(output, ids)
                maxdiff = max(abs(a-b) for a,b in zip(predictions[0], cpu_values))
                flips = sum((a >= .5) != (b >= .5) for a,b in zip(predictions[0], cpu_values))
                cpu_correct = True
            except (ValueError, OSError) as exc:
                cpu['validation_error'] = str(exc)
        result = {'status': 'completed', 'candidate': selected, 'fixture_count': len(ids),
                  'fixture_ids_sha256': hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
                  'pool_manifest_sha256': sha256(root / 'pool_manifest.csv'), 'protocol_sha256': sha256(root / 'experiments/p1_protocol.json'),
                  'gpu_inventory': hardware, 'cpu_count': os.cpu_count(), 'gpu_runs': runs, 'gpu_replay_bytes_equal': True,
                  'cpu_run': cpu, 'cpu_max_probability_difference': maxdiff, 'cpu_label_flips': flips,
                  'cpu_diagnostic_tolerance': 1e-5, 'cpu_within_tolerance': maxdiff <= 1e-5 if cpu_correct else None,
                  'cpu_output_correct': cpu_correct,
                  'cpu_fallback_eligible_for_review': cpu_correct and maxdiff <= 1e-5 and flips == 0 and cpu['wall_seconds'] <= 1200,
                  'cpu_fallback_enabled': False, 'cache_state': 'Fresh processes; disk cache was not cleared.',
                  'scope': 'Deployment feasibility on development checkpoints only; no holdout or test evaluation.'}
    release = {'bundle_path': str(bundle), 'bundle_sha256': sha256(bundle), 'metadata': metadata}
    for name, obj in [('release_results.json', result), ('release_bundle.json', release)]:
        data = (json.dumps(obj, indent=2) + '\n').encode(); (outdir / name).write_bytes(data); emit(name, data)


if __name__ == '__main__':
    main()
