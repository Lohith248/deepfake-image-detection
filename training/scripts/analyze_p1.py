"""Audit the frozen P1/P2 pool experiment without loading images or checkpoints."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e1_metrics import auc, sigmoid64, summarize


ROOT = Path(__file__).resolve().parents[1]
POOL_HASH = '30f7bb575793da6ab0b119abdaa96b8377bc44645de134035517a09bc0f75c85'
E1_HASH = 'ad0f1f75abac558dca4a2e1459463b8ce71200ad7b56c0bb7acb82cbfad8d1e8'
WEIGHT_HASHES = {
    'convnext_tiny': '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d',
    'swin_t': '704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b',
}
METRICS = ('original_auc', 'robust_auc_proxy', 'J04')
SUBSETS = {'all_five': (0, 1, 2, 3, 4), 'fresh_2_3_4': (2, 3, 4), 'selection_used_0_1': (0, 1)}
LIMITS = ('Intervals describe conditional data uncertainty for these saved fits. They do not estimate '
          'training variance or correct selection optimism. Training complements overlap. '
          'Source groups may not capture every shared generator or source dependency.')


@dataclass
class Fit:
    fold: int
    image_ids: np.ndarray
    y: np.ndarray
    group_ids: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    metrics: dict
    receipt: dict


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical_conditions():
    families = {
        'jpeg': ['jpeg'], 'resolution': ['downsample', 'pixelate'],
        'photometric': ['gamma', 'saturation', 'tint'], 'tone': ['posterize', 'solarize', 'inversion'],
        'occlusion': ['neutral_patch', 'color_patch'], 'noise': ['gaussian_noise', 'impulse_noise'],
        'convolution': ['gaussian_blur', 'motion_blur'], 'geometric': ['rotation', 'resize_jpeg'],
    }
    result = [dict(name='original', family='original', operator='original', severity=0, weight=0.)]
    for family, operators in families.items():
        for operator in operators:
            for severity, weight in enumerate((.5, .35, .15)):
                result.append(dict(name=f'{operator}_s{severity+1}', family=family, operator=operator,
                                   severity=severity, weight=weight / len(operators) / len(families)))
    result.extend(dict(name=op, family='diagnostic', operator=op, severity=0, weight=0.)
                  for op in ('centre_only', 'border_only'))
    return result


def read_truth(root, protocol):
    pool = root / 'pool_manifest.csv'
    path = pool if pool.is_file() else root / 'splits/train_manifest.csv'
    expected_hash = POOL_HASH if pool.is_file() else protocol['recipe']['split_manifest_sha256']
    if sha256(path) != expected_hash:
        raise ValueError('Canonical manifest hash changed')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if pool.is_file() and any(row['partition'] != 'pool' for row in rows):
        raise ValueError('Pool manifest contains a non-pool partition')
    rows = [row for row in rows if row['partition'] == 'pool']
    truth = {row['image_id']: row for row in rows}
    if len(truth) != 5950 or len(truth) != len(rows):
        raise ValueError('Pool IDs are missing or duplicated')
    groups = {}
    for row in rows:
        if row['label'] not in ('0', '1') or row['fold'] not in ('0', '1', '2', '3', '4') or not row['group_id']:
            raise ValueError('Invalid canonical pool label, fold, or source group')
        if row['group_id'] in groups and groups[row['group_id']] != row['fold']:
            raise ValueError('Canonical source group crosses folds')
        groups[row['group_id']] = row['fold']
    return truth, {'path': str(path), 'sha256': sha256(path), 'pool_images': len(rows)}


def validate_chunk(chunk, truth, fold, conditions):
    required = {'image_ids', 'group_ids', 'y', 'logits', 'probabilities', 'condition_names', 'condition_weights'}
    if not required.issubset(chunk):
        raise ValueError('Prediction schema is incomplete')
    ids = chunk['image_ids']
    if ids.ndim != 1 or ids.dtype.kind not in 'US' or len(set(ids)) != len(ids):
        raise ValueError('Prediction IDs must be unique strings')
    expected = sorted(image_id for image_id, row in truth.items() if int(row['fold']) == fold)
    if set(ids.tolist()) != set(expected):
        raise ValueError('Prediction IDs differ from canonical pool fold')
    n = len(ids)
    if chunk['y'].shape != (n,) or chunk['group_ids'].shape != (n,) or chunk['group_ids'].dtype.kind not in 'US':
        raise ValueError('Prediction label or group shape changed')
    if not np.isin(chunk['y'], (0, 1)).all() or len(np.unique(chunk['y'])) != 2:
        raise ValueError('Prediction labels must include both binary classes')
    for i, image_id in enumerate(ids):
        row = truth[str(image_id)]
        if int(row['label']) != chunk['y'][i] or row['group_id'] != chunk['group_ids'][i]:
            raise ValueError('Prediction labels or groups differ from canonical manifest')
    if not np.array_equal(chunk['condition_names'], [c['name'] for c in conditions]):
        raise ValueError('Prediction condition names or order changed')
    if not np.array_equal(chunk['condition_weights'], [c['weight'] for c in conditions]):
        raise ValueError('Prediction condition weights changed')
    for key in ('logits', 'probabilities'):
        values = chunk[key]
        if values.shape != (len(conditions), n) or values.dtype.kind != 'f' or not np.isfinite(values).all():
            raise ValueError('Invalid prediction ' + key)
    scores = chunk['probabilities']
    if ((scores < 0) | (scores > 1)).any() or not np.allclose(scores, sigmoid64(chunk['logits']), rtol=0, atol=1e-15):
        raise ValueError('Saved probabilities differ from FP64 sigmoid of saved logits')
    order = np.argsort(ids, kind='stable')
    return {key: chunk[key][:, order] if key in ('logits', 'probabilities') else chunk[key][order]
            for key in ('image_ids', 'group_ids', 'y', 'logits', 'probabilities')}


def verify_sources(root, hashes):
    if not hashes or not isinstance(hashes, dict):
        raise ValueError('Missing source identities')
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError('Source identity mismatch for ' + name)


def fit_paths(raw, stage, fold, architecture, seed, loss):
    prefix = f'{stage}_f{fold}_{architecture}_s{seed}_{loss}'
    if stage == 'e1':
        prefix = f'f{fold}_{architecture}'
    flat = (raw / (prefix + '_e20_predictions.npz'), raw / (prefix + '_summary.json'), raw / (prefix + '_curves.json'))
    directory = raw / prefix
    nested = (directory / 'predictions.npz', directory / 'summary.json', directory / 'curves.json')
    return nested if directory.is_dir() else flat


def load_fit(root, raw, environment, truth, conditions, stage, fold, architecture, seed=17, loss='bce'):
    prediction, summary_path, curves_path = fit_paths(raw, stage, fold, architecture, seed, loss)
    record = read_json(summary_path)
    curves = read_json(curves_path)
    identity = record if stage == 'e1' else record['identity']
    if stage != 'e1' and identity != environment:
        raise ValueError('Fit identity differs from authenticated environment')
    if identity['source_hashes'] != environment['source_hashes'] or identity['pool_manifest_sha256'] != POOL_HASH:
        raise ValueError('Fit source or pool identity changed')
    if stage == 'e1' and identity['protocol_sha256'] != E1_HASH:
        raise ValueError('E1 fit protocol changed')
    if record['fold'] != fold or record['architecture'] != architecture or record['seed'] != seed or record['epochs'] != 20:
        raise ValueError('Fit recipe differs from scheduled fit')
    if stage != 'e1' and (record['stage'] != stage or record['loss'] != loss):
        raise ValueError('Fit stage or loss changed')
    provenance = record['provenance']
    if provenance['sha256'] != WEIGHT_HASHES[architecture] or not provenance['enum'].endswith('.IMAGENET1K_V1'):
        raise ValueError('Original ImageNet initialization identity changed')
    replay_error = record['cli_validation_max_probability_difference']
    if record['cli_replay_byte_identical'] is not True or not np.isfinite(replay_error) or not 0 <= replay_error <= 1e-5:
        raise ValueError('Fit saved-checkpoint replay did not pass')
    if [row['epoch'] for row in curves] != list(range(1, 21)):
        raise ValueError('Fit does not contain the complete 20-epoch curve')
    with np.load(prediction, allow_pickle=False) as archive:
        chunk = {key: archive[key].copy() for key in archive.files}
    chunk = validate_chunk(chunk, truth, fold, conditions)
    if record['validation_images'] != len(chunk['y']) or record['fit_images'] != len(truth) - len(chunk['y']):
        raise ValueError('Fit sample counts differ from canonical fold')
    metrics = summarize(chunk['y'], chunk['probabilities'], conditions)
    for key in METRICS:
        if abs(metrics[key] - record['final_metrics'][key]) > 1e-12 or abs(metrics[key] - curves[-1]['metrics'][key]) > 1e-12:
            raise ValueError('Saved final metrics differ from independently recomputed AUC')
    receipt = {'stage': stage, 'fold': fold, 'architecture': architecture, 'seed': seed, 'loss': loss,
               'prediction_sha256': sha256(prediction), 'summary_sha256': sha256(summary_path),
               'curves_sha256': sha256(curves_path), 'checkpoint_sha256': record['checkpoint_sha256'],
               'source_hashes': identity['source_hashes'], 'initialization_sha256': provenance['sha256'],
               'prediction_path': str(prediction)}
    completion = summary_path.parent / 'receipt.json'
    if stage != 'e1' and prediction.name == 'predictions.npz' and not completion.is_file():
        raise ValueError('Completed nested fit is missing its receipt')
    if stage != 'e1' and completion.is_file():
        completed = read_json(completion)
        if completed['identity'] != identity:
            raise ValueError('Completed fit receipt identity changed')
        for name, path in (('predictions.npz', prediction), ('summary.json', summary_path), ('curves.json', curves_path)):
            if completed['artifacts'][name] != sha256(path):
                raise ValueError('Completed fit artifact hash changed')
        if completed['artifacts']['checkpoint.pt'] != record['checkpoint_sha256']:
            raise ValueError('Completed checkpoint identity differs from summary')
        receipt['completion_receipt_sha256'] = sha256(completion)
    return Fit(fold=fold, metrics=metrics, receipt=receipt, **chunk)


def p1_gate(deltas, fresh_swin_original_advantage, gate):
    eligible = {name: bool(np.mean(deltas[name]) >= gate['minimum_mean_delta_J04'] and
                           np.count_nonzero(np.asarray(deltas[name]) > 0) >= gate['minimum_strict_positive_folds'])
                for name in ('A', 'B')}
    if eligible['A'] and eligible['B']:
        provisional = 'B' if np.mean(deltas['B']) - np.mean(deltas['A']) > gate['seed_pair_must_exceed_arch_pair_by'] else 'A'
    elif eligible['A']:
        provisional = 'A'
    elif eligible['B']:
        provisional = 'B'
    else:
        provisional = 'C'
    review = bool(fresh_swin_original_advantage > gate['review_if_fresh_swin_original_advantage_exceeds'])
    return {'complete': True, 'selected': None if review else provisional, 'provisional_selected': provisional,
            'review_required': review, 'eligible': eligible,
            'mean_delta_J04': {name: float(np.mean(values)) for name, values in deltas.items()},
            'strict_positive_folds': {name: int(np.count_nonzero(np.asarray(values) > 0)) for name, values in deltas.items()},
            'fresh_swin_original_advantage': float(fresh_swin_original_advantage),
            'combination': 'equal saved-logit mean followed by FP64 sigmoid',
            'final_procedure_frozen': False}


def p2_gate(deltas, gate):
    eligible = {loss: bool(np.all(np.asarray(values['original_auc']) > 0) and
                           np.all(np.asarray(values['J04']) > 0) and
                           np.mean(values['J04']) >= gate['minimum_mean_delta_J04'])
                for loss, values in deltas.items()}
    passing = [loss for loss in deltas if eligible[loss]]
    promoted = max(passing, key=lambda loss: (float(np.mean(deltas[loss]['J04'])), loss == 'smooth_01')) if passing else None
    return {'complete': True, 'promoted_loss': promoted, 'eligible': eligible,
            'followup_required': promoted is not None,
            'confirmation_folds': [2, 3, 4] if promoted else [],
            'confirmation_mean_delta_J04_minimum': .003, 'confirmation_strict_positive_folds_minimum': 2,
            'final_loss_selected': None, 'automatic_final_ensemble_change': False,
            'combination_comparison_required_if_recipe_changes': True}


def bootstrap_aucs(y, scores, group_index, counts):
    """Exact E1 weighted tied-score AUC for bootstrap group multiplicities."""
    result = np.empty((len(counts), len(scores)), dtype=np.float64)
    for ci, p in enumerate(scores):
        order = np.argsort(p, kind='stable'); starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
        ys = y[order]; groups = group_index[order]
        for b in range(0, len(counts), 128):
            w = counts[b:b+128, groups].astype(np.float64)
            positive = np.add.reduceat(w*ys, starts, axis=1); negative = np.add.reduceat(w*(1-ys), starts, axis=1)
            denominator = positive.sum(1)*negative.sum(1)
            numerator = (positive*(np.cumsum(negative, axis=1)-.5*negative)).sum(1)
            result[b:b+128, ci] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return result


def interval(values):
    valid = np.isfinite(values)
    return {'ci95': [float(x) for x in np.percentile(values[valid], [2.5, 97.5])] if valid.any() else None,
            'valid_draws': int(valid.sum()), 'invalid_draws': int((~valid).sum())}


def bootstrap_metrics(fit, scores, conditions, draws, seed):
    unique, group_index = np.unique(fit.group_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=draws)
    boot = bootstrap_aucs(fit.y, scores, group_index, counts)
    robust = boot @ np.array([c['weight'] for c in conditions])
    return {'original_auc': boot[:, 0], 'robust_auc_proxy': robust, 'J04': boot[:, 0] + .4*robust}


def summarize_subset(fits, scores, conditions, folds, boots=None):
    per_fold = {str(f): summarize(fits[f].y, scores[f], conditions) for f in folds}
    primary = {key: float(np.mean([per_fold[str(f)][key] for f in folds])) for key in METRICS}
    primary['condition_auc'] = {c['name']: float(np.mean([per_fold[str(f)]['condition_auc'][c['name']] for f in folds]))
                                for c in conditions}
    primary['family_auc'] = {name: float(np.mean([per_fold[str(f)]['family_auc'][name] for f in folds]))
                             for name in per_fold[str(folds[0])]['family_auc']}
    pooled = summarize(np.concatenate([fits[f].y for f in folds]), np.concatenate([scores[f] for f in folds], axis=1), conditions)
    result = {'folds': list(folds), 'mean_fold': primary, 'pooled_secondary': {key: pooled[key] for key in METRICS},
              'per_fold': per_fold}
    if boots is not None:
        result['mean_fold_intervals'] = {key: interval(np.mean([boots[f][key] for f in folds], axis=0)) for key in METRICS}
    return result


def paired_differences(fits, scores_a, scores_b, conditions, folds, boots_a, boots_b):
    delta = {key: [summarize(fits[f].y, scores_a[f], conditions)[key] - summarize(fits[f].y, scores_b[f], conditions)[key]
                   for f in folds] for key in METRICS}
    return {'folds': list(folds), 'per_fold_delta': delta,
            'mean_fold_delta': {key: float(np.mean(values)) for key, values in delta.items()},
            'paired_intervals': {key: interval(np.mean([boots_a[f][key] - boots_b[f][key] for f in folds], axis=0))
                                 for key in METRICS}}


def analyze_p1(fits, conditions, protocol, draws):
    baseline = {f: fits[('convnext_tiny', 17, 'bce', f)] for f in range(5)}
    primary = {name: {} for name in ('A', 'B', 'C', 'Swin17', 'Conv29')}
    secondary = {name: {} for name in ('A', 'B', 'C')}
    for f in range(5):
        conv = baseline[f]; swin = fits[('swin_t', 17, 'bce', f)]; seed29 = fits[('convnext_tiny', 29, 'bce', f)]
        primary['A'][f] = sigmoid64(np.mean([conv.logits, swin.logits], axis=0, dtype=np.float64))
        primary['B'][f] = sigmoid64(np.mean([conv.logits, seed29.logits], axis=0, dtype=np.float64))
        primary['C'][f] = conv.probabilities
        primary['Swin17'][f] = swin.probabilities; primary['Conv29'][f] = seed29.probabilities
        secondary['A'][f] = (conv.probabilities + swin.probabilities) / 2
        secondary['B'][f] = (conv.probabilities + seed29.probabilities) / 2
        secondary['C'][f] = conv.probabilities
    boots = {}
    for name, scores in primary.items():
        print('Paired bootstrap ' + name, flush=True)
        boots[name] = {f: bootstrap_metrics(baseline[f], scores[f], conditions, draws,
                                            protocol['analysis']['bootstrap_seed'] + f) for f in range(5)}
    reports = {}
    for subset, folds in SUBSETS.items():
        reports[subset] = {
            'primary_logit_mean': {name: summarize_subset(baseline, scores, conditions, folds, boots[name])
                                   for name, scores in primary.items()},
            'secondary_probability_mean': {name: summarize_subset(baseline, scores, conditions, folds)
                                           for name, scores in secondary.items()},
            'paired_comparisons': {a + '_minus_' + b: paired_differences(baseline, primary[a], primary[b], conditions,
                                                                       folds, boots[a], boots[b])
                                    for a, b in (('A', 'C'), ('B', 'C'), ('B', 'A'), ('Swin17', 'C'))},
        }
    all_report = reports['all_five']['paired_comparisons']
    deltas = {name: all_report[name + '_minus_C']['per_fold_delta']['J04'] for name in ('A', 'B')}
    reversal = reports['fresh_2_3_4']['paired_comparisons']['Swin17_minus_C']['mean_fold_delta']['original_auc']
    return {'complete': True, 'subsets': reports, 'decision': p1_gate(deltas, reversal, protocol['p1_gate'])}, boots['C']


def analyze_p2(fits, conditions, protocol, draws, baseline_boots=None):
    baseline = {f: fits[('convnext_tiny', 17, 'bce', f)] for f in (0, 1)}
    scores = {f: baseline[f].probabilities for f in (0, 1)}
    if baseline_boots is None:
        baseline_boots = {f: bootstrap_metrics(baseline[f], scores[f], conditions, draws,
                                              protocol['analysis']['bootstrap_seed'] + f) for f in (0, 1)}
    report = {}; deltas = {}
    for loss in ('gce_q07', 'smooth_01'):
        other = {f: fits[('convnext_tiny', 17, loss, f)].probabilities for f in (0, 1)}
        boot = {f: bootstrap_metrics(baseline[f], other[f], conditions, draws,
                                     protocol['analysis']['bootstrap_seed'] + f) for f in (0, 1)}
        comparison = paired_differences(baseline, other, scores, conditions, (0, 1), boot, baseline_boots)
        report[loss] = {'metrics': summarize_subset(baseline, other, conditions, (0, 1), boot), 'minus_BCE': comparison}
        deltas[loss] = comparison['per_fold_delta']
    return {'complete': True, 'folds': [0, 1], 'screen': report, 'decision': p2_gate(deltas, protocol['p2_gate']),
            'interpretation': 'Screen only. A passing loss needs fresh-fold confirmation. A changed ensemble needs a new bounded complete-procedure comparison.'}


def validate_protocol(protocol):
    expected = [('p1', f, a, 17, 'bce') for f in (2, 3, 4) for a in ('convnext_tiny', 'swin_t')]
    expected += [('p1', f, 'convnext_tiny', 29, 'bce') for f in range(5)]
    expected += [('p2', f, 'convnext_tiny', 17, loss) for loss in ('gce_q07', 'smooth_01') for f in (0, 1)]
    if [(j['stage'], j['fold'], j['architecture'], j['seed'], j['loss']) for j in protocol['jobs']] != expected:
        raise ValueError('Frozen fit schedule changed')
    gate = protocol['p1_gate']
    if (gate['minimum_mean_delta_J04'], gate['minimum_strict_positive_folds'], gate['seed_pair_must_exceed_arch_pair_by'],
        gate['review_if_fresh_swin_original_advantage_exceeds'], protocol['p2_gate']['minimum_mean_delta_J04']) != (.003, 4, .002, .005, .004):
        raise ValueError('Frozen decision thresholds changed')
    if protocol['e1_protocol_sha256'] != E1_HASH or protocol['evaluation_epochs'] != [20]:
        raise ValueError('Frozen protocol identity changed')


def load_inputs(root, raw, stage):
    protocol_path = root / 'experiments/p1_protocol.json'; protocol = read_json(protocol_path)
    validate_protocol(protocol)
    if sha256(root / 'experiments/e1_protocol.json') != E1_HASH:
        raise ValueError('E1 protocol hash mismatch')
    truth, manifest_receipt = read_truth(root, protocol)
    conditions = canonical_conditions()
    baseline_raw = root / 'e1-baselines/raw'
    if not baseline_raw.is_dir():
        baseline_raw = root / 'runs/e1-primary/raw'
    environments = {'e1': read_json(baseline_raw / 'environment.json'), 'new': read_json(raw / 'environment.json')}
    for name, environment in environments.items():
        if environment['conditions'] != conditions or environment['pool_manifest_sha256'] != POOL_HASH:
            raise ValueError('Environment condition schema or pool identity changed')
        required = {'deepfake_core.py', 'e1_transforms.py', 'e1_metrics.py', 'scripts/train_e1.py', 'inference_script.py'}
        if name == 'new':
            required |= {'scripts/train_p1.py', 'experiments/e1_protocol.json', 'experiments/p1_protocol.json'}
        if not required.issubset(environment['source_hashes']):
            raise ValueError('Environment omits required source identities')
        verify_sources(root, environment['source_hashes'])
        if name == 'new' and environment['source_hashes'].get('experiments/p1_protocol.json') != sha256(protocol_path):
            raise ValueError('P1 protocol differs from fit source identity')
        if name == 'new' and (root / 'source-hashes.json').is_file():
            if environment['source_hashes_sha256'] != sha256(root / 'source-hashes.json') or read_json(root / 'source-hashes.json') != environment['source_hashes']:
                raise ValueError('Packaged source listing differs from fit identity')
    if environments['e1']['protocol_sha256'] != E1_HASH:
        raise ValueError('E1 environment protocol changed')
    fits = {}
    for architecture in ('convnext_tiny', 'swin_t'):
        for fold in (0, 1):
            fits[(architecture, 17, 'bce', fold)] = load_fit(root, baseline_raw, environments['e1'], truth, conditions,
                                                           'e1', fold, architecture)
    for job in protocol['jobs']:
        if stage == 'p1' and job['stage'] == 'p2':
            continue
        key = (job['architecture'], job['seed'], job['loss'], job['fold'])
        fits[key] = load_fit(root, raw, environments['new'], truth, conditions, **job)
    for fold in range(5):
        in_fold = [fit for fit in fits.values() if fit.fold == fold]
        for fit in in_fold[1:]:
            for key in ('image_ids', 'y', 'group_ids'):
                if not np.array_equal(getattr(in_fold[0], key), getattr(fit, key)):
                    raise ValueError('Paired fit alignment differs after canonical ordering')
    receipt = {'protocol_sha256': sha256(protocol_path), 'manifest': manifest_receipt,
               'analyzer_sha256': sha256(Path(__file__)), 'metrics_sha256': sha256(root / 'e1_metrics.py'),
               'environment_sha256': sha256(raw / 'environment.json'),
               'baseline_environment_sha256': sha256(baseline_raw / 'environment.json'),
               'inputs': [fit.receipt for fit in fits.values()]}
    return fits, conditions, protocol, receipt


def markdown(analysis):
    lines = ['# P1/P2 pool analysis', '', LIMITS, '']
    if not analysis['complete']:
        return '\n'.join(lines + ['Analysis is incomplete. No selection is authorized.', '', analysis['error'], ''])
    if 'p1' in analysis:
        decision = analysis['p1']['decision']
        lines += [f"P1 selected candidate is {decision['selected']}. Review required is {decision['review_required']}.",
                  'Primary combinations average saved logits before FP64 sigmoid. Probability means are secondary.', '']
        for subset, report in analysis['p1']['subsets'].items():
            lines.append(subset.replace('_', ' ') + '.')
            for name in ('A', 'B', 'C'):
                metrics = report['primary_logit_mean'][name]['mean_fold']
                lines.append(f"- {name} mean-fold original AUC {metrics['original_auc']:.6f}, variant proxy {metrics['robust_auc_proxy']:.6f}, J0.4 {metrics['J04']:.6f}.")
            for name in ('A_minus_C', 'B_minus_C', 'B_minus_A'):
                comparison = report['paired_comparisons'][name]
                lines.append(f"- {name} mean delta J0.4 {comparison['mean_fold_delta']['J04']:+.6f}, paired 95% interval {comparison['paired_intervals']['J04']['ci95']}.")
            lines.append('')
        lines += [f"Fresh-fold Swin original advantage is {decision['fresh_swin_original_advantage']:+.6f}.",
                  'Eligibility is decided across all five folds. Only passing ensembles can be selected.', '']
    if 'p2' in analysis:
        decision = analysis['p2']['decision']
        lines += [f"P2 screen promoted loss is {decision['promoted_loss']}. Follow-up required is {decision['followup_required']}.", '']
        for loss, item in analysis['p2']['screen'].items():
            comparison = item['minus_BCE']
            lines.append(f"- {loss} fold original deltas {comparison['per_fold_delta']['original_auc']}, fold J0.4 deltas {comparison['per_fold_delta']['J04']}, mean J0.4 delta {comparison['mean_fold_delta']['J04']:+.6f}.")
        lines += ['', 'A passing loss requires Conv17 confirmation on folds 2, 3, 4. Mean delta J0.4 must reach .003 with at least two strictly positive folds.',
                  'A changed final ensemble needs a new written bounded complete-procedure comparison. No mixed recipe is selected automatically.', '']
    lines += ['Pooled and probability-mean results are secondary. Per-condition values, paired intervals and input hashes are in analysis.json.',
              'No H or test pixels were opened. No final procedure or independent final performance is established.', '']
    return '\n'.join(lines)


def write_outputs(output, analysis, stage):
    output.mkdir(parents=True, exist_ok=True)
    for name in ('p1', 'p2'):
        if name in analysis:
            receipt = dict(analysis['receipt'])
            if name == 'p1':
                receipt['inputs'] = [item for item in receipt['inputs'] if item['loss'] == 'bce']
            value = dict(analysis[name]['decision'], receipt=receipt)
        elif not analysis['complete'] and (stage == 'all' or stage == name):
            value = {'complete': False, 'error': analysis['error'], 'selected': None, 'review_required': True} if name == 'p1' else {
                'complete': False, 'error': analysis['error'], 'promoted_loss': None, 'followup_required': False}
        else:
            continue
        (output / (name + '_decision.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'ANALYSIS.md').write_text(markdown(analysis), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--stage', choices=('p1', 'p2', 'all'), default='all')
    parser.add_argument('--check-sklearn', action='store_true', help='Optional independent local AUC cross-check')
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    root = args.root.resolve()
    raw = args.raw.resolve() if args.raw else (root / 'runs/p1-primary/raw')
    if args.raw is None and not raw.is_dir() and (root / 'results').is_dir():
        raw = root / 'results'
    output = args.output.resolve() if args.output else root / 'runs/p1-primary'
    try:
        fits, conditions, protocol, receipt = load_inputs(root, raw, args.stage)
        receipt.update(bootstrap_draws=args.bootstrap, bootstrap_seed=protocol['analysis']['bootstrap_seed'], numpy=np.__version__)
        analysis = {'complete': True, 'stage': args.stage, 'bootstrap_draws': args.bootstrap,
                    'bootstrap_seed': protocol['analysis']['bootstrap_seed'], 'uncertainty_limits': LIMITS,
                    'aggregation': 'unweighted mean within-fold AUC; pooled secondary', 'receipt': receipt}
        if args.check_sklearn:
            from sklearn.metrics import roc_auc_score
            difference = max(abs(roc_auc_score(fit.y, scores) - auc(fit.y, scores))
                             for fit in fits.values() for scores in fit.probabilities)
            if difference > 1e-12:
                raise ValueError('NumPy AUC differs from independent sklearn AUC')
            analysis['maximum_auc_difference_vs_sklearn'] = float(difference)
        baseline_boots = None
        if args.stage in ('p1', 'all'):
            analysis['p1'], baseline_boots = analyze_p1(fits, conditions, protocol, args.bootstrap)
        if args.stage in ('p2', 'all'):
            analysis['p2'] = analyze_p2(fits, conditions, protocol, args.bootstrap, baseline_boots)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        analysis = {'complete': False, 'stage': args.stage, 'error': f'{type(error).__name__}: {error}'}
        write_outputs(output, analysis, args.stage)
        print('Analysis incomplete. ' + analysis['error'], file=sys.stderr)
        return 2
    write_outputs(output, analysis, args.stage)
    print(json.dumps({name: analysis[name]['decision'] for name in ('p1', 'p2') if name in analysis}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
