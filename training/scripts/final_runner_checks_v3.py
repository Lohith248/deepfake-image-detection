"""CPU-only failure-path checks. No image dataset, training, or CUDA queries."""
import sys
sys.dont_write_bytecode = True
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from PIL import Image
import train_final_v3 as F
import final_inference_v3 as I
from final_inference_v3 import choose_device, validate_metadata
from final_release_v3 import fallback_eligible


class FinalChecks(unittest.TestCase):
    def test_fallback_is_hash_bound_and_only_cuda_unavailable(self):
        allowed = {'cpu_fallback_enabled': True, 'bundle_sha256': 'a'}
        self.assertEqual(choose_device('cuda', True, {}, 'x'), 'cuda')
        self.assertEqual(choose_device('cuda', False, allowed, 'a'), 'cpu')
        self.assertEqual(choose_device('cpu', False, {}, 'a'), 'cpu')
        for policy, digest in [({}, 'a'), (allowed, 'b'), ({'cpu_fallback_enabled': 'true', 'bundle_sha256': 'a'}, 'a')]:
            with self.assertRaises(ValueError):
                choose_device('cuda', False, policy, digest)

    def test_cpu_gate_requires_time_agreement_and_no_label_flips(self):
        cpu = {'timed_out': False, 'returncode': 0, 'wall_seconds': 1200}
        self.assertTrue(fallback_eligible(cpu, [.1, .9], [.1, .9])[0])
        self.assertFalse(fallback_eligible({**cpu, 'wall_seconds': 1200.01}, [.1, .9], [.1, .9])[0])
        self.assertFalse(fallback_eligible(cpu, [.1, .90002], [.1, .9])[0])
        self.assertFalse(fallback_eligible(cpu, [.5], [.499999])[0])
        self.assertFalse(fallback_eligible({**cpu, 'timed_out': True}, None, [.1])[0])

    def test_scoring_contract_rejects_changed_procedure(self):
        valid = {'candidate': 'CT_SS', 'combination': 'logit_mean', 'threshold': .5,
                 'temperature': 1., 'precision': 'fp32', 'provenance': [{'architecture': 'convnext_tiny'},{'architecture': 'swin_s'}]}
        validate_metadata([None, None], valid)
        for key, value in [('temperature', 2.), ('threshold', .4), ('precision', 'fp16'), ('combination', 'probability_mean'),
                             ('provenance', [{'architecture': 'convnext_tiny'}, {'architecture': 'swin_t'}]), ('provenance', None)]:
            with self.assertRaises(ValueError):
                validate_metadata([None, None], {**valid, key: value})

    def test_stage_receipt_detects_any_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            staging = Path(folder) / 'attempt'
            staging.mkdir()
            (staging / 'data.json').write_text('{}')
            final = Path(folder) / 'complete'
            with patch.object(F.os, 'open', return_value=123), patch.object(F.os, 'close'), patch.object(F.os, 'fsync'):
                F.commit_stage(staging, final, {'protocol': 'a'})
            F.validate_stage(final, {'protocol': 'a'})
            (final / 'data.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                F.validate_stage(final, {'protocol': 'a'})

    def test_holdout_interruption_cannot_restart_or_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'results'
            out.mkdir()
            (out / 'holdout_started.json').write_text('{}')
            with patch.object(F.T, 'decode_rgb', side_effect=AssertionError('decoded H')), patch.object(F.T, 'sha256', side_effect=AssertionError('read H')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    F.holdout_readout(root, out, [], {}, {}, {})

    def test_full_pool_cannot_refit_after_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'holdout_started.json').write_text('{}')
            with patch.object(F.T, 'readiness', side_effect=AssertionError('probed GPU')):
                with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                    F.train_member(root, root, 'full_pool', 'convnext_tiny', [], None, {}, {})

    def test_complete_stage_resumes_without_training_and_rejects_renaming(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            staging = root / 'attempt'
            staging.mkdir()
            (staging / 'checkpoint.pt').write_bytes(b'synthetic receipt test, no model')
            final = root / 'all_data_convnext_tiny'
            F.commit_stage(staging, final, {'frozen': True})
            with patch.object(F.T, 'readiness', side_effect=AssertionError('GPU access')), patch.object(F, 'publish'):
                self.assertEqual(F.train_member(root, root, 'all_data', 'convnext_tiny', [], None, {}, {'frozen': True}), final / 'checkpoint.pt')
            changed = root / 'all_data_swin_s'
            final.rename(changed)
            with self.assertRaises(ValueError):
                F.validate_stage(changed, {'frozen': True})

    def test_synthetic_readout_commits_paired_analysis_and_never_reopens(self):
        import deepfake_core_x1 as deepfake_core
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'results'
            out.mkdir()
            images = root / 'holdout-images'
            images.mkdir()
            rows = []
            for i in range(8):
                path = images / (str(i) + '.jpg')
                path.write_bytes(b'synthetic fixture bytes only')
                rows.append({'image_id': str(i), 'group_id': 'g' + str(i), 'label': str(i % 2),
                             'relative_path': 'holdout-images/' + path.name, 'sha256': F.T.sha256(path)})
            checkpoints = {}
            for architecture in F.ARCHITECTURES:
                path = root / (architecture + '.pt')
                path.write_bytes(b'synthetic checkpoint only; loader mocked')
                checkpoints[architecture] = path
            scores = np.array([-2., 1., -.5, 2., -1., .5, .2, 1.5])
            with patch.object(F.T, 'decode_rgb', return_value=np.zeros((2, 2, 3), dtype=np.uint8)), patch.object(F.T, 'build_validation_cache', return_value=[None]*54), patch.object(F.T, 'logits_for', return_value=scores), patch.object(deepfake_core, 'load_bundle', return_value=([object()], {})), patch.object(F.torch.cuda, 'empty_cache'), patch.object(F.T, 'emit_artifact'):
                F.holdout_readout(root, out, rows, checkpoints, {}, {'synthetic': True})
            analysis = json.loads((out / 'holdout/analysis.json').read_text())
            self.assertEqual(analysis['bootstrap_draws'], 2000)
            self.assertEqual(analysis['paired_CT_SS_minus_member_ci95']['swin_s']['J04'], [0., 0.])
            self.assertTrue((out / 'holdout_started.json').is_file())
            with patch.object(F.T, 'decode_rgb', side_effect=AssertionError('reopened readout')), patch.object(F.T, 'logits_for', side_effect=AssertionError('reevaluated model')), patch.object(F.T, 'emit_artifact'):
                F.holdout_readout(root, out, rows, checkpoints, {}, {'synthetic': True})

    def test_dry_run_never_opens_holdout_or_probes_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'experiments').mkdir()
            (root / F.PROTOCOL).write_text(json.dumps({'manifest_hashes': {'full': 'x', 'pool': 'x', 'frozen_split': 'x'}, 'recipe': {'split_manifest_sha256': 'x'}}))
            with patch.object(F, 'check_protocol'), patch.object(F, 'source_identity', return_value={'full_manifest_sha256':'x'}), patch.object(F.T, 'sha256', return_value='x'), patch.object(F.T, 'verify_pool', return_value=[{}]*5950), patch.object(F, 'read_metadata', return_value=([{}]*7000, [{}]*1050)), patch.object(F.T, 'decode_rgb', side_effect=AssertionError('image decode')), patch.object(F.torch.cuda, 'is_available', side_effect=AssertionError('CUDA probe')):
                F.dry_run(root)

    def test_gce_fp32_formula_and_gradient_are_finite(self):
        z = torch.tensor([-1000., -2., 0., 2., 1000.], requires_grad=True)
        y = torch.tensor([0., 1., 1., 0., 1.])
        loss = F.T.training_loss(z, y, 'gce_q07')
        expected = ((1-torch.sigmoid((2*y-1)*z).pow(.7))/.7).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_real_cli_pixels_rows_replay_and_corrupt_file_failure(self):
        class PixelModel(torch.nn.Module):
            def forward(self, x):
                return x.mean((1, 2, 3))
        meta = {'candidate': 'CT_SS', 'combination': 'logit_mean', 'threshold': .5,
                'temperature': 1., 'precision': 'fp32', 'provenance': [{'architecture': 'convnext_tiny'},{'architecture': 'swin_s'}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = root / 'fixture'
            fixture.mkdir()
            Image.new('RGB', (16, 16), (0, 0, 0)).save(fixture / 'dark.JPG')
            Image.new('RGBA', (13, 11), (255, 255, 255, 255)).save(fixture / 'light.png')
            weights = root / 'synthetic.pt'
            weights.write_bytes(b'synthetic fixture only; load mocked')
            common = ['inference_script.py', '--test_dir', str(fixture), '--weights', str(weights), '--device', 'cpu']
            with patch.object(I.torch.cuda, 'is_available', return_value=False), patch.object(I, 'load_bundle', return_value=([PixelModel(), PixelModel()], meta)):
                for index in range(2):
                    with patch.object(sys, 'argv', common + ['--output', str(root / f'run{index}.csv')]):
                        I.main()
                first = (root / 'run0.csv').read_bytes()
                self.assertEqual(first, (root / 'run1.csv').read_bytes())
                with (root / 'run0.csv').open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([r['image_id'] for r in rows], ['dark', 'light'])
                self.assertLess(float(rows[0]['confidence']), .5)
                self.assertGreater(float(rows[1]['confidence']), .5)
                (fixture / 'broken.jpg').write_bytes(b'not an image')
                with patch.object(sys, 'argv', common + ['--output', str(root / 'failed.csv')]):
                    with self.assertRaises(ValueError):
                        I.main()
                self.assertFalse((root / 'failed.csv').exists())
                (fixture / 'broken.jpg').unlink()
                with patch.object(sys, 'argv', common + ['--output', str(root / 'run0.csv'), '--batch_size', '32', '--threshold', '0.5']):
                    I.main()
                self.assertEqual(first, (root / 'run0.csv').read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
