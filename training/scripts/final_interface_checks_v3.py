"""Judge invocation regressions with synthetic CPU pixels and models only."""
import sys
sys.dont_write_bytecode = True
import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
import deepfake_core_x1 as C

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=Path, default=ROOT/'scripts/final_inference_v3.py')
args, remaining = parser.parse_known_args()
spec = importlib.util.spec_from_file_location('judge_inference_target', args.target)
I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)


class PixelModel(torch.nn.Module):
    def forward(self, x):
        return x.mean((1, 2, 3))


class JudgeInvocationChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.fixture = self.root/'images'
        self.fixture.mkdir()
        Image.new('RGB', (16,16), (0,0,0)).save(self.fixture/'dark.JPG')
        Image.new('RGB', (16,16), (255,255,255)).save(self.fixture/'light.png')
        self.weights = self.root/'synthetic.pt'
        self.weights.write_bytes(b'synthetic fixture; actual tensor loader replaced')
        self.meta = {'candidate':'CT_SS','combination':'logit_mean','threshold':.5,'temperature':1.,'precision':'fp32','provenance':[{'architecture':'convnext_tiny'},{'architecture':'swin_s'}]}

    def tearDown(self):
        self.folder.cleanup()

    def invoke(self, output, extra=()):
        argv = ['inference_script.py','--test_dir',str(self.fixture),'--weights',str(self.weights),
                '--output',str(output),'--device','cpu',*extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys,'argv',argv), patch.object(I.torch.cuda,'is_available',return_value=False), patch.object(I,'load_bundle',return_value=([PixelModel(),PixelModel()],self.meta)), redirect_stdout(stdout), redirect_stderr(stderr):
            I.main()
        return stdout.getvalue(), stderr.getvalue()

    def baseline(self):
        output = self.root/'reference.csv'
        self.invoke(output)
        return output.read_bytes()

    def test_existing_predictions_are_replaced_on_success(self):
        expected = self.baseline()
        output = self.root/'predictions.csv'
        output.write_bytes(b'previous submitted CSV')
        _, stderr = self.invoke(output)
        self.assertEqual(output.read_bytes(), expected)
        self.assertIn('replac', stderr.lower())

    def test_official_batch_size_flag_keeps_identical_scores(self):
        expected = self.baseline()
        output = self.root/'batch.csv'
        _, stderr = self.invoke(output, ['--batch_size','32','--workers','2'])
        self.assertEqual(output.read_bytes(),expected)
        self.assertIn('64',stderr)
        self.assertIn('0',stderr)

    def test_official_threshold_flag_is_accepted(self):
        expected = self.baseline()
        output = self.root/'threshold.csv'
        self.invoke(output,['--threshold','0.5'])
        self.assertEqual(output.read_bytes(),expected)

    def test_compatibility_flags_cannot_change_frozen_procedure(self):
        expected = self.baseline()
        output = self.root/'unchanged.csv'
        _, stderr = self.invoke(output,['--batch_size','1','--workers','8','--threshold','0.9'])
        self.assertEqual(output.read_bytes(),expected)
        self.assertIn('0.5',stderr)

    def test_bad_input_preserves_existing_csv_and_raises(self):
        output = self.root/'predictions.csv'
        output.write_bytes(b'previous submitted CSV')
        (self.fixture/'broken.jpg').write_bytes(b'not a JPEG')
        with self.assertRaises(ValueError):
            self.invoke(output)
        self.assertEqual(output.read_bytes(),b'previous submitted CSV')

    def test_failed_atomic_replace_preserves_existing_bytes(self):
        output = self.root/'predictions.csv'
        output.write_bytes(b'previous submitted CSV')
        with patch.object(C.os,'replace',side_effect=OSError('synthetic replacement failure')):
            with self.assertRaises(OSError):
                C.write_predictions(output,['x'],[.25],.5)
        self.assertEqual(output.read_bytes(),b'previous submitted CSV')
        self.assertEqual(list(self.root.glob('*.tmp-*')),[])

    def test_device_completion_failure_preserves_existing_csv(self):
        output=self.root/'predictions.csv'
        output.write_bytes(b'previous submitted CSV')
        argv=['inference_script.py','--test_dir',str(self.fixture),'--weights',str(self.weights),
              '--output',str(output),'--device','cuda']
        with patch.object(sys,'argv',argv), patch.object(I.torch.cuda,'is_available',return_value=True), patch.object(I,'load_bundle',return_value=([PixelModel(),PixelModel()],self.meta)), patch.object(I,'predict',return_value=(['dark','light'],[.1,.9])), patch.object(I.torch.cuda,'synchronize',side_effect=RuntimeError('synthetic device-completion error')), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'device-completion error'):
                I.main()
        self.assertEqual(output.read_bytes(),b'previous submitted CSV')


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)
