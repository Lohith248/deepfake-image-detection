"""Behavior checks using synthetic images only. CPU, no downloads/training."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from deepfake_core import decode_rgb, enumerate_images, ImageFolder, predict, write_predictions


class PixelModel(torch.nn.Module):
    def forward(self, x): return x[:, 0].mean((1, 2), keepdim=False)[:, None]


class ContractChecks(unittest.TestCase):
    def test_modes_sizes_and_uppercase(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, mode, size, color in [('gray.png','L',(112,112),80), ('alpha.png','RGBA',(320,120),(40,80,120,128)),
                                           ('photo.JPG','RGB',(224,224),(10,40,120)), ('palette.png','P',(40,60),0)]:
                Image.new(mode,size,color).save(root/name)
            for p in root.iterdir():
                array=decode_rgb(p)
                self.assertEqual(array.shape,(224,224,3));self.assertEqual(array.dtype,np.uint8)
            self.assertEqual(len(enumerate_images(root)),4)

    def test_duplicate_and_bad_inputs_are_visible(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            Image.new('RGB',(10,10)).save(root/'same.JPG');Image.new('RGB',(10,10)).save(root/'same.png')
            with self.assertRaisesRegex(ValueError,'Duplicate'):enumerate_images(root)
            (root/'bad.jpg').write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'Cannot decode image bad.jpg'):decode_rgb(root/'bad.jpg')

    def test_predictions_follow_pixels_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            Image.new('RGB',(224,224),(0,0,0)).save(root/'z.png')
            Image.new('RGB',(224,224),(255,255,255)).save(root/'a.png')
            loader=DataLoader(ImageFolder(root),batch_size=1)
            ids,scores=predict([PixelModel()],loader,torch.device('cpu'),{'temperature':1.,'combination':'probability_mean'})
            self.assertEqual(ids,['a','z']);self.assertGreater(scores[0],scores[1])
            ids2,scores2=predict([PixelModel()],DataLoader(ImageFolder(root),batch_size=2),torch.device('cpu'),{'temperature':1.,'combination':'probability_mean'})
            self.assertEqual(ids,ids2);self.assertTrue(np.allclose(scores,scores2,atol=1e-7))

    def test_csv_preserves_double_scores_and_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'nested/out.csv';values=[.9997564634602684,.9997965730219448]
            write_predictions(out,['one','two'],values)
            with out.open(newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual([float(r['confidence']) for r in rows],values)
            with self.assertRaises(ValueError):write_predictions(out,['same','same'],values)
            with self.assertRaises(ValueError):write_predictions(out,['one'],[float('nan')])


if __name__=='__main__':unittest.main(verbosity=2)
