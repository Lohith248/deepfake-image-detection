"""CPU behavior checks for corruption generation and weighted AUC."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from e1_transforms import conditions,transform,stable_seed,training_view
from e1_metrics import auc,sigmoid64,summarize


class E1Checks(unittest.TestCase):
    def test_auc_matches_pair_definition_with_ties_and_weights(self):
        y=np.array([0,1,0,1,1,0]);p=np.array([.1,.5,.5,.9,.5,.9]);w=np.array([2,1,3,2,1,4])
        numerator=sum(w[i]*w[j]*((p[i]>p[j])+.5*(p[i]==p[j])) for i in range(len(y)) for j in range(len(y)) if y[i]==1 and y[j]==0)
        expected=numerator/(w[y==1].sum()*w[y==0].sum())
        self.assertAlmostEqual(auc(y,p,w),expected)
        self.assertAlmostEqual(auc(1-y,p,w),1-expected)

    def test_all_transforms_deterministic_and_do_not_change_source(self):
        source=np.random.default_rng(9).integers(0,256,(224,224,3),dtype=np.uint8);copy=source.copy()
        for c in conditions():
            seed=stable_seed('fixture:'+c['name'])
            a=transform(source,c['operator'],c['severity'],seed);b=transform(source,c['operator'],c['severity'],seed)
            self.assertEqual(a.shape,(224,224,3));self.assertEqual(a.dtype,np.uint8);np.testing.assert_array_equal(a,b)
            self.assertGreater(float(a.std()),1.,c['name']+' unexpectedly erased all source variation')
        np.testing.assert_array_equal(source,copy)
        np.testing.assert_array_equal(training_view(source,'id',3,17,0),training_view(source,'id',3,17,0))

    def test_suite_weights_and_summary(self):
        cs=conditions();self.assertAlmostEqual(sum(c['weight'] for c in cs),1.)
        for family in {c['family'] for c in cs}-{'original','diagnostic'}:
            self.assertAlmostEqual(sum(c['weight'] for c in cs if c['family']==family),1/8)
        y=[0,1,0,1];scores=np.tile([.1,.9,.2,.8],(len(cs),1));result=summarize(y,scores,cs)
        self.assertEqual(result['original_auc'],1);self.assertAlmostEqual(result['robust_auc_proxy'],1)
        self.assertAlmostEqual(result['J04'],1.4)

    def test_sigmoid_extremes_are_finite_and_monotone(self):
        z=np.array([-1000,-20,-1,0,1,20,1000]);p=sigmoid64(z)
        self.assertTrue(np.isfinite(p).all());self.assertTrue((np.diff(p)>=0).all());self.assertEqual(p[3],.5)


if __name__=='__main__':unittest.main(verbosity=2)
