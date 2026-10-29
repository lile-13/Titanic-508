import unittest
import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from src import preprocess as pp
from src.competition_features import build_pool
from src.g_deck_models import g_mask
from src.g_offset_models import GOffsetCalibrator,offset_objective,corrected_probability


class GOffsetTest(unittest.TestCase):
    def test_offset_gradient_matches_finite_difference(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(40,6));beta=rng.normal(size=6)
        offset=rng.normal(size=40);y=rng.integers(0,2,size=40)
        error=check_grad(lambda b:offset_objective(b,X,offset,y,.1)[0],
            lambda b:offset_objective(b,X,offset,y,.1)[1],beta)
        self.assertLess(error,1e-6)

    def test_non_g_labels_do_not_fit_the_correction_and_predictions_are_unchanged(self):
        raw,test=pp.load_raw();features,t=build_pool(raw,test);f=features.iloc[:400]
        public=pd.concat([features.drop(columns=pp.TARGET),t],ignore_index=True)
        y=raw.Transported.to_numpy(int)[:400];p=np.linspace(.1,.9,400);mask=g_mask(f)
        first=GOffsetCalibrator().fit(f,p,y,public)
        altered=y.copy();altered[~mask]=1-altered[~mask]
        second=GOffsetCalibrator().fit(f,p,altered,public)
        np.testing.assert_array_equal(first.coef_,second.coef_)
        pred=corrected_probability(first,f,p,.5)
        np.testing.assert_array_equal(pred[~mask],p[~mask])
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(((pred>=0)&(pred<=1)).all())


if __name__=="__main__":unittest.main()
