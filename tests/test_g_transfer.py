import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src import preprocess as pp
from src.competition_features import build_pool
from src.g_deck_models import g_mask
from src.g_transfer_models import fit_member


class Recorder:
    def fit(self,X,y,sample_weight):
        self.X=X;self.y=y;self.weight=sample_weight
        return self


class GTransferTest(unittest.TestCase):
    def test_fit_uses_allowed_rows_with_feature_based_weights_and_no_target_column(self):
        raw,test=pp.load_raw();features,t=build_pool(raw,test)
        frame=features.iloc[:400]
        public=pd.concat([features.drop(columns=pp.TARGET),t],ignore_index=True)
        recorder=Recorder()
        with patch("lightgbm.LGBMClassifier",return_value=recorder):
            m=fit_member(dict(family="lightgbm",roles=True),42,frame,raw.Transported.to_numpy(int)[:400],public)
        self.assertIs(m["model"],recorder)
        self.assertEqual(len(recorder.X),400)
        self.assertNotIn(pp.TARGET,recorder.X)
        np.testing.assert_array_equal(recorder.y,raw.Transported.to_numpy(int)[:400])
        np.testing.assert_array_equal(recorder.weight,np.where(g_mask(frame),2.,1.))


if __name__=="__main__":unittest.main()
