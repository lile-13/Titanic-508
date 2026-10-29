import unittest
import numpy as np
import pandas as pd

from src.accuracy_threshold import AccuracyThreshold


class ThresholdTest(unittest.TestCase):
    def test_training_cutoff_maps_to_half_with_shrinkage_and_roundtrip(self):
        p=np.repeat([.2,.4,.6,.8],20);y=np.repeat([0,1,1,1],20)
        frame=pd.DataFrame(dict(GroupId=np.arange(len(p))))
        raw=AccuracyThreshold(shrinkage=1.).fit(frame,p,y)
        shrunk=AccuracyThreshold(shrinkage=.5).fit(frame,p,y)
        self.assertLess(raw.effective_threshold_,.5)
        self.assertTrue(raw.effective_threshold_<shrunk.effective_threshold_<.5)
        boundary=pd.DataFrame(dict(GroupId=[999]))
        self.assertAlmostEqual(shrunk.predict(boundary,[shrunk.effective_threshold_])[0],.5)
        identity=AccuracyThreshold(shrinkage=0.).fit(frame,p,y)
        np.testing.assert_allclose(identity.predict(frame,p),p,atol=1e-14)
        np.testing.assert_allclose(raw.predict(frame,p),AccuracyThreshold(shrinkage=1.).fit(frame,p,y).predict(frame,p))

    def test_invalid_probability_and_target_rejected(self):
        frame=pd.DataFrame(dict(GroupId=[1,2]))
        for p,y in [([np.nan,.4],[0,1]),([-.1,.4],[0,1]),([.4],[0,1]),([.4,.5],[0,2])]:
            with self.assertRaises(ValueError):AccuracyThreshold().fit(frame,p,y)


if __name__=="__main__":unittest.main()
