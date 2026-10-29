import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.probability_calibration import ProbabilityCalibrator,calibration_frame
from src.probability_calibration import CalibratedFoldModel


class CalibrationTests(unittest.TestCase):
    def test_calibrated_fold_model_preserves_predictions_after_model_reload(self):
        from catboost import CatBoostClassifier
        from src.competition_features import FeatureBuilder,InputEncoder,build_pool
        from tests.test_train import raw_example
        raw=raw_example();frame,test=build_pool(raw,raw.iloc[:2].drop(columns="Transported"))
        builder=FeatureBuilder("base","native").fit(frame)
        X=builder.transform(frame);encoder=InputEncoder("catboost").fit(X)
        model=CatBoostClassifier(iterations=4,depth=2,cat_features=encoder.cat_columns_,verbose=False,allow_writing_files=False)
        model.fit(encoder.transform(X),raw.Transported.astype(int))
        probability=model.predict_proba(encoder.transform(X))[:,1]
        calibrator=ProbabilityCalibrator("global",.1).fit(frame,probability,raw.Transported.astype(int))
        combined=CalibratedFoldModel([dict(weight=1.,builder=builder,encoder=encoder,model=model)],calibrator,.5)
        expected=.5*probability+.5*calibrator.predict(frame,probability)
        np.testing.assert_allclose(combined.predict_proba(frame)[:,1],expected)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.joblib";joblib.dump(combined,path)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(frame),combined.predict_proba(frame))

    def test_validation_labels_cannot_change_features_or_predictions(self):
        n=24
        frame=pd.DataFrame(dict(HomePlanet=["Earth","Europa"]*(n//2),Deck=["G","B"]*(n//2),
            Side=["P","S"]*(n//2),CryoSleep=[False,True]*(n//2),GroupSize=np.ones(n),
            Transported=[False,True]*(n//2)))
        p=np.linspace(.05,.95,n);y=np.array([0,1]*(n//2))
        for mode in ["global","context","slope_context"]:
            model=ProbabilityCalibrator(mode,.1).fit(frame,p,y)
            changed=frame.copy();changed.Transported=~changed.Transported
            pd.testing.assert_frame_equal(calibration_frame(frame,p,mode),calibration_frame(changed,p,mode))
            np.testing.assert_array_equal(model.predict(frame,p),model.predict(changed,p))
            unknown=frame.iloc[:2].copy();unknown.HomePlanet="unknown"
            self.assertTrue(np.isfinite(model.predict(unknown,p[:2])).all())


if __name__=="__main__":unittest.main()
