import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from src.competition_features import FeatureBuilder, InputEncoder, build_pool, development_splits, PROFILES
from src.optimization import estimator, fit_model
from tests.test_train import raw_example


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.raw = raw_example()
        test = self.raw.iloc[:2].drop(columns="Transported").copy()
        test["PassengerId"] = ["9001_01", "9001_02"]
        self.features, self.test = build_pool(self.raw, test)

    def test_confirmation_fold_never_in_development(self):
        assignments=np.repeat(np.arange(1,6),2)
        for train, valid in development_splits(assignments):
            self.assertFalse((assignments[train]==5).any())
            self.assertFalse((assignments[valid]==5).any())
            self.assertFalse(set(train)&set(valid))

    def test_pool_features_do_not_use_target(self):
        changed=self.raw.copy();changed.Transported=~changed.Transported
        a,b=build_pool(changed,self.test[self.raw.drop(columns="Transported").columns])
        pd.testing.assert_frame_equal(a.drop(columns="Transported"),self.features.drop(columns="Transported"))
        pd.testing.assert_frame_equal(b,self.test)

    def test_profiles_are_numeric_or_explicit_categories(self):
        for feature_set,strategy in PROFILES:
            builder=FeatureBuilder(feature_set,strategy).fit(self.features.iloc[:8])
            X=builder.transform(self.features.iloc[:8]);T=builder.transform(self.test)
            self.assertNotIn("Transported",X)
            self.assertNotIn("PassengerId",X)
            self.assertEqual(list(X),list(T))
            for c in X.select_dtypes(object):
                self.assertTrue(X[c].map(lambda v:isinstance(v,str)).all())
            for family in ["catboost","lightgbm"]:
                encoder=InputEncoder(family).fit(X)
                transformed=encoder.transform(T)
                self.assertEqual(list(transformed),list(X))

    def test_profile_fitting_does_not_use_target(self):
        changed = self.features.copy()
        changed.Transported = ~changed.Transported
        for feature_set, strategy in PROFILES:
            original = FeatureBuilder(feature_set, strategy).fit(self.features.iloc[:8])
            flipped = FeatureBuilder(feature_set, strategy).fit(changed.iloc[:8])
            pd.testing.assert_frame_equal(original.transform(self.features), flipped.transform(changed))
            pd.testing.assert_frame_equal(original.transform(self.test), flipped.transform(self.test))

    def test_both_estimators_and_serialized_predictions(self):
        for family in ["catboost","lightgbm"]:
            builder=FeatureBuilder("rich","native").fit(self.features)
            X=builder.transform(self.features);T=builder.transform(self.test)
            encoder=InputEncoder(family).fit(X);X=encoder.transform(X);T=encoder.transform(T)
            params=dict(iterations=5,depth=2,learning_rate=.1,l2_leaf_reg=3.) if family=="catboost" else dict(iterations=5,num_leaves=3,min_child_samples=1)
            config=dict(family=family,params=params,seed=42)
            model=estimator(config,encoder.cat_columns_,threads=1)
            fit_model(model,family,X,self.raw.Transported.astype(int))
            pred=model.predict_proba(T)[:,1]
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"model.joblib";joblib.dump([builder,encoder,model],path)
                bb,ee,mm=joblib.load(path)
                np.testing.assert_allclose(mm.predict_proba(ee.transform(bb.transform(self.test)))[:,1],pred)


if __name__=="__main__":unittest.main()
