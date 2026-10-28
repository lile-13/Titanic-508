import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from src.competition_features import InputEncoder,build_pool
from src.focused_features import FocusedBuilder
from tests.test_train import raw_example


class FocusedTests(unittest.TestCase):
    def setUp(self):
        raw=raw_example();test=raw.iloc[:2].drop(columns="Transported").copy()
        test.PassengerId=["9001_01","9001_02"]
        self.frame,self.test=build_pool(raw,test)

    def test_public_context_target_flip_cannot_change_features(self):
        context=self.frame.copy();changed=context.copy();changed.Transported=~changed.Transported
        for mode in ["compact","location","spend_location","physical"]:
            first=FocusedBuilder(mode).fit(self.frame,context)
            second=FocusedBuilder(mode).fit(changed,changed)
            pd.testing.assert_frame_equal(first.transform(self.test),second.transform(self.test))
            self.assertNotIn("Transported",first.transform(self.frame))
            self.assertNotIn("CabinKey",first.transform(self.frame))

    def test_planet_mapping_fills_only_unknown_and_refuses_conflicting_surnames(self):
        context=self.frame.copy()
        context["Surname"]="Shared";context["HomePlanet"]=context.HomePlanet.astype(object)
        builder=FocusedBuilder().fit(self.frame,context)
        incoming=self.frame.iloc[[0,2]].copy();incoming["Surname"]="Shared"
        incoming["HomePlanet"]=pd.Series([None,"Mars"],index=incoming.index,dtype=object)
        incoming["Deck"]=pd.Series([None,None],index=incoming.index,dtype=object)
        result=builder.transform(incoming)
        self.assertEqual(result.HomePlanet.iloc[0],"__missing__")
        self.assertEqual(result.HomePlanet.iloc[1],"Mars")

    def test_xgboost_native_categories_and_joblib_reload(self):
        from xgboost import XGBClassifier
        builder=FocusedBuilder("physical").fit(self.frame)
        X=builder.transform(self.frame);T=builder.transform(self.test)
        encoder=InputEncoder("lightgbm").fit(X)
        model=XGBClassifier(n_estimators=5,max_depth=2,n_jobs=1,tree_method="hist",enable_categorical=True)
        model.fit(encoder.transform(X),self.frame.Transported.astype(int))
        probability=model.predict_proba(encoder.transform(T))[:,1]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.joblib";joblib.dump((builder,encoder,model),path)
            b,e,m=joblib.load(path)
            np.testing.assert_array_equal(m.predict_proba(e.transform(b.transform(self.test)))[:,1],probability)


if __name__=="__main__":unittest.main()
