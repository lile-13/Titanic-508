import unittest

import numpy as np
import pandas as pd

from src.competition_features import InputEncoder,build_pool
from src.domain_models import ConstrainedBuilder,DeckExperts
from tests.test_train import raw_example


class DomainTests(unittest.TestCase):
    def test_unknown_deck_falls_back_to_global_prediction(self):
        raw=raw_example();raw["Cabin"]=[f"G/{i}/S" for i in range(10)]
        frame,test=build_pool(raw,raw.iloc[:2].drop(columns="Transported"))
        builder=ConstrainedBuilder("compact").fit(frame)
        X=builder.transform(frame);self.assertNotIn("TotalSpend",X)
        encoder=InputEncoder("lightgbm").fit(X);X=encoder.transform(X)
        model=DeckExperts(expert_weight=.5,min_region_size=2).fit(X,raw.Transported.astype(int))
        self.assertIn("G",model.experts_)
        full=model.global_.predict_proba(X)[:,1]
        local=model.experts_["G"].predict_proba(X)[:,1]
        np.testing.assert_allclose(model.predict_proba(X)[:,1],.5*(full+local))
        unknown=X.copy();unknown["Deck"]=pd.Categorical(["unknown"]*len(X),categories=["G","unknown"])
        # LightGBM retains training categories and handles unseen levels as missing.
        np.testing.assert_allclose(model.predict_proba(unknown),model.global_.predict_proba(unknown))


if __name__=="__main__":unittest.main()
