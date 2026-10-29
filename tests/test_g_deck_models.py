import unittest
import numpy as np
import pandas as pd

from src import preprocess as pp
from src.competition_features import build_pool,InputEncoder
from src.g_deck_models import GDeckBuilder,ROLE_COLUMNS,blend_g,g_mask,GOnlyFoldModel
from src.probability_calibration import IdentityTransform


class ConstantModel:
    def __init__(self,p,g_only=False):self.p=p;self.g_only=g_only
    def predict_proba(self,frame):
        if self.g_only:assert g_mask(frame).all()
        return np.tile([1-self.p,self.p],(len(frame),1))


class GDeckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tr,te=pp.load_raw();f,t=build_pool(tr,te)
        cls.public=pd.concat([f.drop(columns=pp.TARGET),t],ignore_index=True)
        cls.frame=f.iloc[:100].copy()

    def test_role_values_equal_brute_force_and_are_batch_independent(self):
        b=GDeckBuilder().fit(self.frame,self.public)
        all_roles=b.role_features(self.frame)
        for _,row in self.frame.iloc[:20].iterrows():
            other=self.public[self.public.GroupId.eq(row.GroupId)&self.public.PassengerId.ne(row.PassengerId)]
            got=all_roles.loc[row.name]
            self.assertEqual(got.OtherChildCount,other.Age.le(12).sum())
            np.testing.assert_allclose(got.OtherAdultMeanAge,other.Age[other.Age.ge(18)].mean(),equal_nan=True,atol=1e-12)
            np.testing.assert_allclose(got.OtherKnownCryoShare,other.CryoSleep.dropna().astype(float).mean(),equal_nan=True)
        pd.testing.assert_frame_equal(b.transform(self.frame.iloc[[3]]),b.transform(self.frame).iloc[[3]])
        self.assertTrue(set(ROLE_COLUMNS).isdisjoint(GDeckBuilder(False).fit(self.frame,self.public).transform(self.frame)))

    def test_target_invariance_and_other_values_exclude_self(self):
        b=GDeckBuilder().fit(self.frame,self.public);X=b.transform(self.frame)
        altered=self.public.copy();altered[pp.TARGET]=np.arange(len(altered))%2
        query=self.frame.copy();query[pp.TARGET]=~query[pp.TARGET]
        pd.testing.assert_frame_equal(X,GDeckBuilder().fit(query,altered).transform(query))
        changed=self.frame.copy();changed.Age=99.;changed.CryoSleep=False
        cols=["OtherKnownCryoShare","OtherAdultMeanAge","OtherChildCount","OtherAdultCount"]
        pd.testing.assert_frame_equal(b.role_features(self.frame)[cols],b.role_features(changed)[cols])

    def test_new_role_category_obeys_encoder_contract(self):
        X=GDeckBuilder().fit(self.frame,self.public).transform(self.frame)
        self.assertEqual(X.GroupAgeRole.dtype,object)
        encoder=InputEncoder("lightgbm").fit(X)
        self.assertIn("GroupAgeRole",encoder.cat_columns_)
        self.assertIsInstance(encoder.transform(X).GroupAgeRole.dtype,pd.CategoricalDtype)

    def test_blending_never_changes_non_g_even_with_invalid_unused_values(self):
        frame=pd.DataFrame(dict(Deck=["G","F",None,"G"]))
        ref=np.array([.2,.3,.6,.8]);special=np.array([.9,np.nan,np.inf,.1])
        result=blend_g(ref,special,frame,.25)
        np.testing.assert_array_equal(result[~g_mask(frame)],ref[~g_mask(frame)])
        np.testing.assert_allclose(result[g_mask(frame)],[.375,.625])
        for weight in [-.1,1.1]:
            with self.assertRaises(ValueError):blend_g(ref,special,frame,weight)

    def test_fold_wrapper_only_calls_specialist_for_g_and_retains_reference(self):
        identity=IdentityTransform()
        def member(p,weight=None,g_only=False):
            m=dict(builder=identity,encoder=identity,model=ConstantModel(p,g_only))
            if weight is not None:m["weight"]=weight
            return m
        wrapper=GOnlyFoldModel([member(.2,.5),member(.6,.5)],[member(.9,g_only=True)],.5)
        frame=pd.DataFrame(dict(Deck=["G","F",None]))
        np.testing.assert_allclose(wrapper.predict_proba(frame)[:,1],[.65,.4,.4])
        np.testing.assert_allclose(wrapper.predict_proba(frame.iloc[[1,2]])[:,1],[.4,.4])


if __name__=="__main__":unittest.main()
