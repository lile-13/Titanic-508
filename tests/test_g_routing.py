import unittest
import numpy as np
import pandas as pd

from src.g_routing import cohort_masks,route_probability,GRoutedFoldModel
from src.probability_calibration import IdentityTransform


class Constant:
    def __init__(self,p):self.p=p
    def predict_proba(self,frame):return np.tile([1-self.p,self.p],(len(frame),1))


class ColdCalibrator:
    def predict(self,frame,probability):
        assert (frame.Deck.eq("G")&frame.CryoSleep.eq(True)).all()
        return np.full(len(frame),.8)


class RoutingTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(Deck=["G","G","G","G","F",None,"G"],
            CryoSleep=pd.array([True,False,False,None,False,True,False],dtype="boolean"),
            Age=[8.,8.,30.,8.,8.,8.,np.nan]))

    def test_masks_disjoint_and_unknown_or_untargeted_rows_unchanged(self):
        f=self.frame();cold,child=cohort_masks(f)
        np.testing.assert_array_equal(np.flatnonzero(cold),[0]);np.testing.assert_array_equal(np.flatnonzero(child),[1])
        ref=np.full(len(f),.4);pc=np.full(len(f),np.nan);pk=pc.copy();pc[cold]=.8;pk[child]=.2
        result=route_probability(f,ref,pc,pk)
        np.testing.assert_allclose(result[:2],[.5,.35])
        np.testing.assert_array_equal(result[~(cold|child)],ref[~(cold|child)])
        with self.assertRaises(ValueError):route_probability(f,ref,pc,pk,child_weight=1.1)

    def test_wrapper_routes_each_branch_and_leaves_other_g_alone(self):
        identity=IdentityTransform()
        def member(p,weight=None):
            m=dict(builder=identity,encoder=identity,model=Constant(p))
            if weight is not None:m["weight"]=weight
            return m
        model=GRoutedFoldModel([member(.4,1.)],ColdCalibrator(),[member(.2)])
        expected=np.array([.5,.35,.4,.4,.4,.4,.4])
        np.testing.assert_allclose(model.predict_proba(self.frame())[:,1],expected)


if __name__=="__main__":unittest.main()
