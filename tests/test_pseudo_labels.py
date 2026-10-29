import unittest

import numpy as np
import pandas as pd

from src.pseudo_labels import augment_training
from src.pseudo_labels import fit_student
from src.competition_features import build_pool
from tests.test_train import raw_example


class PseudoTests(unittest.TestCase):
    def test_soft_target_student_accepts_native_categories_and_predicts_probabilities(self):
        raw=raw_example();test=raw.iloc[:2].drop(columns="Transported").copy()
        test.PassengerId=["9001_01","9001_02"]
        frame,public_test=build_pool(raw,test)
        config=dict(feature_set="spend",confidence=.95,pseudo_weight=.25,
            teacher_config=dict(family="catboost",feature_set="spend",strategy="native",
                params=dict(iterations=4,depth=2,learning_rate=.1,l2_leaf_reg=3.)))
        member=fit_student(config,42,frame,raw.Transported.astype(int),public_test,np.array([.99,.01]))
        X=member["encoder"].transform(member["builder"].transform(public_test))
        p=member["model"].predict_proba(X)
        self.assertEqual(p.shape,(2,2));self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1),1.)
        self.assertEqual(member["pseudo_audit"]["pseudo_rows"],2)

    def test_real_test_labels_are_ignored_and_low_confidence_samples_are_excluded(self):
        train=pd.DataFrame(dict(GroupId=["0001","0002"],Age=[20.,30.],Transported=[False,True]))
        test=pd.DataFrame(dict(GroupId=["0003","0004","0005"],Age=[21.,22.,23.],Transported=[False,False,True]))
        p=np.array([.98,.5,.01])
        frame,y,w,audit=augment_training(train,[0,1],test,p,.95,.25)
        self.assertNotIn("Transported",frame)
        np.testing.assert_array_equal(y,[0.,1.,.98,.01]);np.testing.assert_array_equal(w,[1.,1.,.25,.25])
        self.assertEqual(audit["pseudo_rows"],2)
        changed=test.copy();changed.Transported=~changed.Transported
        other=augment_training(train,[0,1],changed,p,.95,.25)
        pd.testing.assert_frame_equal(frame,other[0]);np.testing.assert_array_equal(y,other[1])
        overlap=test.copy();overlap.loc[0,"GroupId"]="0001"
        with self.assertRaises(ValueError):augment_training(train,[0,1],overlap,p,.95,.25)


if __name__=="__main__":unittest.main()
