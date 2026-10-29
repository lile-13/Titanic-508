from types import SimpleNamespace
import unittest

import numpy as np

from src.ensemble_weights import reweight_members


class EnsembleWeightTests(unittest.TestCase):
    def test_component_identity_not_list_order_controls_reweighting_and_source_is_preserved(self):
        plan=dict(weights=[.25,.25,.5],members=[dict(config=dict(family=f,feature_set=s,strategy=t))
            for f,s,t in [("catboost","spend","native"),("catboost","rich_names","native"),("lightgbm","base","rule_group")]])
        members=[]
        for weight,m in zip(plan["weights"],plan["members"]):
            c=m["config"]
            for _ in range(3):members.append(dict(weight=weight/3,encoder=SimpleNamespace(family=c["family"]),
                builder=SimpleNamespace(feature_set=c["feature_set"],strategy=c["strategy"])))
        shuffled=[members[i] for i in [7,0,3,8,4,1,5,2,6]]
        before=[m["weight"] for m in shuffled]
        changed=reweight_members(shuffled,plan,[.4,.15,.45])
        np.testing.assert_array_equal([m["weight"] for m in shuffled],before)
        self.assertAlmostEqual(sum(m["weight"] for m in changed if m["builder"].feature_set=="spend"),.4)
        self.assertAlmostEqual(sum(m["weight"] for m in changed if m["builder"].feature_set=="rich_names"),.15)
        with self.assertRaises(ValueError):reweight_members(shuffled,plan,[.4,.15,.5])


if __name__=="__main__":unittest.main()
