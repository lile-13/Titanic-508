import pickle
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.tabpfn_models import from_context


class TabPFNContextTest(unittest.TestCase):
    def test_copy_and_roundtrip_exclude_unpicklable_runtime(self):
        original=SimpleNamespace(builder=None,training_X=pd.DataFrame({"Age":[10.,30.],"Side":["P","S"]}),training_y=np.array([0,1]))
        member=from_context(original)
        member._model=lambda x:x  # A CUDA runtime must not become serialized state.
        restored=pickle.loads(pickle.dumps(member))
        self.assertIsNone(restored._model)
        pd.testing.assert_frame_equal(restored.training_X,original.training_X)
        np.testing.assert_array_equal(restored.training_y,original.training_y)
        original.training_X.loc[0,"Age"]=99.
        original.training_y[0]=1
        self.assertEqual(restored.training_X.loc[0,"Age"],10.)
        self.assertEqual(member.training_X.loc[0,"Age"],10.)
        self.assertEqual(member.training_y[0],0)


if __name__=="__main__":
    unittest.main()
