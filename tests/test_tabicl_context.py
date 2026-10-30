import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.tabicl_models import TabICLMember


class IdentityBuilder:
    def transform(self, frame):
        return frame[["x"]].copy()


class ProbeModel:
    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        return self

    def predict_proba(self, X):
        return np.tile([.4, .6], (len(X), 1))


class TabICLContextTest(unittest.TestCase):
    def test_reload_drops_runtime_and_uses_only_saved_training_context(self):
        training = pd.DataFrame({"x": [1., 2.]})
        member = TabICLMember(IdentityBuilder(), training, np.array([0, 1]))
        validation = pd.DataFrame({"x": [100.], "Transported": [False]})
        with patch("src.tabicl_models.make_model", side_effect=ProbeModel) as factory:
            first = member.predict_proba(validation)
            member.predict_proba(validation.assign(Transported=True))
            self.assertEqual(factory.call_count, 1)
            np.testing.assert_array_equal(member._model.y, [0, 1])
            pd.testing.assert_frame_equal(member._model.X, training)
            restored = pickle.loads(pickle.dumps(member))
            self.assertIsNone(restored._model)
            np.testing.assert_array_equal(restored.predict_proba(validation), first)
            self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
