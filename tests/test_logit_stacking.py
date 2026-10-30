import unittest

import numpy as np
from scipy.optimize import check_grad

from src.logit_stacking import objective, LogitStacker


class LogitStackingTest(unittest.TestCase):
    def test_analytic_gradient(self):
        rng = np.random.default_rng(42)
        X, offset = rng.normal(size=(50, 2)), rng.normal(size=50)
        y = rng.integers(0, 2, 50)
        beta = np.array([.1, .6])
        error = check_grad(lambda b: objective(b, X, offset, y, .01)[0],
                           lambda b: objective(b, X, offset, y, .01)[1], beta)
        self.assertLess(error, 1e-6)

    def test_constrained_weight_and_reference_only_control(self):
        rng = np.random.default_rng(2026)
        y = rng.integers(0, 2, 100)
        reference = rng.uniform(.1, .9, 100)
        tfm = np.where(y == 1, .9, .1)
        stacker = LogitStacker().fit(reference, tfm, y)
        self.assertGreaterEqual(stacker.weight_, 0)
        self.assertLessEqual(stacker.weight_, 1)
        self.assertTrue(np.isfinite(stacker.predict([0, 1], [1, 0])).all())
        control = LogitStacker(tfm_enabled=False).fit(reference, tfm, y)
        self.assertEqual(control.weight_, 0)
        np.testing.assert_array_equal(control.predict(reference, tfm), control.predict(reference, 1-tfm))


if __name__ == "__main__":
    unittest.main()
