import unittest

import numpy as np

from src.hard_example_training import training_weights


class HardWeightsTests(unittest.TestCase):
    def test_bounded_normalized_and_symmetric(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0., 0., 1., 1.])
        g = np.array([False, False, True, True])
        for arm in ["control", "hard", "g_control", "hard_g"]:
            w = training_weights(y, p, g, arm)
            self.assertAlmostEqual(w.mean(), 1.)
            self.assertLessEqual(w.max()/w.min(), 3.)
            np.testing.assert_array_equal(w, training_weights(1-y, 1-p, g, arm))
        np.testing.assert_array_equal(training_weights(y, p, g, "control"), np.ones(4))
        w = training_weights(y, p, np.zeros(4, bool), "hard")
        self.assertEqual(w[1]/w[0], 2.)

    def test_invalid_inputs(self):
        for p in [[np.nan, .5], [-.1, .5], [.5, 1.1]]:
            with self.assertRaises(ValueError):
                training_weights([0, 1], p, [False, True], "hard")
        with self.assertRaises(ValueError):
            training_weights([0, 1], [.1, .2], [True], "hard")


if __name__ == "__main__":
    unittest.main()
