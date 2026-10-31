import unittest

import numpy as np
import pandas as pd

from src.tabm_models import TabMEncoder


class TabMEncoderTest(unittest.TestCase):
    def test_query_outliers_unknown_categories_and_missing_values_do_not_refit(self):
        train = pd.DataFrame(dict(Age=[10., 20., 30.], Spa=[0., 10., 20.],
                                  Empty=[np.nan]*3, Constant=[1.]*3, Deck=["A", "B", "A"]))
        encoder = TabMEncoder().fit(train)
        before = encoder.medians_.copy()
        query = pd.DataFrame(dict(Age=[np.nan, 1e9], Spa=[np.nan, 1e9], Empty=[99., 99.],
                                  Constant=[2., 2.], Deck=["unseen", "B"]))
        n, c = encoder.transform(query)
        self.assertTrue(np.isfinite(n).all())
        self.assertEqual(n.shape, (2, 2))
        self.assertEqual(c[0, 0], 0)
        self.assertTrue((np.abs(n) <= 8).all())
        pd.testing.assert_series_equal(before, encoder.medians_)
        np.testing.assert_array_equal(encoder.transform(query.iloc[[0]])[0], n[[0]])

    def test_target_is_rejected_and_training_order_does_not_change_encoding(self):
        X = pd.DataFrame(dict(Age=[5., 12., 32., 40.], Deck=["B", "A", "B", "A"]))
        with self.assertRaises(ValueError):
            TabMEncoder().fit(X.assign(Transported=True))
        a = TabMEncoder().fit(X)
        b = TabMEncoder().fit(X.iloc[::-1])
        for x, y in zip(a.transform(X), b.transform(X)):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
