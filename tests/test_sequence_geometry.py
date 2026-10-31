import unittest

import numpy as np
import pandas as pd

from src.sequence_geometry import PublicCabinGeometry, COLUMNS


class SequenceGeometryTest(unittest.TestCase):
    def setUp(self):
        self.public = pd.DataFrame(dict(GroupId=[1, 2, 2, 3], Deck=["G"]*4, Side=["P"]*4,
                                        CabinNum=[3., 5., 5., 7.], Transported=[0,1,0,1]))

    def test_public_counts_and_linear_residuals_ignore_all_labels(self):
        a = PublicCabinGeometry().fit(self.public)
        b = PublicCabinGeometry().fit(self.public.assign(Transported=1-self.public.Transported))
        x, y = a.transform(self.public), b.transform(self.public)
        pd.testing.assert_frame_equal(x,y)
        np.testing.assert_allclose(x.CabinSequenceResidual,0.,atol=1e-14)
        np.testing.assert_array_equal(x.PublicCabinOccupancy,[1,2,2,1])
        np.testing.assert_array_equal(x.PublicCabinNeighborCount10,[4]*4)
        np.testing.assert_array_equal(x.NearestOtherCabinGap,[2.]*4)
        self.assertNotIn("Transported",x)

    def test_unknown_region_and_missing_cabin_have_explicit_fallbacks(self):
        fitted = PublicCabinGeometry().fit(self.public)
        unknown = self.public.iloc[[0]].assign(Deck="A")
        self.assertTrue(fitted.transform(unknown).isna().all().all())
        missing = self.public.iloc[[0]].assign(CabinNum=np.nan)
        actual = fitted.transform(missing)
        self.assertEqual(actual.SequenceExpectedCabin.iloc[0],3.)
        self.assertTrue(actual[COLUMNS[1:]].isna().all().all())

    def test_query_batch_size_does_not_change_public_population_counts(self):
        fitted = PublicCabinGeometry().fit(self.public)
        full = fitted.transform(self.public)
        pd.testing.assert_frame_equal(fitted.transform(self.public.iloc[[1]]),full.iloc[[1]])


if __name__=="__main__":
    unittest.main()
