import unittest

import numpy as np
import pandas as pd

from src.diversity_features import DiversityEncoder


class DiversityTests(unittest.TestCase):
    def test_unknown_categories_and_extremes_do_not_refit_training_statistics(self):
        train=pd.DataFrame(dict(Age=[10.,20.,30.],FoodCourt=[0.,10.,100.],HomePlanet=pd.Series(["Earth","Mars","Earth"],dtype=object)))
        encoder=DiversityEncoder().fit(train)
        before=encoder.transform(train)
        unknown=pd.DataFrame(dict(Age=[np.nan,1e10],FoodCourt=[np.nan,1e20],HomePlanet=["unseen","unseen"]))
        result=encoder.transform(unknown)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(np.max(np.abs(result)),8.)
        np.testing.assert_array_equal(encoder.transform(train),before)
        np.testing.assert_array_equal(result[:,-2:],np.zeros((2,2)))


if __name__=="__main__":unittest.main()
