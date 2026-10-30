import unittest

import numpy as np
import pandas as pd

from src.competition_features import build_pool, InputEncoder
from src import preprocess as pp
from src.unlabeled_views import UnlabeledViewBuilder


class UnlabeledViewsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, test = pp.load_raw()
        cls.train, cls.test = build_pool(train, test)
        cls.public = pd.concat([cls.train.drop(columns=pp.TARGET), cls.test], ignore_index=True)

    def test_labels_do_not_change_features_and_batching_is_stable(self):
        training = self.train.iloc[:80]
        query = self.train.iloc[[1, 4, 10, 30]]
        altered = training.assign(Transported=~training.Transported)
        context_with_labels = self.public.assign(Transported=True)
        for view in ["names", "coordinates"]:
            first = UnlabeledViewBuilder(view).fit(training, self.public)
            second = UnlabeledViewBuilder(view).fit(altered, context_with_labels)
            expected = first.transform(query)
            pd.testing.assert_frame_equal(expected, second.transform(query))
            individual = pd.concat([first.transform(query.iloc[[i]]) for i in range(len(query))])
            pd.testing.assert_frame_equal(expected, individual)
            self.assertNotIn(pp.TARGET, expected)

    def test_name_counts_use_one_public_pool_and_categories_remain_object(self):
        builder = UnlabeledViewBuilder("names").fit(self.train.iloc[:60], self.public)
        query = self.train.iloc[[1]].copy()
        name = str(query.FirstName.iloc[0]).lower()
        expected = int(self.public.FirstName.astype("string").str.lower().eq(name).sum())
        X = builder.transform(query)
        self.assertAlmostEqual(X.FirstNamePublicLogCount.iloc[0], np.log1p(expected))
        self.assertIn("FirstSuffix3", InputEncoder("catboost").fit(X).cat_columns_)
        query["FirstName"] = pd.Series(["never_seen_token"], index=query.index, dtype=object)
        self.assertEqual(builder.transform(query).FirstNamePublicLogCount.iloc[0], 0)

    def test_coordinate_neighbors_exclude_own_group_and_other_regions(self):
        frame = self.train.iloc[:5].copy()
        frame["GroupId"] = ["a", "a", "b", "c", "d"]
        frame["GroupIdNum"] = [1., 1., 2., 3., 4.]
        frame["Deck"] = pd.Series(["G", "G", "G", "F", "G"], index=frame.index, dtype=object)
        frame["Side"] = pd.Series(["P", "P", "P", "P", "S"], index=frame.index, dtype=object)
        frame["CabinNum"] = [10., 10., 11., 10., 10.]
        frame["CabinRelative"] = .1
        X = UnlabeledViewBuilder("coordinates").fit(frame, frame).transform(frame)
        self.assertAlmostEqual(X.OtherGroupsWithin10CabinsLogCount.iloc[0], np.log(2))
        self.assertEqual(X.OtherGroupsWithin10CabinsLogCount.iloc[3], 0)
        query = frame.iloc[[0]].copy()
        query["CabinNum"] = np.nan
        self.assertTrue(UnlabeledViewBuilder("coordinates").fit(frame, frame).transform(query).CabinRankWithinDeckSide.isna().all())


if __name__ == "__main__":
    unittest.main()
