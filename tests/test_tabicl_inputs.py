import unittest

import numpy as np
import pandas as pd

from src.competition_features import build_pool
from src.focused_features import FocusedBuilder
from src.tabicl_inputs import TabICLInputBuilder, VIEWS, EXTRA, fit_tfm
from tests.test_train import raw_example


class TabICLInputsTest(unittest.TestCase):
    def setUp(self):
        raw = raw_example()
        test = raw.iloc[:2].drop(columns="Transported").copy()
        test.PassengerId = ["9001_01", "9001_02"]
        self.frame, self.test = build_pool(raw, test)
        self.public = pd.concat([self.frame.drop(columns="Transported"), self.test], ignore_index=True)

    def test_compact_control_is_exact_and_removing_sequence_changes_one_column(self):
        expected = FocusedBuilder("compact", True).fit(self.frame, self.public).transform(self.test)
        actual = TabICLInputBuilder("compact").fit(self.frame, self.public).transform(self.test)
        pd.testing.assert_frame_equal(actual, expected)
        reduced = TabICLInputBuilder("compact_no_sequence").fit(self.frame, self.public).transform(self.test)
        pd.testing.assert_frame_equal(reduced, expected.drop(columns="GroupIdNum"))
        self.assertEqual(len(actual.columns), 18)

    def test_expansion_uses_existing_public_aggregates_and_preserves_the_control(self):
        X = TabICLInputBuilder("expanded").fit(self.frame, self.public).transform(self.test)
        base = TabICLInputBuilder("compact").fit(self.frame, self.public).transform(self.test)
        pd.testing.assert_frame_equal(X[list(base)], base)
        self.assertEqual(len(X.columns), 34)
        for column in EXTRA:
            np.testing.assert_array_equal(X[column], self.test[column])
        np.testing.assert_allclose(X.log_Spa, np.log1p(base.Spa), equal_nan=True)

    def test_training_label_flip_does_not_change_inputs_and_context_is_copied(self):
        altered = self.frame.assign(Transported=~self.frame.Transported)
        for view in VIEWS:
            a = TabICLInputBuilder(view).fit(self.frame, self.public).transform(self.test)
            b = TabICLInputBuilder(view).fit(altered, self.public.assign(Transported=True)).transform(self.test)
            pd.testing.assert_frame_equal(a, b)
            self.assertNotIn("Transported", a)
        y = self.frame.Transported.to_numpy(dtype=int)
        m = fit_tfm("expanded", self.frame, y, self.public)
        before = m.training_y.copy()
        y[:] = 1-y
        np.testing.assert_array_equal(m.training_y, before)

    def test_missing_side_uses_safe_public_group_value_even_for_one_query(self):
        query = self.test.iloc[[0]].copy()
        context = self.public.copy()
        context["GroupId"] = context.GroupId.astype(object)
        context.loc[context.PassengerId.eq(query.PassengerId.iloc[0]), "Side"] = None
        query["Side"] = pd.Series([None], index=query.index, dtype=object)
        other = context[context.GroupId.eq(query.GroupId.iloc[0])].Side.dropna()
        self.assertEqual(other.nunique(), 1)
        result = TabICLInputBuilder("expanded").fit(self.frame, context).transform(query)
        self.assertEqual(result.Side.iloc[0], str(other.iloc[0]))


if __name__ == "__main__":
    unittest.main()
