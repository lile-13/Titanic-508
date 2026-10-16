"""预处理回归测试：python -m unittest discover -s tests -v。"""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from src import preprocess as pp
from src.paths import PROJECT_ROOT


def example_raw():
    return pd.DataFrame({
        "PassengerId": ["0001_01", "0001_02", "0002_01", "0003_01", "0004_01", "0005_01"],
        "HomePlanet": ["Europa", None, "Earth", "Earth", "Earth", "Earth"],
        "CryoSleep": [False] * 6,
        "Cabin": ["B/1/S", None, "F/2/P", "F/3/P", "F/4/P", "F/5/P"],
        "Destination": ["PSO J318.5-22", None] + ["TRAPPIST-1e"] * 4,
        "Age": [10.0, np.nan, 20.0, 30.0, 80.0, 90.0],
        "VIP": [False] * 6,
        **{c: [0.0] * 6 for c in pp.SPEND_COLS},
        "Name": ["A Smith", None, "B Jones", "C Jones", "D Lee", "E Lee"],
        "Transported": [False, True, False, True, False, True],
    })


class PreprocessRegressionTests(unittest.TestCase):
    def test_data_path_from_root_and_notebooks(self):
        root = PROJECT_ROOT
        original = Path.cwd()
        try:
            for directory in [root, root / "notebooks", root / "notebooks" / "source"]:
                os.chdir(directory)
                self.assertEqual(pp.find_data_dir(), root / "data" / "raw")
                train, test = pp.load_raw()
                self.assertEqual(train.shape, (8693, 14))
                self.assertEqual(test.shape, (4277, 13))
        finally:
            os.chdir(original)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                pp.load_raw(directory)

    def test_indicators_reach_model_and_exclude_label(self):
        raw = example_raw()
        features = pp.build_features(raw)
        marked = pp.Preprocessor("indicator").fit_transform(features)
        marked["Transported_na"] = True
        X, y = pp.make_model_matrix(marked, target=True)
        expected = [c for c in pp.MISSING_INDICATOR_COLS if raw[c].isna().any()]
        self.assertEqual(set(c for c in X if c.endswith("_na")),
                         {f"{c}_na" for c in expected})
        for c in expected:
            np.testing.assert_array_equal(X[f"{c}_na"], raw[c].isna().astype(int))
        self.assertFalse(any("Transported" in c for c in X))
        np.testing.assert_array_equal(y, raw.Transported.astype(int))
        simple = pp.make_model_matrix(pp.Preprocessor("simple").fit_transform(features))
        self.assertGreater(X.shape[1], simple.shape[1])

    def test_only_consistent_group_categories_are_copied(self):
        features = pp.build_features(example_raw())
        clean = pp.Preprocessor("rule_group").fit_transform(features)
        self.assertEqual(clean.loc[1, "HomePlanet"], "Europa")
        self.assertEqual(clean.loc[1, "Side"], "S")
        self.assertEqual(clean.loc[1, "Destination"], "TRAPPIST-1e")
        self.assertEqual(clean.loc[1, "Deck"], "F")

    def test_fold_imputation_uses_training_rows_only(self):
        features = pp.build_features(example_raw())
        Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(
            features, [0, 2, 3], [1, 4, 5], strategy="simple")
        self.assertEqual(Xva.loc[0, "Age"], 20.0)
        full = pp.Preprocessor("simple").fit_transform(features)
        self.assertEqual(full.loc[1, "Age"], 30.0)
        self.assertEqual(list(Xtr), list(Xva))
        self.assertEqual(len(ytr), 3)
        self.assertEqual(len(yva), 3)
        pd.testing.assert_frame_equal(features, pp.build_features(example_raw()))

    def test_combined_counts_labels_and_row_order(self):
        raw = example_raw()
        train, test = pp.build_features_combined(raw.iloc[:3], raw.iloc[3:].drop(columns=pp.TARGET))
        self.assertEqual(train.loc[2, "SurnameSize"], 2)
        self.assertEqual(test.loc[0, "SurnameSize"], 2)
        self.assertEqual(train.PassengerId.tolist(), raw.PassengerId.iloc[:3].tolist())
        self.assertEqual(test.PassengerId.tolist(), raw.PassengerId.iloc[3:].tolist())
        self.assertNotIn(pp.TARGET, test)

    def test_real_data_folds_and_training_integration(self):
        raw, raw_test = pp.load_raw()
        features, test = pp.build_features_combined(raw, raw_test)
        assignments = pd.read_csv(PROJECT_ROOT / "features" / "folds.csv",
                                  dtype={"PassengerId": str, "GroupId": str})
        fold_of = features.PassengerId.map(assignments.set_index("PassengerId").fold)
        self.assertFalse(fold_of.isna().any())
        self.assertTrue((assignments.groupby("GroupId").fold.nunique() == 1).all())
        folds = pp.group_folds(features)
        for i, (tr, va) in enumerate(folds, 1):
            self.assertFalse(set(features.iloc[tr].GroupId) & set(features.iloc[va].GroupId))
            self.assertEqual(set(features.iloc[va].PassengerId),
                             set(assignments.loc[assignments.fold == i, "PassengerId"]))
        tr, va = folds[0]
        Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(features, tr, va)
        for X in [Xtr, Xva]:
            self.assertTrue(np.isfinite(X.to_numpy(float)).all())
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(Xtr, ytr)
        self.assertEqual(len(model.predict(Xva)), len(yva))


if __name__ == "__main__":
    unittest.main(verbosity=2)
