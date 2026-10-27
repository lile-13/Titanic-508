"""训练回归测试：折映射、原生类别、折外覆盖及保存模型的预测复现。"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import preprocess as pp
from src.train import (TrainingConfig, make_inputs, make_model, make_submission,
                       positive_probability, predict_saved_model, run_training, validate_fold_map)


def raw_example():
    n = 10
    return pd.DataFrame({
        "PassengerId": [f"{g:04d}_{p:02d}" for g in range(1, 6) for p in (1, 2)],
        "HomePlanet": ["Earth", "Europa"] * 5,
        "CryoSleep": [False, True] * 5,
        "Cabin": [f"F/{i * 100}/P" for i in range(n)],
        "Destination": ["TRAPPIST-1e"] * n,
        "Age": [20., 30., np.nan, 30., 20., 30., 20., 30., 20., 30.],
        "VIP": [False] * n,
        **{c: [50., 0.] * 5 for c in pp.SPEND_COLS},
        "Name": [f"Person{i} Family{i // 2}" for i in range(n)],
        "Transported": [False, True] * 5,
    })


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.raw = raw_example()
        self.features = pp.build_features(self.raw)
        self.fold_map = pd.DataFrame({"PassengerId": self.raw.PassengerId,
                                      "fold": np.repeat(np.arange(1, 6), 2)})

    def test_fold_alignment_and_rejection_of_invalid_maps(self):
        shuffled = self.fold_map.sample(frac=1, random_state=0)
        np.testing.assert_array_equal(validate_fold_map(self.features, shuffled), np.repeat(np.arange(1, 6), 2))
        invalid = [self.fold_map.iloc[:-1], pd.concat([self.fold_map, self.fold_map.iloc[:1]])]
        cross_group = self.fold_map.copy()
        cross_group.loc[0, "fold"] = 2
        invalid.append(cross_group)
        for frame in invalid:
            with self.assertRaises(ValueError):
                validate_fold_map(self.features, frame)

    def test_native_catboost_inputs_and_probability(self):
        clean = pp.Preprocessor("rule_group").fit_transform(self.features)
        X = make_inputs(clean, "catboost")
        self.assertEqual(X.shape[1], 30)
        self.assertFalse(any(c in X for c in ["PassengerId", "GroupId", "Transported", "CabinNum"]))
        for col in pp.CAT_FEATURES:
            self.assertTrue(X[col].map(lambda v: isinstance(v, str)).all())
        model = make_model(TrainingConfig(model="catboost", iterations=5, depth=2, threads=1))
        model.fit(X, self.raw.Transported.astype(int))
        probability = positive_probability(model, X)
        self.assertEqual(len(probability), len(X))
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_submission_order_and_boolean_labels(self):
        template = pd.DataFrame({"PassengerId": ["9998_01", "9999_01"], "Transported": [False, False]})
        submission = make_submission(template.PassengerId, [.2, .8], template)
        self.assertEqual(submission.Transported.tolist(), [False, True])
        self.assertTrue(pd.api.types.is_bool_dtype(submission.Transported))
        with self.assertRaises(ValueError):
            make_submission(template.PassengerId.iloc[::-1], [.2, .8], template)
        with self.assertRaises(ValueError):
            make_submission(template.PassengerId, [np.nan, .8], template)

    def test_end_to_end_oof_coverage_and_saved_model_reproduction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            test = self.raw.iloc[:2].drop(columns="Transported").copy()
            test["PassengerId"] = ["9998_01", "9999_01"]
            self.raw.to_csv(data / "train.csv", index=False)
            test.to_csv(data / "test.csv", index=False)
            template = pd.DataFrame({"PassengerId": test.PassengerId, "Transported": [False] * len(test)})
            template.to_csv(data / "sample_submission.csv", index=False)
            folds = root / "folds.csv"
            self.fold_map.to_csv(folds, index=False)
            options = dict(data_dir=data, folds_path=folds, fit_full=True,
                           report_root=root / "reports", output_root=root / "outputs",
                           model_root=root / "models")
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_training(TrainingConfig(), "test_run", **options)
            oof = pd.read_csv(summary["oof_path"])
            self.assertEqual(oof.PassengerId.tolist(), self.raw.PassengerId.tolist())
            self.assertTrue(oof.PassengerId.is_unique)
            self.assertTrue(oof.probability.notna().all())
            self.assertEqual(set(oof.fold), set(range(1, 6)))
            predicted, submission = predict_saved_model(summary["model_bundle"], data_dir=data)
            saved = pd.read_csv(root / "outputs" / "test_run" / "test_predictions.csv")
            np.testing.assert_allclose(predicted.probability, saved.probability, rtol=0, atol=1e-14)
            self.assertEqual(submission.PassengerId.tolist(), test.PassengerId.tolist())
            self.assertEqual(summary["status"], "completed")
            with self.assertRaises(FileExistsError):
                run_training(TrainingConfig(), "test_run", **options)
            test.loc[0, "Age"] = 99
            test.to_csv(data / "test.csv", index=False)
            with self.assertRaises(ValueError):
                predict_saved_model(summary["model_bundle"], data_dir=data)


if __name__ == "__main__":
    unittest.main()
