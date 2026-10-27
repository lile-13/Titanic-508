"""补齐成员4固定LightGBM的模型与测试概率，不改历史Notebook或建议提交。"""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src import preprocess as pp
from src.paths import EXPERIMENTS_DIR, PROJECT_ROOT
from src.train import _data_hashes, _write_csv, _write_json


def predict(bundle_path):
    bundle = joblib.load(bundle_path)
    assert bundle["format"] == "member4_lightgbm_v1"
    if bundle["data_hashes"] != _data_hashes(pp.find_data_dir()):
        raise ValueError("原始数据发生变化，需要重新训练")
    raw, test = pp.load_raw()
    _, test_features = pp.build_features_combined(raw, test)
    clean = bundle["preprocessor"].transform(test_features)
    X = pp.make_model_matrix(clean).reindex(columns=bundle["columns"], fill_value=0)
    probability = bundle["model"].predict_proba(X)[:, 1]
    submission = pd.DataFrame({"PassengerId": test.PassengerId, "Transported": probability > .5})
    predictions = pd.DataFrame({"PassengerId": test.PassengerId, "probability": probability,
                                "prediction": probability > .5})
    return predictions, submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="member4_lightgbm_export_20261025")
    parser.add_argument("--predict-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or PROJECT_ROOT / "outputs" / args.name
    if output.exists():
        raise FileExistsError("输出目录已存在，请指定新的 --output-dir")
    if args.predict_bundle:
        predictions, submission = predict(args.predict_bundle)
    else:
        model_dir = PROJECT_ROOT / "models" / args.name
        if model_dir.exists():
            raise FileExistsError("模型目录已存在，请使用新的 --name")
        raw, test = pp.load_raw()
        features, test_features = pp.build_features_combined(raw, test)
        preprocessor = pp.Preprocessor("rule_group").fit(features)
        X, y = pp.make_model_matrix(preprocessor.transform(features), target=True)
        T = pp.make_model_matrix(preprocessor.transform(test_features)).reindex(columns=X.columns, fill_value=0)
        model = LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15,
                              random_state=42, verbosity=-1, n_jobs=2)
        model.fit(X, y)
        probability = model.predict_proba(T)[:, 1]
        assert np.isfinite(probability).all()
        expected = pd.read_csv(EXPERIMENTS_DIR / "member4_suggested_submission.csv", dtype={"PassengerId": str})
        original = pd.DataFrame({"PassengerId": test.PassengerId, "Transported": probability > .5})
        pd.testing.assert_frame_equal(original, expected)
        bundle = dict(format="member4_lightgbm_v1", model=model, preprocessor=preprocessor,
                      columns=list(X), data_hashes=_data_hashes(pp.find_data_dir()),
                      params=model.get_params(), threshold=.5, comparison="strictly_greater")
        model_dir.mkdir(parents=True, exist_ok=False)
        bundle_path = model_dir / "bundle.joblib"
        joblib.dump(bundle, bundle_path, compress=3)
        predictions, submission = predict(bundle_path)
        np.testing.assert_allclose(predictions.probability, probability, atol=1e-14, rtol=0)
        pd.testing.assert_frame_equal(submission, original)
        (EXPERIMENTS_DIR / args.name).mkdir(parents=True, exist_ok=True)
        _write_json(EXPERIMENTS_DIR / args.name / "summary.json",
                    dict(model="LightGBM", feature_count=len(X.columns), test_rows=len(test),
                         params=model.get_params(), data_hashes=bundle["data_hashes"],
                         notebook_sha256=hashlib.sha256((PROJECT_ROOT / "notebooks/source/05_对照模型与误差分析.py").read_bytes()).hexdigest(),
                         matches_original_suggested_submission=True, saved_model_predictions_reproduced=True,
                         max_probability_reproduction_error=float(np.max(np.abs(predictions.probability-probability))),
                         bundle=str(bundle_path), output=str(output), submitted_to_kaggle=False))
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(predictions, output / "test_predictions.csv")
    _write_csv(submission, output / "submission.csv")
    print(f"已保存并复现成员4的 {len(submission)} 行LightGBM预测：{output}")


if __name__ == "__main__":
    main()
