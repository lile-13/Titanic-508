"""核对真实实验、冻结协议及保存模型：python -m scripts.verify_optimization。"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src import preprocess as pp
from src.optimization import predict_ensemble
from src.paths import EXPERIMENTS_DIR, FEATURES_DIR, PROJECT_ROOT
from src.train import _data_hashes, _write_json, validate_fold_map


def verify(name):
    report = EXPERIMENTS_DIR / name
    protocol = json.loads((report / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((report / "final_summary.json").read_text(encoding="utf-8"))
    plan = json.loads((report / "frozen_plan.json").read_text(encoding="utf-8"))
    assert result["plan"] == plan
    raw, test = pp.load_raw()
    assert protocol["data_hashes"] == _data_hashes(pp.find_data_dir())
    for relative, digest in protocol["sources"].items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest
    dependencies = json.loads((report / "dependency_hashes.json").read_text(encoding="utf-8"))
    for relative, digest in dependencies.items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest
    folds = pd.read_csv(FEATURES_DIR / "folds.csv", dtype={"PassengerId": str, "GroupId": str})
    fold_of = validate_fold_map(raw.assign(GroupId=raw.PassengerId.str.split("_").str[0]), folds)
    count = 0
    for path in report.glob("trial_*.json"):
        trial = json.loads(path.read_text(encoding="utf-8"))
        assert 5 not in trial["folds"]
        probability = np.load(PROJECT_ROOT / "outputs" / name / "search" / f"{trial['trial']}.npz")["probability"]
        assert np.isnan(probability[fold_of == 5]).all()
        scored = np.isin(fold_of, trial["folds"])
        assert np.isfinite(probability[scored]).all()
        assert ((probability[scored] >= 0) & (probability[scored] <= 1)).all()
        scores = [accuracy_score(raw.Transported[fold_of == f], probability[fold_of == f] >= .5)
                  for f in trial["folds"]]
        losses = [log_loss(raw.Transported[fold_of == f], probability[fold_of == f])
                  for f in trial["folds"]]
        np.testing.assert_allclose(scores, trial["fold_scores"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.mean(scores), trial["accuracy"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.mean(losses), trial["log_loss"], atol=1e-14, rtol=0)
        count += 1
    checks = dict(name=name, status=result["status"], trials_checked=count,
                  protocol_sources_match=True, data_hashes_match=True,
                  dependency_hashes_match=True, search_metrics_match=True,
                  confirmation_not_scored_in_search=True, frozen_plan_matches=True)
    if result["status"] == "promoted_candidate":
        output = PROJECT_ROOT / "outputs" / name
        saved = pd.read_csv(output / "test_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
        submitted = pd.read_csv(output / "submission.csv", dtype={"PassengerId": str})
        oof = pd.read_csv(report / "oof_predictions.csv", dtype={"PassengerId": str, "GroupId": str}, float_precision="round_trip")
        assert oof.PassengerId.is_unique and len(oof) == len(raw)
        np.testing.assert_array_equal(oof.PassengerId, raw.PassengerId)
        np.testing.assert_array_equal(oof.fold, fold_of)
        np.testing.assert_array_equal(oof.Transported, raw.Transported)
        assert np.isfinite(oof.probability).all() and oof.probability.between(0, 1).all()
        np.testing.assert_array_equal(oof.prediction, oof.probability >= plan["threshold"])
        np.testing.assert_allclose(accuracy_score(oof.Transported, oof.prediction), result["oof_accuracy"], atol=1e-14)
        np.testing.assert_allclose(log_loss(oof.Transported, oof.probability), result["oof_log_loss"], atol=1e-14)
        fold_metrics = pd.read_csv(report / "fold_metrics.csv")
        for metric in fold_metrics.itertuples():
            mask = oof.fold == metric.fold
            np.testing.assert_allclose(accuracy_score(oof.Transported[mask], oof.prediction[mask]),
                                       metric.accuracy, atol=1e-14, rtol=0)
            np.testing.assert_allclose(log_loss(oof.Transported[mask], oof.probability[mask]),
                                       metric.log_loss, atol=1e-14, rtol=0)
        np.testing.assert_allclose(fold_metrics.accuracy.mean(), result["selected_oof_mean"], atol=1e-14)
        confirmation = oof.loc[oof.fold == 5]
        np.testing.assert_allclose(accuracy_score(confirmation.Transported, confirmation.prediction),
                                   result["confirmation"]["accuracy"], atol=1e-14)
        assert list(submitted) == ["PassengerId", "Transported"] and submitted.Transported.dtype == bool
        assert len(saved) == len(test) == len(submitted) and submitted.PassengerId.is_unique
        np.testing.assert_array_equal(saved.PassengerId, test.PassengerId)
        np.testing.assert_array_equal(submitted.PassengerId, test.PassengerId)
        bundle_path = PROJECT_ROOT / "models" / name / "ensemble.joblib"
        bundle = joblib.load(bundle_path)
        weight_sum = sum(member["weight"] for member in bundle["members"])
        np.testing.assert_allclose(weight_sum, 1, atol=1e-14)
        assert bundle["plan"] == plan and bundle["data_hashes"] == protocol["data_hashes"]
        probability, reproduced = predict_ensemble(bundle_path)
        np.testing.assert_allclose(probability, saved.probability, atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(reproduced.reset_index(drop=True), submitted.reset_index(drop=True))
        np.testing.assert_array_equal(submitted.Transported, saved.probability >= plan["threshold"])
        checks.update(oof_rows=len(oof), submission_rows=len(submitted), model_count=len(bundle["members"]),
                      weight_sum=weight_sum, max_probability_reproduction_error=float(np.max(np.abs(probability-saved.probability))),
                      submission_reproduced=True, oof_metrics_match=True,
                      submission_sha256=hashlib.sha256((output / "submission.csv").read_bytes()).hexdigest())
    checks["passed"] = True
    _write_json(PROJECT_ROOT / "reports" / "checks" / f"{name}.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="optimization_20261025")
    verify(parser.parse_args().name)


if __name__ == "__main__":
    main()
