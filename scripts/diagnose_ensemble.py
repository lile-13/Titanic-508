"""冻结模型训练后诊断；不调参，不重新选择候选。"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.competition_features import build_pool
from src import preprocess as pp
from src.paths import EXPERIMENTS_DIR, PROJECT_ROOT
from src.train import _write_csv, _write_json


def diagnose(name):
    directory = EXPERIMENTS_DIR / name
    result = json.loads((directory / "final_summary.json").read_text(encoding="utf-8"))
    if result["status"] != "promoted_candidate":
        return
    bundle = joblib.load(PROJECT_ROOT / "models" / name / "ensemble.joblib")
    raw, test = pp.load_raw()
    features, _ = build_pool(raw, test)
    oof = pd.read_csv(directory / "oof_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
    np.testing.assert_array_equal(oof.PassengerId, raw.PassengerId)
    y = raw.Transported.to_numpy(int)
    # src.optimization.finalize固定先保存第5折，再保存1～4折；按配置核对每块数量。
    per_fold = sum(len(m["seeds"]) for w, m in zip(bundle["plan"]["weights"], bundle["plan"]["members"]) if w > 0)
    assert len(bundle["members"]) == per_fold * 5
    gaps = []
    for block, fold in enumerate([5, 1, 2, 3, 4]):
        members = bundle["members"][block*per_fold:(block+1)*per_fold]
        total_weight = sum(m["weight"] for m in members)
        tr, va = np.flatnonzero(oof.fold.to_numpy() != fold), np.flatnonzero(oof.fold.to_numpy() == fold)
        train_probability, valid_probability = np.zeros(len(tr)), np.zeros(len(va))
        for member in members:
            builder, encoder, model = member["builder"], member["encoder"], member["model"]
            train_probability += member["weight"]/total_weight * model.predict_proba(encoder.transform(builder.transform(features.iloc[tr])))[:, 1]
            valid_probability += member["weight"]/total_weight * model.predict_proba(encoder.transform(builder.transform(features.iloc[va])))[:, 1]
        np.testing.assert_allclose(valid_probability, oof.probability.to_numpy()[va], atol=1e-12, rtol=0)
        train_acc = float(accuracy_score(y[tr], train_probability >= bundle["plan"]["threshold"]))
        valid_acc = float(accuracy_score(y[va], valid_probability >= bundle["plan"]["threshold"]))
        gaps.append(dict(fold=fold,train_rows=len(tr),validation_rows=len(va),train_accuracy=train_acc,
                         validation_accuracy=valid_acc,accuracy_gap=train_acc-valid_acc,
                         train_log_loss=float(log_loss(y[tr], train_probability)),
                         validation_log_loss=float(log_loss(y[va], valid_probability))))
    gap_frame = pd.DataFrame(gaps).sort_values("fold")
    _write_csv(gap_frame, directory / "training_gaps.csv")
    baseline_output = PROJECT_ROOT / "outputs/catboost_baseline_20261018/test_predictions.csv"
    comparison = dict(training_gap_mean=float(gap_frame.accuracy_gap.mean()),
                      training_accuracy_mean=float(gap_frame.train_accuracy.mean()),
                      validation_accuracy_mean=float(gap_frame.validation_accuracy.mean()),
                      saved_fold_blocks_reproduce_oof=True,
                      interpretation="Post-training diagnostic only; no changes to the frozen plan")
    if baseline_output.exists():
        old = pd.read_csv(baseline_output, dtype={"PassengerId": str})
        new = pd.read_csv(PROJECT_ROOT / "outputs" / name / "test_predictions.csv", dtype={"PassengerId": str})
        np.testing.assert_array_equal(old.PassengerId, new.PassengerId)
        comparison.update(test_labels_changed=int(((old.probability >= .5) != new.prediction).sum()),
                          baseline_test_positive_rate=float((old.probability >= .5).mean()),
                          candidate_test_positive_rate=float(new.prediction.mean()),
                          test_probability_correlation=float(np.corrcoef(old.probability,new.probability)[0,1]))
    _write_json(directory / "training_diagnostics.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="optimization_20261025")
    diagnose(parser.parse_args().name)
