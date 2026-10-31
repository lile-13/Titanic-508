"""Describe four existing OOF repetitions; never fit or route a predictor."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.optimization import SearchRunner
from src.paths import EXPERIMENTS_DIR, PROJECT_ROOT
from src.train import _write_csv, _write_json


def main():
    old = SearchRunner()
    report = EXPERIMENTS_DIR / "accuracy_headroom_20261031"
    report.mkdir(parents=True, exist_ok=True)
    probabilities, sources = [], {}
    for run, column in [("tabicl_probability_mean_20261030", "alternative"),
                        ("tabm_20261031", "reference")]:
        for repeat in [1, 2]:
            path = EXPERIMENTS_DIR / run / f"fresh_repeat_{repeat}_predictions.csv"
            table = pd.read_csv(path, dtype={"PassengerId": str}, float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported, old.y)
            assert table.groupby(old.features.GroupId).fold.nunique().max() == 1
            probabilities.append(table[column].to_numpy())
            sources[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "column": column}
    p = np.stack(probabilities)
    errors = (p >= .5) != old.y
    d = old.features
    mean_p = p.mean(axis=0)
    cohorts = {
        "all": np.ones(len(d), bool),
        "wrong_all_four": errors.all(axis=0),
        "wrong_one_to_three": errors.any(axis=0) & ~errors.all(axis=0),
        "right_all_four": ~errors.any(axis=0),
        "G_CryoTrue": (d.Deck.eq("G") & d.CryoSleep.eq(True)).fillna(False).to_numpy(bool),
        "complete_raw_features": d.RawMissingCount.eq(0).to_numpy(bool),
        "missing_raw_features": d.RawMissingCount.gt(0).to_numpy(bool),
    }
    rows = []
    for name, mask in cohorts.items():
        rows.append(dict(cohort=name, rows=int(mask.sum()),
            mean_repeat_accuracy=float(1-errors[:, mask].mean()),
            four_prediction_mean_accuracy=float(((mean_p[mask] >= .5) == old.y[mask]).mean()),
            wrong_all_four=int(errors[:, mask].all(axis=0).sum()),
            actual_rate=float(old.y[mask].mean()), mean_probability=float(mean_p[mask].mean()),
            auc_of_mean=float(roc_auc_score(old.y[mask], mean_p[mask])),
            name_known_rate=float(d.Name[mask].notna().mean()),
            mean_raw_missing=float(d.RawMissingCount[mask].mean())))
    _write_csv(pd.DataFrame(rows), report / "cohorts.csv")
    detail = pd.DataFrame(dict(PassengerId=old.raw.PassengerId, Transported=old.y,
                              wrong_repetitions=errors.sum(axis=0), mean_probability=mean_p))
    for i in range(4):
        detail[f"probability_{i+1}"] = p[i]
    _write_csv(detail, report / "passenger_diagnostics.csv")
    _write_json(report / "summary.json", dict(rows=len(old.y), repetitions=4,
        error_count_histogram=np.bincount(errors.sum(axis=0), minlength=5).tolist(),
        train_groups=int(d.GroupId.nunique()), test_groups=int(old.test_features.GroupId.nunique()),
        shared_train_test_groups=len(set(d.GroupId) & set(old.test_features.GroupId)),
        sources=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=["Four historical partitions of the same passengers, not four independent datasets.",
                 "Wrong/right cohorts are selected using labels and cannot be used as deployable routing rules.",
                 "Mean OOF probabilities are descriptive; no new ensemble or submission is fitted.",
                 "Persistent mistakes neither prove label noise nor estimate a Bayes accuracy ceiling."]))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
