"""Describe recurring errors in the actual platform-best model, without fitting."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization import SearchRunner
from src.paths import EXPERIMENTS_DIR
from src.train import _write_csv, _write_json


def main():
    old = SearchRunner()
    source = EXPERIMENTS_DIR/"tabicl_probability_mean_20261030"
    report = EXPERIMENTS_DIR/"incumbent_errors_20261031"
    report.mkdir(parents=True, exist_ok=True)
    tables = [pd.read_csv(source/f"fresh_repeat_{r}_predictions.csv", dtype={"PassengerId":str}, float_precision="round_trip") for r in [1,2]]
    for t in tables:
        np.testing.assert_array_equal(t.PassengerId, old.raw.PassengerId)
        np.testing.assert_array_equal(t.Transported, old.y)
    probabilities = np.stack([t.alternative.to_numpy() for t in tables])
    errors = (probabilities>=.5) != old.y
    d = old.features[["PassengerId", "HomePlanet", "Deck", "CryoSleep", "Age", "GroupSize", "TotalSpend"]].copy()
    d["mean_error"] = errors.mean(axis=0)
    d["wrong_in_both"] = errors.all(axis=0)
    d["mean_probability"] = probabilities.mean(axis=0)
    d["actual"] = old.y
    for c in ["HomePlanet", "Deck", "CryoSleep"]:
        d[c] = d[c].astype("string").fillna("Unknown")
    rows = []
    for columns in [["HomePlanet"], ["Deck"], ["CryoSleep"], ["HomePlanet", "Deck"], ["Deck", "CryoSleep"]]:
        for key, cohort in d.groupby(columns, observed=True):
            if len(cohort) < 100:
                continue
            key = key if isinstance(key, tuple) else (key,)
            rows.append(dict(cohort=" / ".join(f"{k}={v}" for k,v in zip(columns,key)), rows=len(cohort),
                mean_accuracy=float(1-cohort.mean_error.mean()), mean_errors=float(cohort.mean_error.sum()),
                wrong_in_both=int(cohort.wrong_in_both.sum()), mean_probability=float(cohort.mean_probability.mean()),
                actual_rate=float(cohort.actual.mean())))
    _write_csv(pd.DataFrame(rows).sort_values("mean_errors", ascending=False), report/"cohorts.csv")
    _write_csv(d[d.wrong_in_both], report/"recurring_errors.csv")
    _write_json(report/"summary.json", dict(model_run="tabicl_probability_mean_20261030",observed_platform_score=.81178,
        diagnostic_rows=len(old.y), wrong_in_both=int(errors.all(axis=0).sum()), wrong_in_either=int(errors.any(axis=0).sum()),
        sources={f"fresh_repeat_{r}_predictions.csv":hashlib.sha256((source/f"fresh_repeat_{r}_predictions.csv").read_bytes()).hexdigest() for r in [1,2]},
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat="Two historical grouped repetitions of the same labeled passengers. Overlapping descriptive cohorts; no fitting, routing, hypothesis selection or independent performance claim."))
    print("Current incumbent recurring-error diagnosis saved")


if __name__ == "__main__":
    main()
