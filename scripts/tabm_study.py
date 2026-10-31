"""One fixed TabM addition, compared against the actual 0.81178 incumbent."""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from scripts.tabicl_study import compare
from src import preprocess as pp
from src.frozen_reference import fit_reference
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabicl_aggregation import fit_probability_mean
from src.tabm_models import fit_member, PARAMS
from src.train import _write_json, _write_csv


NAME = "tabm_20261031"
PARENT = "tabicl_probability_mean_20261030"
FRESH_SEEDS = [20261031, 20261108]
WEIGHT = .25


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def passes(rows):
    return (len(rows) == 2 and all(r["gain"] >= 0 and r["log_loss"] <= r["reference_log_loss"]+.002 for r in rows)
            and np.mean([r["gain"] for r in rows]) >= .001 and sum(r["improved_folds"] for r in rows) >= 6)


def main():
    old = SearchRunner()
    report, output, models = EXPERIMENTS_DIR/NAME, PROJECT_ROOT/"outputs"/NAME, PROJECT_ROOT/"models"/NAME
    for d in [report, output, models]:
        d.mkdir(parents=True, exist_ok=True)
    parent = EXPERIMENTS_DIR/PARENT
    protocol = dict(name=NAME, params=PARAMS, input="compact18; log1p consumption; training median/IQR; 16 quantile bins; PLE16",
        author_source="https://github.com/yandex-research/tabm", optimizer="AdamW; mean of 32 independent BCE losses; mean probabilities for inference",
        weight=WEIGHT, threshold=.5, incumbent=PARENT, incumbent_observed_score=.81178,
        selection="One predeclared model/input/seed/weight. Inner GroupShuffleSplit only selects epoch, then refit full outer training. No outer-validation early stopping, threshold or weight search.",
        paired_splits="Yesterday's two fresh group partitions, seeds 20261030/20261107, now historical. Baseline is 0.5 original tree + 0.5 arithmetic TabICL, not old tree alone.",
        fresh_seeds=FRESH_SEEDS,
        gate="Each stage: both repeats nonnegative, mean gain >=.001, >=6/10 strict fold gains, logloss deterioration <=.002 each. Stop after first negative/loss/zero-improved repeat. Fresh stage runs only if paired stage passes. No changed settings or replacement seeds.",
        caveat="All stages use known historical labeled rows; new splits are sensitivity checks, not independent blind data. Final production is permitted only after both stages pass.",
        parent_protocol_sha256=sha(parent/"protocol.json"),
        reference_files={f"fresh_repeat_{r}_predictions.csv":sha(parent/f"fresh_repeat_{r}_predictions.csv") for r in [1,2]},
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"],
        source_hashes={p:sha(PROJECT_ROOT/p) for p in ["scripts/tabm_study.py", "src/tabm_models.py", "requirements-tabm.txt",
            "src/tabicl_inputs.py", "src/focused_features.py", "src/competition_features.py", "src/preprocess.py", "src/optimization.py",
            "src/tabicl_aggregation.py", "src/tabicl_models.py", "src/frozen_reference.py", "scripts/tabicl_study.py"]},
        environment={p:importlib.metadata.version(p) for p in ["tabm", "rtdl_num_embeddings", "torch", "numpy", "pandas", "scikit-learn", "catboost", "lightgbm"]},
        installed_sources={p:sha(Path(__import__(p).__file__)) for p in ["tabm", "rtdl_num_embeddings"]})
    path = report/"protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == protocol, "Frozen TabM study changed"
    else:
        _write_json(path, protocol)
    for p, digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p) == digest
    if (report/"summary.json").exists():
        print("Completed TabM study; audit rather than retrain", flush=True)
        return
    public = pd.concat([old.features.drop(columns=pp.TARGET), old.test_features], ignore_index=True)
    plan = json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))

    def fitted(ti, vi, tag):
        cache, mp, rp = output/f"{tag}.npz", models/f"{tag}.joblib", report/f"{tag}.json"
        if cache.exists() and mp.exists() and rp.exists():
            z = np.load(cache)
            np.testing.assert_array_equal(z["training"], ti)
            np.testing.assert_array_equal(z["validation"], vi)
            assert sha(mp) == json.loads(rp.read_text(encoding="utf-8"))["model_sha256"]
            return z["probability"]
        assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
        print(f"TabM {tag}: nested epoch selection and full outer refit", flush=True)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        member = fit_member(old.features.iloc[ti], old.y[ti], public)
        p = member.predict_proba(old.features.iloc[vi])[:,1]
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        joblib.dump(member, mp, compress=3)
        np.savez_compressed(cache, training=ti, validation=vi, probability=p)
        _write_json(rp, dict(model_sha256=sha(mp), training_rows=len(ti), validation_rows=len(vi),
            selected_epoch=member.selection["selected_epoch"], inner_best_loss=member.selection["best_inner_log_loss"],
            active_numeric=member.encoder.active_, categorical=member.encoder.categorical_,
            seconds=time.perf_counter()-started, gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
        del member
        gc.collect()
        torch.cuda.empty_cache()
        print(f"TabM {tag}: completed", flush=True)
        return p

    stages = {}
    for stage in ["paired", "fresh"]:
        rows = []
        for repeat in [1,2]:
            reference, p = np.full(len(old.y), np.nan), np.full(len(old.y), np.nan)
            folds = np.zeros(len(old.y), int)
            if stage == "paired":
                table = pd.read_csv(parent/f"fresh_repeat_{repeat}_predictions.csv", dtype={"PassengerId":str}, float_precision="round_trip")
                np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
                np.testing.assert_array_equal(table.Transported, old.y)
                split = []
                for fold in range(1,6):
                    z = np.load(PROJECT_ROOT/f"outputs/{PARENT}/fresh_repeat_{repeat}_fold_{fold}.npz")
                    split.append((z["training"], z["validation"]))
                reference[:] = table.alternative.to_numpy()
            else:
                split = list(GroupKFold(5, shuffle=True, random_state=FRESH_SEEDS[repeat-1]).split(old.features, groups=old.features.GroupId))
            for fold, (ti,vi) in enumerate(split,1):
                tag = f"{stage}_repeat_{repeat}_fold_{fold}"
                if stage == "fresh":
                    q = fit_reference(old, plan, ti, vi, tag, output, models, report)
                    cm, cp = models/f"reference_tfm_{tag}.joblib", output/f"reference_tfm_{tag}.npz"
                    if cm.exists() and cp.exists():
                        z = np.load(cp)
                        np.testing.assert_array_equal(z["training"],ti)
                        np.testing.assert_array_equal(z["validation"],vi)
                        t = z["probability"]
                        assert sha(cm) == json.loads((report/f"reference_tfm_{tag}.json").read_text())["model_sha256"]
                    else:
                        m = fit_probability_mean(old.features.iloc[ti], old.y[ti], public)
                        t = m.predict_proba(old.features.iloc[vi])[:,1]
                        joblib.dump(m, cm, compress=3)
                        np.savez_compressed(cp, training=ti, validation=vi, probability=t)
                        _write_json(report/f"reference_tfm_{tag}.json", dict(model_sha256=sha(cm)))
                        del m
                        gc.collect()
                        torch.cuda.empty_cache()
                    reference[vi] = .5*q+.5*t
                p[vi] = fitted(ti,vi,tag)
                folds[vi] = fold
            alternative = (1-WEIGHT)*reference+WEIGHT*p
            r = dict(repeat=repeat, **compare(old.y, reference, alternative, folds),
                     standalone_accuracy=float(((p>=.5)==old.y).mean()))
            rows.append(r)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId, Transported=old.y, fold=folds,
                reference=reference, tabm=p, alternative=alternative)), report/f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report/f"{stage}_results.json", rows)
            print(f"{stage} repeat {repeat}: {r['reference_accuracy']:.6f} -> {r['accuracy']:.6f}, net {r['corrected']-r['new_errors']:+d}, {r['improved_folds']}/5 improved", flush=True)
            if r["gain"]<0 or r["log_loss"]>r["reference_log_loss"]+.002 or (repeat==1 and r["improved_folds"]==0):
                break
        stages[stage] = rows
        if not passes(rows):
            _write_json(report/"summary.json", dict(status="no_candidate", failed_stage=stage, stages=stages,
                reason="Fixed TabM addition failed the predeclared gate; no post-result tuning", kaggle_score=None))
            return
    _write_json(report/"qualified.json", dict(stages=stages))
    _write_json(report/"summary.json", dict(status="qualified_for_production", stages=stages, kaggle_score=None))


if __name__ == "__main__":
    main()
