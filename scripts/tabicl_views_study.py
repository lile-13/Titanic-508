"""A single 8-to-32-view change against the actual current best model."""
import gc
import hashlib
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.tabicl_study import compare
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabicl_models import PARAMS, CHECKPOINT_SHA256, CHECKPOINT
from src.tabicl_views import from_context
from src.train import _write_json, _write_csv


NAME = "tabicl_views32_20261031"
PARENT = "tabicl_probability_mean_20261030"
ADDITIONAL = "tabm_20261031"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gate(rows):
    return (len(rows)==2 and all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows)
            and np.mean([r["gain"] for r in rows])>=.001 and sum(r["improved_folds"] for r in rows)>=6)


def main():
    old = SearchRunner()
    report, out, models = EXPERIMENTS_DIR/NAME, PROJECT_ROOT/"outputs"/NAME, PROJECT_ROOT/"models"/NAME
    for d in [report, out, models]:
        d.mkdir(parents=True, exist_ok=True)
    plan = read(old.report/"frozen_plan.json")
    paths = {f"{run}/fresh_repeat_{r}_predictions.csv":sha(EXPERIMENTS_DIR/run/f"fresh_repeat_{r}_predictions.csv")
             for run in [PARENT, ADDITIONAL] for r in [1,2]}
    protocol = dict(name=NAME, params=dict(PARAMS, n_estimators=32, average_logits=False),
        incumbent=PARENT, incumbent_observed_kaggle_score=.81178, external_tree_weight=.5, threshold=.5,
        change="One fixed increase from eight to 32 internal views. Keep every generated view; arithmetic probability aggregation, compact18 features, original context rows/labels, seed42 and checkpoint unchanged. The first eight views need not be the same subset when generator size changes.",
        selection="No weight, threshold, seed, cohort, feature or intermediate-view-count search. Baseline includes the actual eight-view probability mean, not only the weaker old trees.",
        stages=dict(paired="Seeds20261030/20261107 from probability-mean study", additional_paired="Seeds20261031/20261108 from TabM study; use its eight-view/tree baseline, never its TabM alternative"),
        caveat="All four partitions and labels are historical at freeze. This is adaptive historical paired sensitivity, not untouched validation or independent blind evidence.",
        gate="Each stage: both repeats nonnegative, mean gain>=.001, >=6/10 strictly improved folds, logloss deterioration<=.002 each. Stop stage after first negative gain/loss or first zero-improved repeat. Stop experiment at a failed stage. Production only if both stages pass.",
        reference_tables=paths, reference_protocols={run:sha(EXPERIMENTS_DIR/run/"protocol.json") for run in [PARENT,ADDITIONAL]},
        original_plan_sha256=sha(old.report/"frozen_plan.json"), checkpoint_sha256=CHECKPOINT_SHA256,
        data_hashes=read(old.report/"protocol.json")["data_hashes"],
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/tabicl_views_study.py", "src/tabicl_views.py", "src/tabicl_models.py", "src/tabicl_aggregation.py",
            "src/focused_features.py", "src/tabicl_inputs.py", "src/competition_features.py", "src/preprocess.py", "src/optimization.py", "scripts/tabicl_study.py", "features/folds.csv"]},
        environment={p:importlib.metadata.version(p) for p in ["tabicl","torch","numpy","pandas","scikit-learn"]})
    ppfile = report/"protocol.json"
    if ppfile.exists():
        assert read(ppfile)==protocol, "Frozen 32-view experiment changed"
    else:
        _write_json(ppfile, protocol)
    for p,digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==digest
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT)==CHECKPOINT_SHA256
    if (report/"summary.json").exists():
        print("Completed 32-view study; audit rather than retrain", flush=True)
        return
    stages = {}
    for stage, run in [("paired",PARENT), ("additional_paired",ADDITIONAL)]:
        rows = []
        for repeat in [1,2]:
            table = pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv", dtype={"PassengerId":str}, float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported, old.y)
            reference = table.alternative.to_numpy() if stage=="paired" else table.reference.to_numpy()
            component, tree = np.full(len(old.y),np.nan), np.full(len(old.y),np.nan)
            for fold in range(1,6):
                original_tag = f"fresh_repeat_{repeat}_fold_{fold}"
                tag = f"{stage}_repeat_{repeat}_fold_{fold}"
                base_stem = original_tag if stage=="paired" else f"reference_tfm_{original_tag}"
                source = PROJECT_ROOT/"models"/run/f"{base_stem}.joblib"
                source_record = read(EXPERIMENTS_DIR/run/f"{base_stem}.json")
                assert sha(source)==source_record["model_sha256"]
                base_cache = np.load(PROJECT_ROOT/"outputs"/run/f"{base_stem}.npz")
                ti, vi = base_cache["training"], base_cache["validation"]
                assert (table.fold.iloc[vi]==fold).all()
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                if stage=="paired":
                    tree[vi] = table.reference.iloc[vi]
                    np.testing.assert_allclose(reference[vi], .5*tree[vi]+.5*base_cache["probability"], atol=1e-14, rtol=0)
                else:
                    tree[vi] = 0.
                    for j,(weight,cfg) in enumerate(zip(plan["weights"],plan["members"])):
                        for seed in cfg["seeds"]:
                            z = np.load(PROJECT_ROOT/"outputs"/run/f"reference_{original_tag}_component_{j}_seed_{seed}.npz")
                            np.testing.assert_array_equal(z["training"],ti)
                            np.testing.assert_array_equal(z["validation"],vi)
                            tree[vi] += weight/len(cfg["seeds"])*z["probability"]
                    np.testing.assert_allclose(reference[vi], .5*tree[vi]+.5*base_cache["probability"], atol=1e-14, rtol=0)
                cache, mp, rp = out/f"{tag}.npz", models/f"{tag}.joblib", report/f"{tag}.json"
                if cache.exists() and mp.exists() and rp.exists():
                    z = np.load(cache)
                    np.testing.assert_array_equal(z["training"],ti)
                    np.testing.assert_array_equal(z["validation"],vi)
                    assert sha(mp)==read(rp)["model_sha256"] and read(rp)["source_context_sha256"]==sha(source)
                    component[vi] = z["probability"]
                    continue
                original = joblib.load(source)
                np.testing.assert_array_equal(original.training_y,old.y[ti])
                member = from_context(original)
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                p = member.predict_proba(old.features.iloc[vi])[:,1]
                views = sum(len(v) for v in member._model.ensemble_generator_.class_shuffles_.values())
                assert views==32 and member._model.get_params()["average_logits"] is False
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                component[vi] = p
                joblib.dump(member,mp,compress=3)
                np.savez_compressed(cache,training=ti,validation=vi,probability=p)
                _write_json(rp,dict(model_sha256=sha(mp),source_context_sha256=sha(source),views=views,
                    training_rows=len(ti),validation_rows=len(vi),seconds=time.perf_counter()-started,
                    gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
                del member,original
                gc.collect()
                torch.cuda.empty_cache()
                print(f"32-view {tag} completed",flush=True)
            alternative = .5*tree+.5*component
            result = dict(repeat=repeat,**compare(old.y,reference,alternative,table.fold.to_numpy()),
                component_accuracy=float(((component>=.5)==old.y).mean()))
            rows.append(result)
            _write_json(report/f"{stage}_results.json",rows)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=table.fold,
                reference=reference,tree=tree,component=component,alternative=alternative)), report/f"{stage}_repeat_{repeat}_predictions.csv")
            print(f"{stage} repeat {repeat}: net {result['corrected']-result['new_errors']:+d}, {result['improved_folds']}/5 improved",flush=True)
            if result["gain"]<0 or result["log_loss"]>result["reference_log_loss"]+.002 or (repeat==1 and result["improved_folds"]==0):
                break
        stages[stage] = rows
        if not gate(rows):
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None,
                reason="Fixed 32-view increase failed the predeclared gate; no new weights/seeds/view counts selected"))
            return
    _write_json(report/"qualified.json",dict(stages=stages))
    _write_json(report/"summary.json",dict(status="qualified_for_production",stages=stages,kaggle_score=None))


if __name__=="__main__":
    main()
