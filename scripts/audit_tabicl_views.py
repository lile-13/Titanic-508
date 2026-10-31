"""Reload view-count contexts and independently reconstruct every paired gate."""
import gc
import hashlib
import importlib.metadata
import json
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from scripts.tabicl_views_study import NAME, PARENT, ADDITIONAL, gate
from scripts.tabicl_study import compare
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabicl_models import PARAMS, CHECKPOINT, CHECKPOINT_SHA256
from src.train import _write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = SearchRunner()
    report, out, models = EXPERIMENTS_DIR/NAME, PROJECT_ROOT/"outputs"/NAME, PROJECT_ROOT/"models"/NAME
    protocol, summary = read(report/"protocol.json"), read(report/"summary.json")
    for p,digest in protocol["sources"].items():
        assert sha(PROJECT_ROOT/p)==digest
    for p,digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==digest
    for p,version in protocol["environment"].items():
        assert importlib.metadata.version(p)==version
    for run,digest in protocol["reference_protocols"].items():
        assert sha(EXPERIMENTS_DIR/run/"protocol.json")==digest
    for p,digest in protocol["reference_tables"].items():
        assert sha(EXPERIMENTS_DIR/p)==digest
    assert sha(old.report/"frozen_plan.json")==protocol["original_plan_sha256"]
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT)==CHECKPOINT_SHA256
    plan = read(old.report/"frozen_plan.json")
    total, baseline_total, max_error = 0, 0, 0.
    outcomes, deltas = {}, []
    for stage,rows in summary["stages"].items():
        run = PARENT if stage=="paired" else ADDITIONAL
        seedbase = [20261030,20261107] if stage=="paired" else [20261031,20261108]
        measured = []
        for row in rows:
            repeat = row["repeat"]
            table = pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv", dtype={"PassengerId":str}, float_precision="round_trip")
            source_table = pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv", dtype={"PassengerId":str}, float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            expected_reference = source_table.alternative if stage=="paired" else source_table.reference
            np.testing.assert_allclose(table.reference,expected_reference,atol=1e-14,rtol=0)
            splitter = GroupKFold(5,shuffle=True,random_state=seedbase[repeat-1]).split(old.features,groups=old.features.GroupId)
            for fold,(ti,vi) in enumerate(splitter,1):
                original_tag = f"fresh_repeat_{repeat}_fold_{fold}"
                base_stem = original_tag if stage=="paired" else f"reference_tfm_{original_tag}"
                tag = f"{stage}_repeat_{repeat}_fold_{fold}"
                record = read(report/f"{tag}.json")
                source = PROJECT_ROOT/"models"/run/f"{base_stem}.joblib"
                assert sha(source)==record["source_context_sha256"]==read(EXPERIMENTS_DIR/run/f"{base_stem}.json")["model_sha256"]
                original = joblib.load(source)
                np.testing.assert_array_equal(original.training_y,old.y[ti])
                base_cache = np.load(PROJECT_ROOT/"outputs"/run/f"{base_stem}.npz")
                np.testing.assert_array_equal(base_cache["training"],ti)
                np.testing.assert_array_equal(base_cache["validation"],vi)
                p8 = original.predict_proba(old.features.iloc[vi])[:,1]
                np.testing.assert_allclose(p8,base_cache["probability"],atol=2e-6,rtol=0)
                assert original._model.get_params()["average_logits"] is False
                assert sum(len(v) for v in original._model.ensemble_generator_.class_shuffles_.values())==8
                original._model = None
                gc.collect()
                torch.cuda.empty_cache()
                baseline_total += 1
                if stage=="paired":
                    tree = source_table.reference.iloc[vi].to_numpy()
                else:
                    tree = np.zeros(len(vi))
                    for j,(weight,component) in enumerate(zip(plan["weights"],plan["members"])):
                        for seed in component["seeds"]:
                            cache = np.load(PROJECT_ROOT/"outputs"/run/f"reference_{original_tag}_component_{j}_seed_{seed}.npz")
                            np.testing.assert_array_equal(cache["training"],ti)
                            np.testing.assert_array_equal(cache["validation"],vi)
                            tree += weight/len(component["seeds"])*cache["probability"]
                np.testing.assert_allclose(tree,table.tree.iloc[vi],atol=1e-14,rtol=0)
                np.testing.assert_allclose(.5*tree+.5*base_cache["probability"],table.reference.iloc[vi],atol=1e-14,rtol=0)
                assert sha(models/f"{tag}.joblib")==record["model_sha256"]
                member = joblib.load(models/f"{tag}.joblib")
                assert member._model is None
                pd.testing.assert_frame_equal(original.training_X,member.training_X)
                pd.testing.assert_frame_equal(original.builder.transform(old.features.iloc[vi]),member.builder.transform(old.features.iloc[vi]))
                np.testing.assert_array_equal(member.training_y,old.y[ti])
                cache = np.load(out/f"{tag}.npz")
                np.testing.assert_array_equal(cache["training"],ti)
                np.testing.assert_array_equal(cache["validation"],vi)
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                p32 = member.predict_proba(old.features.iloc[vi])[:,1]
                for key,value in {**PARAMS,"n_estimators":32,"average_logits":False}.items():
                    assert member._model.get_params()[key]==value
                assert sum(len(v) for v in member._model.ensemble_generator_.class_shuffles_.values())==32
                np.testing.assert_allclose(p32,cache["probability"],atol=2e-6,rtol=0)
                np.testing.assert_allclose(p32,table.component.iloc[vi],atol=2e-6,rtol=0)
                np.testing.assert_array_equal(p32>=.5,cache["probability"]>=.5)
                max_error = max(max_error,float(np.max(np.abs(p32-cache["probability"]))))
                assert (table.fold.iloc[vi]==fold).all()
                total += 1
                del member,original
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Audited eight/32-view pair {tag}",flush=True)
            np.testing.assert_allclose(table.alternative,.5*table.tree+.5*table.component,atol=1e-14,rtol=0)
            current = dict(repeat=repeat,**compare(old.y,table.reference.to_numpy(),table.alternative.to_numpy(),table.fold.to_numpy()),
                           component_accuracy=float(((table.component.to_numpy()>=.5)==old.y).mean()))
            for key,value in current.items():
                np.testing.assert_allclose(value,row[key],atol=1e-14,rtol=0)
            measured.append(current)
            if stage=="additional_paired":
                deltas.append(((table.alternative.to_numpy()>=.5)==old.y).astype(float)-((table.reference.to_numpy()>=.5)==old.y))
        outcomes[stage] = measured
    passed = all(gate(outcomes.get(stage,[])) for stage in ["paired","additional_paired"])
    assert passed==(summary["status"]=="qualified_for_production")
    if not passed:
        assert not (models/"ensemble.joblib").exists() and not list(out.glob("submission*.csv"))
    uncertainty = None
    if len(deltas)==2:
        delta = np.mean(deltas,axis=0)
        codes,groups = pd.factorize(old.features.GroupId)
        sums,sizes = np.bincount(codes,weights=delta),np.bincount(codes)
        rng,boot = np.random.default_rng(21481031),[]
        for _ in range(20):
            draw = rng.integers(0,len(groups),size=(100,len(groups)))
            boot.extend((sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)).tolist())
        uncertainty = dict(mean_gain=float(delta.mean()),group_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
            caveat="Conditional fixed-prediction group resampling; historical passengers and adaptive selection, not independent holdout or a promotion gate")
    logpath = PROJECT_ROOT/"outputs/tabicl_views32_20261031_tests.log"
    log = logpath.read_text(encoding="utf-8",errors="replace")
    match = re.search(r"Ran (\d+) tests",log)
    assert match and re.search(r"\nOK\s*$",log)
    ledger = read(EXPERIMENTS_DIR/"kaggle_submissions.json")
    assert ledger["current_platform_best"]["model_run"]==PARENT and ledger["current_platform_best"]["score"]==.81178
    assert sha(PROJECT_ROOT/ledger["current_platform_best"]["submission"])=="bd2858e0412f37d22ac38040401bfbe3e459a5ab5690f0e4b98924c78aed8ac0"
    audit = dict(passed=True,contexts_32_reloaded=total,contexts_8_reloaded=baseline_total,
        max_reload_error=max_error,exact_same_training_contexts=True,all_views_retained=True,
        group_isolation=True,source_data_environment_verified=True,reference_is_actual_best=True,
        gates_recomputed=True,gate_passed=bool(passed),tests_passed=int(match.group(1)),
        test_log_sha256=sha(logpath),conditional_uncertainty=uncertainty,outcomes=outcomes,
        platform_score=.81178,kaggle_score=None)
    _write_json(PROJECT_ROOT/f"reports/checks/{NAME}.json",audit)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
