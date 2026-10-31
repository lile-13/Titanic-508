"""Rebuild target-free coordinates and independently reload both model arms."""
import hashlib
import importlib.metadata
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.sequence_geometry_study import NAME, PARENT, ADDITIONAL, WEIGHT, gate
from scripts.audit_unlabeled_views import stats
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.sequence_geometry import SequenceGeometryBuilder, COLUMNS
from src.train import _write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    old = SearchRunner()
    report,out,models = EXPERIMENTS_DIR/NAME,PROJECT_ROOT/"outputs"/NAME,PROJECT_ROOT/"models"/NAME
    protocol,summary = read(report/"protocol.json"),read(report/"summary.json")
    for p,digest in protocol["sources"].items():
        assert sha(PROJECT_ROOT/p)==digest
    for p,digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==digest
    for p,version in protocol["environment"].items():
        assert importlib.metadata.version(p)==version
    for p,digest in protocol["references"].items():
        assert sha(EXPERIMENTS_DIR/p)==digest
    for run,digest in protocol["reference_audits"].items():
        path = PROJECT_ROOT/f"reports/checks/{run}.json"
        assert sha(path)==digest and read(path)["passed"]
    assert sha(old.report/"frozen_plan.json")==protocol["original_plan_sha256"]
    public = pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public
    reloaded, maximum, outcomes = 0,0.,{}
    for stage,rows in summary["stages"].items():
        run = PARENT if stage=="paired" else ADDITIONAL
        measured = []
        for row in rows:
            repeat = row["repeat"]
            table = pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            parent = pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            reference = parent.alternative if stage=="paired" else parent.reference
            np.testing.assert_allclose(table.reference,reference,atol=1e-14,rtol=0)
            seed = protocol["stages"][stage][repeat-1]
            for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(old.features,groups=old.features.GroupId),1):
                assert (table.fold.iloc[vi]==fold).all()
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                arms = {}
                for enabled in [False,True]:
                    arm = "geometry" if enabled else "control"
                    tag = f"{stage}_repeat_{repeat}_fold_{fold}_{arm}"
                    record = read(report/f"{tag}.json")
                    assert sha(models/f"{tag}.joblib")==record["model_sha256"]
                    member = joblib.load(models/f"{tag}.joblib")
                    assert member["builder"].enabled==enabled
                    rebuilt = SequenceGeometryBuilder(enabled).fit(old.features.iloc[ti],public)
                    pd.testing.assert_frame_equal(rebuilt.transform(old.features.iloc[ti]),member["builder"].transform(old.features.iloc[ti]))
                    expected,actual = rebuilt.transform(old.features.iloc[vi]),member["builder"].transform(old.features.iloc[vi])
                    pd.testing.assert_frame_equal(expected,actual)
                    pd.testing.assert_frame_equal(member["builder"].transform(old.features.iloc[vi[:1]]),actual.iloc[:1])
                    assert list(actual)==record["columns"] and pp.TARGET not in actual
                    modelparams = member["model"].get_params()
                    for key,value in protocol["estimator_config"]["params"].items():
                        assert modelparams["n_estimators" if key=="iterations" else key]==value
                    assert modelparams["random_state"]==42
                    cache = np.load(out/f"{tag}.npz")
                    np.testing.assert_array_equal(cache["training"],ti)
                    np.testing.assert_array_equal(cache["validation"],vi)
                    p = member["model"].predict_proba(member["encoder"].transform(actual))[:,1]
                    np.testing.assert_allclose(p,cache["probability"],atol=1e-14,rtol=0)
                    np.testing.assert_allclose(p,table[f"{arm}_component"].iloc[vi],atol=1e-14,rtol=0)
                    maximum = max(maximum,float(np.max(np.abs(p-cache["probability"]))))
                    reloaded += 1
                    arms[enabled] = actual
                pd.testing.assert_frame_equal(arms[False],arms[True].drop(columns=COLUMNS))
            for col,component in [("control","control_component"),("alternative","geometry_component")]:
                np.testing.assert_allclose(table[col],(1-WEIGHT)*table.reference+WEIGHT*table[component],atol=1e-14,rtol=0)
            recomputed = dict(repeat=repeat,**stats(old.y,table.reference.to_numpy(),table.alternative.to_numpy(),table.fold.to_numpy()),
                control_accuracy=float(((table.control.to_numpy()>=.5)==old.y).mean()),
                feature_gain_over_control=float(((table.alternative.to_numpy()>=.5)==old.y).mean()-((table.control.to_numpy()>=.5)==old.y).mean()),
                standalone_control_accuracy=float(((table.control_component.to_numpy()>=.5)==old.y).mean()),
                standalone_geometry_accuracy=float(((table.geometry_component.to_numpy()>=.5)==old.y).mean()))
            for key,value in recomputed.items():
                np.testing.assert_allclose(value,row[key],atol=1e-14,rtol=0)
            measured.append(recomputed)
        outcomes[stage] = measured
    qualified = all(gate(outcomes.get(s,[])) for s in ["paired","additional_paired"])
    assert qualified==(summary["status"]=="qualified_for_production")
    if not qualified:
        assert not (models/"ensemble.joblib").exists() and not list(out.glob("submission*.csv"))
    logpath = PROJECT_ROOT/"outputs/sequence_geometry_20261031_tests.log"
    log = logpath.read_text(encoding="utf-8",errors="replace")
    match = re.search(r"Ran (\d+) tests",log)
    assert match and re.search(r"\nOK\s*$",log)
    ledger = read(EXPERIMENTS_DIR/"kaggle_submissions.json")
    assert ledger["current_platform_best"]["model_run"]==PARENT and ledger["current_platform_best"]["score"]==.81178
    assert sha(PROJECT_ROOT/ledger["current_platform_best"]["submission"])=="bd2858e0412f37d22ac38040401bfbe3e459a5ab5690f0e4b98924c78aed8ac0"
    check = dict(passed=True,models_reloaded=reloaded,max_reload_error=maximum,source_data_environment_verified=True,
        maps_rebuilt_without_labels=True,query_batch_invariance=True,control_matrix_matches=True,group_isolation=True,
        gates_recomputed=True,gate_passed=bool(qualified),reference_is_actual_best=True,
        tests_passed=int(match.group(1)),test_log_sha256=sha(logpath),outcomes=outcomes,platform_score=.81178,kaggle_score=None)
    _write_json(PROJECT_ROOT/f"reports/checks/{NAME}.json",check)
    print(json.dumps(check,indent=2))


if __name__=="__main__":
    main()
