"""Rebuild difficulty predictions, weights, group isolation and all comparisons."""
import importlib.metadata
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.audit_unlabeled_views import assert_stats
from scripts.hard_example_study import NAME, STAGES, INNER_SEED, ARMS, WEIGHT, read, sha, metrics
from src import preprocess as pp
from src.competition_features import FeatureBuilder, InputEncoder
from src.hard_example_training import predict
from src.optimization import SearchRunner
from src.paths import EXPERIMENTS_DIR, PROJECT_ROOT
from src.train import _write_csv, _write_json


def weight_rebuild(y,p,g,arm):
    w=np.ones(len(y))
    if arm in ["hard","hard_g"]:
        w=1+np.where(y==1,1-p,p)
    if arm in ["g_control","hard_g"]:
        w[g]*=1.5
    return w/w.mean()


def independent_gate(rows,arm):
    if len(rows)!=2:
        return False
    conditions=[r["gain"]>=0 and r["gain_over_control"]>=0 and r["g"]["gain"]>=0
        and r["non_g"]["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows]
    return bool(all(conditions) and sum(r["gain"] for r in rows)/2>=.001
        and sum(r["gain_over_control"] for r in rows)/2>=.0005
        and sum(r["improved_folds"] for r in rows)>=6
        and (arm!="hard_g" or sum(r["g"]["gain"] for r in rows)/2>=.003))


def main():
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,PROJECT_ROOT/"outputs"/NAME,PROJECT_ROOT/"models"/NAME
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    for p,h in protocol["sources"].items():
        assert sha(PROJECT_ROOT/p)==h
    for p,h in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==h
    for p,h in protocol["references"].items():
        assert sha(EXPERIMENTS_DIR/p)==h
    for p,v in protocol["environment"].items():
        assert importlib.metadata.version(p)==v
    cfg=protocol["config"]
    # Label-selected historical cases are read only after all fitting has finished.
    diagnostic_path=EXPERIMENTS_DIR/"accuracy_headroom_20261031/passenger_diagnostics.csv"
    diagnostic=pd.read_csv(diagnostic_path,dtype={"PassengerId":str})
    np.testing.assert_array_equal(diagnostic.PassengerId,old.raw.PassengerId)
    persistent=diagnostic.wrong_repetitions.eq(4).to_numpy()
    assert persistent.sum()==1203
    max_error,reloaded,teacher_count=0.,0,0
    description=[]
    stages={}
    active={"hard","hard_g"}
    for stage,run,seeds in STAGES:
        if stage not in summary["stages"]:
            break
        rows={arm:[] for arm in active}
        stage_active=active.copy()
        repeats=max(len(r) for r in summary["stages"][stage].values())
        for repeat,seed in enumerate(seeds[:repeats],1):
            table=pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            reference=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            np.testing.assert_allclose(table.reference,reference.alternative if stage=="paired" else reference.reference,rtol=0,atol=1e-14)
            for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(old.features,groups=old.features.GroupId),1):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                assert (table.fold.iloc[vi]==fold).all()
                difficulty=np.full(len(ti),np.nan)
                inner_of=np.zeros(len(ti),int)
                for inner,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=INNER_SEED).split(ti,groups=old.features.GroupId.iloc[ti]),1):
                    stem=f"{tag}_teacher_{inner}"
                    mp=models/f"{stem}.joblib"
                    assert sha(mp)==read(report/f"{stem}.json")["model_sha256"]
                    z=np.load(out/f"{stem}.npz")
                    np.testing.assert_array_equal(z["training"],ti[it])
                    np.testing.assert_array_equal(z["validation"],ti[iv])
                    assert not set(old.features.GroupId.iloc[ti[it]])&set(old.features.GroupId.iloc[ti[iv]])
                    model=joblib.load(mp)
                    p=predict(model,old.features.iloc[ti[iv]])
                    np.testing.assert_allclose(p,z["probability"],rtol=0,atol=1e-14)
                    max_error=max(max_error,float(np.max(np.abs(p-z["probability"]))))
                    difficulty[iv]=p
                    inner_of[iv]=inner
                    teacher_count+=1
                dz=np.load(out/f"{tag}_difficulty.npz")
                np.testing.assert_array_equal(dz["training"],ti)
                np.testing.assert_array_equal(dz["validation"],vi)
                np.testing.assert_array_equal(dz["inner_fold"],inner_of)
                np.testing.assert_allclose(dz["probability"],difficulty,rtol=0,atol=1e-14)
                g=old.features.Deck.iloc[ti].eq("G").fillna(False).to_numpy(bool)
                builder=FeatureBuilder(cfg["feature_set"],cfg["strategy"]).fit(old.features.iloc[ti])
                fresh_X=builder.transform(old.features.iloc[ti])
                for arm in ARMS:
                    stem=f"{tag}_{arm}"
                    record=read(report/f"{stem}.json")
                    assert sha(models/f"{stem}.joblib")==record["model_sha256"]
                    z=np.load(out/f"{stem}.npz")
                    np.testing.assert_array_equal(z["training"],ti)
                    np.testing.assert_array_equal(z["validation"],vi)
                    w=weight_rebuild(old.y[ti],difficulty,g,arm)
                    np.testing.assert_allclose(z["weights"],w,rtol=0,atol=1e-15)
                    assert abs(w.mean()-1)<1e-14 and w.max()/w.min()<=3+1e-14
                    model=joblib.load(models/f"{stem}.joblib")
                    pd.testing.assert_frame_equal(model["builder"].transform(old.features.iloc[ti]),fresh_X)
                    assert pp.TARGET not in fresh_X
                    encoder=InputEncoder("catboost").fit(fresh_X)
                    assert model["encoder"].columns_==encoder.columns_
                    assert model["encoder"].cat_columns_==encoder.cat_columns_
                    params=model["model"].get_params()
                    for k,v in cfg["params"].items():
                        assert params[k]==v
                    assert params["random_seed"]==42
                    p=predict(model,old.features.iloc[vi])
                    np.testing.assert_allclose(p,z["probability"],rtol=0,atol=1e-14)
                    np.testing.assert_allclose(table[f"component_{arm}"].iloc[vi],p,rtol=0,atol=1e-14)
                    max_error=max(max_error,float(np.max(np.abs(p-z["probability"]))))
                    reloaded+=1
                print(f"Audited hard-example {tag}",flush=True)
            g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
            for arm in ARMS:
                np.testing.assert_allclose(table[arm],(1-WEIGHT)*table.reference+WEIGHT*table[f"component_{arm}"],rtol=0,atol=1e-14)
            for arm in sorted(stage_active):
                control=table.control if arm=="hard" else table.g_control
                row=dict(repeat=repeat,**metrics(old.y,table.reference.to_numpy(),table[arm].to_numpy(),control.to_numpy(),table.fold.to_numpy(),g))
                saved=summary["stages"][stage][arm][repeat-1]
                assert_stats({k:v for k,v in row.items() if k not in ["g","non_g"]},saved)
                assert_stats(row["g"],saved["g"])
                assert_stats(row["non_g"],saved["non_g"])
                rows[arm].append(row)
                correct_before=(table.reference.to_numpy()>=.5)==old.y
                correct_after=(table[arm].to_numpy()>=.5)==old.y
                description.append(dict(stage=stage,repeat=repeat,arm=arm,persistent_cases=int(persistent.sum()),
                    persistent_corrected=int((persistent&~correct_before&correct_after).sum()),
                    all_corrected=int((~correct_before&correct_after).sum()),all_new_errors=int((correct_before&~correct_after).sum()),
                    g_corrected=row["g"]["corrected"],g_new_errors=row["g"]["new_errors"],
                    non_g_corrected=row["non_g"]["corrected"],non_g_new_errors=row["non_g"]["new_errors"]))
                if row["gain"]<0 or row["gain_over_control"]<0 or row["g"]["gain"]<0 or row["non_g"]["gain"]<0 or row["log_loss"]>row["reference_log_loss"]+.002 or row["improved_folds"]==0:
                    stage_active.remove(arm)
        stages[stage]=rows
        active={arm for arm in stage_active if independent_gate(rows[arm],arm)}
        if not active:
            assert summary["status"]=="no_candidate" and summary["failed_stage"]==stage
            break
    if active:
        assert summary["status"]=="qualified_for_production" and set(summary["qualified_arms"])==active
    else:
        assert not list(out.glob("submission*.csv")) and not (models/"ensemble.joblib").exists()
    log=(PROJECT_ROOT/f"outputs/{NAME}_tests.log").read_text(encoding="utf-8-sig")
    assert re.search(r"Ran 63 tests",log) and re.search(r"\bOK\b",log)
    best=PROJECT_ROOT/"outputs/kaggle_ready_4690/submission_tabicl_mean4690.csv"
    assert sha(best)=="bd2858e0412f37d22ac38040401bfbe3e459a5ab5690f0e4b98924c78aed8ac0"
    _write_csv(pd.DataFrame(description),report/"persistent_error_and_g_diagnostics.csv")
    _write_json(PROJECT_ROOT/f"reports/checks/{NAME}.json",dict(passed=True,gate_passed=bool(active),
        teacher_models_reloaded=teacher_count,weighted_and_control_models_reloaded=reloaded,max_reload_error=max_error,
        training_only_group_crossfit_verified=True,weights_independently_rebuilt=True,tests=63,
        diagnostic_case_source_sha256=sha(diagnostic_path),
        diagnostics_are_postfit_only=True,stages=stages,diagnostics=description))


if __name__=="__main__":
    main()
