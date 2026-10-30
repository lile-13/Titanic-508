"""Audit the completed rejected TFM studies, including every saved training context."""
import gc
import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from scripts.audit_unlabeled_views import stats, assert_stats
from src import preprocess as pp
from src.focused_features import FocusedBuilder
from src.logit_stacking import LogitStacker
from src.optimization import SearchRunner, config_id
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabicl_models import PARAMS, CHECKPOINT, CHECKPOINT_SHA256
from src.train import _write_json, _write_csv

RUNS=["tabicl_20261030","tabicl_paired_20261030","tabicl_stacking_20261030"]


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=SearchRunner();dev=old.fold_of!=5
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public
    plan=read(old.report/"frozen_plan.json")
    expected="270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    assert sha(old.output/"submission.csv")==expected
    assert sha(PROJECT_ROOT/"outputs/kaggle_ready_4690/submission_ensemble4690.csv")==expected
    ledger_hash=sha(EXPERIMENTS_DIR/"kaggle_submissions.json")
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT)==CHECKPOINT_SHA256
    protocols={run:read(EXPERIMENTS_DIR/run/"protocol.json") for run in RUNS}
    summaries={run:read(EXPERIMENTS_DIR/run/"summary.json") for run in RUNS}
    for run,p in protocols.items():
        for path,digest in p["sources"].items():assert sha(PROJECT_ROOT/path)==digest,path
        for path,digest in p["data_hashes"].items():assert sha(pp.find_data_dir()/path)==digest,path
        assert summaries[run]["status"]=="no_candidate","New candidate requires its own full-export audit"
        assert not (PROJECT_ROOT/"models"/run/"ensemble.joblib").exists()
        assert not list((PROJECT_ROOT/"outputs"/run).glob("submission*.csv"))
    parent=EXPERIMENTS_DIR/RUNS[0];paired=EXPERIMENTS_DIR/RUNS[1];stacked=EXPERIMENTS_DIR/RUNS[2]
    assert sha(parent/"pretrained_source.json")==protocols[RUNS[0]]["pretrained_source_sha256"]
    assert sha(parent/"protocol.json")==protocols[RUNS[1]]["parent_protocol_sha256"]
    assert sha(parent/"summary.json")==protocols[RUNS[1]]["parent_summary_sha256"]
    for name,digest in protocols[RUNS[2]]["parent_protocols"].items():assert sha(EXPERIMENTS_DIR/name/"protocol.json")==digest
    for name,digest in protocols[RUNS[2]]["parent_results"].items():assert sha(EXPERIMENTS_DIR/name/"summary.json")==digest
    reference=np.full(len(old.y),np.nan);reference[dev]=0.
    for weight,component in zip(plan["weights"],plan["members"]):
        for seed in component["seeds"]:
            trial=config_id(dict(component["config"],seed=seed),"confirmation",[1,2,3,4])
            p=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            reference[dev]+=weight/len(component["seeds"])*p[dev]
    reload_count=0;max_error=0.;context_rows=0

    def audit_context(run,stem,ti,vi):
        nonlocal reload_count,max_error,context_rows
        out=PROJECT_ROOT/"outputs"/run;modelpath=PROJECT_ROOT/"models"/run/f"{stem}.joblib"
        saved=np.load(out/f"{stem}.npz")
        np.testing.assert_array_equal(saved["training"],ti);np.testing.assert_array_equal(saved["validation"],vi)
        assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
        meta=read(EXPERIMENTS_DIR/run/f"{stem}.json")
        assert sha(modelpath)==meta["model_sha256"]
        m=joblib.load(modelpath);assert m._model is None
        np.testing.assert_array_equal(m.training_y,old.y[ti])
        expected_X=FocusedBuilder("compact",True).fit(old.features.iloc[ti],public).transform(old.features.iloc[ti])
        pd.testing.assert_frame_equal(m.training_X,expected_X)
        assert pp.TARGET not in m.training_X
        q=m.predict_proba(old.features.iloc[vi])[:,1]
        np.testing.assert_allclose(q,saved["probability"],atol=2e-6,rtol=0)
        np.testing.assert_array_equal(q>=.5,saved["probability"]>=.5)
        for key,value in PARAMS.items():assert m._model.get_params()[key]==value,key
        max_error=max(max_error,float(np.max(np.abs(q-saved["probability"]))))
        reload_count+=1;context_rows+=len(ti)
        del m;gc.collect();torch.cuda.empty_cache()
        print(f"Audited {run}/{stem}",flush=True)
        return saved["probability"]

    table=pd.read_csv(parent/"development_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId);np.testing.assert_array_equal(table.Transported,old.y)
    np.testing.assert_allclose(table.reference,reference,atol=1e-14,rtol=0,equal_nan=True)
    tfm=np.full(len(old.y),np.nan)
    for fold,(ti,vi) in enumerate(old.splits,1):
        assert not (old.fold_of[ti]==5).any()
        tfm[vi]=audit_context(RUNS[0],f"development_fold_{fold}",ti,vi)
    np.testing.assert_allclose(tfm,table.tabicl,atol=1e-14,rtol=0,equal_nan=True)
    alternative=.5*reference+.5*tfm
    np.testing.assert_allclose(alternative,table.alternative,atol=1e-14,rtol=0,equal_nan=True)
    d=stats(old.y[dev],reference[dev],alternative[dev],old.fold_of[dev]);assert_stats(d,summaries[RUNS[0]]["development"])
    gate=d["gain"]>=.001 and d["improved_folds"]>=3 and d["log_loss"]<=d["reference_log_loss"]+.002
    assert not gate and not read(parent/"development.json")["passed"]
    paired_rows=[]
    for repeat,seed in enumerate(protocols[RUNS[1]]["split_seeds"],1):
        rows=pd.read_csv(paired/f"repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
        np.testing.assert_array_equal(rows.PassengerId,old.raw.PassengerId);np.testing.assert_array_equal(rows.Transported,old.y)
        ref,q,folds=np.zeros(len(old.y)),np.zeros(len(old.y)),np.zeros(len(old.y),int)
        for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(old.features,groups=old.features.GroupId),1):
            tag=f"repeat_{repeat}_fold_{fold}"
            base=np.load(PROJECT_ROOT/f"outputs/fixed_weight_confirmation_20261029/{tag}.npz")
            np.testing.assert_array_equal(base["training"],ti);np.testing.assert_array_equal(base["validation"],vi)
            q[vi]=audit_context(RUNS[1],tag,ti,vi)
            ref[vi]=base["probability"]@np.array(plan["weights"]);folds[vi]=fold
        alt=.5*ref+.5*q
        for col,values in [("reference",ref),("alternative",alt),("tabicl",q),("fold",folds)]:
            np.testing.assert_allclose(rows[col],values,atol=1e-14,rtol=0)
        r=stats(old.y,ref,alt,folds);assert_stats(r,summaries[RUNS[1]]["repeats"][repeat-1]);paired_rows.append(r)
    passed=(all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in paired_rows) and
            np.mean([r["gain"] for r in paired_rows])>=.001 and sum(r["improved_folds"] for r in paired_rows)>=6)
    assert not passed and summaries[RUNS[1]]["original_development_gate_still_failed"]
    result=pd.read_csv(stacked/"development_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    alt,control=np.full(len(old.y),np.nan),np.full(len(old.y),np.nan)
    for fold,(ti,vi) in enumerate(old.splits,1):
        record=read(stacked/f"development_fold_{fold}_calibrators.json")
        cache=PROJECT_ROOT/record["inner_source"]
        assert sha(cache)==record["inner_source_sha256"]
        z=np.load(cache);fit,cal=z["fit_indices"],z["cal_indices"]
        a,b=next(GroupShuffleSplit(1,test_size=.25,random_state=20261029+fold).split(ti,groups=old.features.GroupId.iloc[ti]))
        np.testing.assert_array_equal(fit,ti[a]);np.testing.assert_array_equal(cal,ti[b])
        assert not set(np.r_[fit,cal])&set(vi)
        p=audit_context(RUNS[2],f"development_fold_{fold}_inner_tfm",fit,cal)
        for enabled,label,output in [(True,"stack",alt),(False,"control",control)]:
            c=LogitStacker(tfm_enabled=enabled).fit(z["probability"],p,old.y[cal])
            np.testing.assert_allclose([c.intercept_,c.weight_],[record[label]["intercept"],record[label]["tfm_weight"]],atol=1e-12,rtol=0)
            output[vi]=c.predict(reference[vi],tfm[vi])
    np.testing.assert_array_equal(result.PassengerId,old.raw.PassengerId)
    for column,value in [("reference",reference),("tfm",tfm),("alternative",alt),("control",control)]:
        np.testing.assert_allclose(result[column],value,atol=1e-14,rtol=0,equal_nan=True)
        assert np.isnan(result[column].to_numpy()[~dev]).all()
    s=stats(old.y[dev],reference[dev],alt[dev],old.fold_of[dev]);c=stats(old.y[dev],reference[dev],control[dev],old.fold_of[dev])
    assert_stats(s,summaries[RUNS[2]]["development"]);assert_stats(c,summaries[RUNS[2]]["control"])
    assert not (s["gain"]>=.001 and s["improved_folds"]>=3 and s["log_loss"]<=s["reference_log_loss"]+.002 and s["accuracy"]>=c["accuracy"]+.0005)
    assert summaries[RUNS[2]]["completed_repeats"]==0
    # Descriptive subgroups; these are not used to select a new routing rule.
    subgroup=[]
    for name,mask in [("全部开发",dev),("G甲板",dev&old.features.Deck.eq("G").fillna(False).to_numpy(bool)),
                      ("非G甲板",dev&~old.features.Deck.eq("G").fillna(False).to_numpy(bool))]:
        row=stats(old.y[mask],reference[mask],alt[mask],old.fold_of[mask])
        subgroup.append(dict(cohort=name,rows=int(mask.sum()),**row))
    destination=EXPERIMENTS_DIR/"information_search_20261030";destination.mkdir(exist_ok=True)
    _write_csv(pd.DataFrame([dict(stage="固定50%开发",**d),*[dict(stage=f"固定50%复核{n}",**r) for n,r in enumerate(paired_rows,1)],
                            dict(stage="训练内学习组合开发",**s),dict(stage="仅旧模型截距对照",**c)]),destination/"comparison.csv")
    _write_json(destination/"stacking_subgroups.json",subgroup)
    _write_json(destination/"summary.json",dict(status="no_candidate",new_submission=False,observed_best=.81108,
                model_contexts_audited=reload_count,original_development=d,paired=paired_rows,stacking=s,control=c,
                finding="Pretrained probabilities improve logloss consistently, but accuracy gains fail fold/repartition stability; no qualified new information source established"))
    assert sha(EXPERIMENTS_DIR/"kaggle_submissions.json")==ledger_hash
    log_path=PROJECT_ROOT/"outputs/information_search_20261030_tests.log"
    log=log_path.read_text(encoding="utf-8",errors="replace")
    match=re.search(r"Ran (\d+) tests",log);assert match and re.search(r"\nOK\s*$",log)
    audit=dict(passed=True,tfm_contexts_reloaded=reload_count,training_context_rows_verified=context_rows,
                max_gpu_reprediction_error=max_error,source_and_data_hashes_match=True,checkpoint_sha256=CHECKPOINT_SHA256,
                nested_group_isolation_verified=True,all_saved_training_features_and_labels_match=True,
                gates_independently_recomputed=True,all_studies_no_candidate=True,original_failure_preserved=True,
                tests_passed=int(match.group(1)),test_log_sha256=sha(log_path),incumbent_csv_sha256=expected,
                platform_ledger_unchanged=True,no_new_csv=True,
                limitation="Adaptive historical studies and repeat rows; no new independent holdout or Kaggle score")
    _write_json(PROJECT_ROOT/"reports/checks/information_search_20261030.json",audit)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
