"""Reconstruct all arithmetic-mean gates and audit a qualified export if present."""
import gc
import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from scripts.audit_unlabeled_views import stats,assert_stats
from src import preprocess as pp
from src.focused_features import FocusedBuilder
from src.optimization import SearchRunner,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.tabicl_models import PARAMS,CHECKPOINT,CHECKPOINT_SHA256
from src.train import _write_json,_write_csv

NAME="tabicl_probability_mean_20261030"


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(rows):
    return (len(rows)==2 and all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows)
            and np.mean([r["gain"] for r in rows])>=.001 and sum(r["improved_folds"] for r in rows)>=6)


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/NAME;out=PROJECT_ROOT/"outputs"/NAME;models=PROJECT_ROOT/"models"/NAME
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    plan=read(old.report/"frozen_plan.json");public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    for path,digest in protocol["sources"].items():assert sha(PROJECT_ROOT/path)==digest,path
    for path,digest in protocol["data_hashes"].items():assert sha(pp.find_data_dir()/path)==digest,path
    assert sha(old.report/"frozen_plan.json")==protocol["original_plan_sha256"]
    assert sha(EXPERIMENTS_DIR/"tabicl_failure_diagnosis_20261030/per_view_summary.json")==protocol["diagnostic_sha256"]
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT)==CHECKPOINT_SHA256
    expected="270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    assert sha(old.output/"submission.csv")==expected
    assert sha(PROJECT_ROOT/"outputs/kaggle_ready_4690/submission_ensemble4690.csv")==expected
    contexts=0;reference_models=0;max_error=0.

    def context(tag,ti,vi):
        nonlocal contexts,max_error
        z=np.load(out/f"{tag}.npz");np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
        assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
        path=models/f"{tag}.joblib";record=read(report/f"{tag}.json");assert sha(path)==record["model_sha256"]
        m=joblib.load(path);assert m._model is None
        np.testing.assert_array_equal(m.training_y,old.y[ti])
        X=FocusedBuilder("compact",True).fit(old.features.iloc[ti],public).transform(old.features.iloc[ti])
        pd.testing.assert_frame_equal(m.training_X,X);assert pp.TARGET not in X
        q=m.predict_proba(old.features.iloc[vi])[:,1]
        np.testing.assert_allclose(q,z["probability"],atol=2e-6,rtol=0)
        np.testing.assert_array_equal(q>=.5,z["probability"]>=.5)
        for key,value in {**PARAMS,"average_logits":False}.items():assert m._model.get_params()[key]==value,key
        assert sum(len(v) for v in m._model.ensemble_generator_.class_shuffles_.values())==8
        max_error=max(max_error,float(np.max(np.abs(q-z["probability"]))));contexts+=1
        if record["original_context_sha256"]:
            match=re.fullmatch(r"historical_repeat_(\d)_fold_(\d)",tag);assert match
            original=PROJECT_ROOT/f"models/tabicl_paired_20261030/repeat_{match[1]}_fold_{match[2]}.joblib"
            assert sha(original)==record["original_context_sha256"]
            before=joblib.load(original);pd.testing.assert_frame_equal(before.training_X,m.training_X)
            np.testing.assert_array_equal(before.training_y,m.training_y)
        del m;gc.collect();torch.cuda.empty_cache();print(f"Audited arithmetic context {tag}",flush=True)
        return z["probability"]

    rows_by_stage={};fresh_deltas=[]
    for stage,saved_rows in summary["stages"].items():
        rows=[]
        for saved_row in saved_rows:
            repeat=saved_row["repeat"]
            table=pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId);np.testing.assert_array_equal(table.Transported,old.y)
            ref,p,folds=np.zeros(len(old.y)),np.zeros(len(old.y)),np.zeros(len(old.y),int)
            seed=([20261029,20261106] if stage=="historical" else [20261030,20261107])[repeat-1]
            for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(old.features,groups=old.features.GroupId),1):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                p[vi]=context(tag,ti,vi);folds[vi]=fold
                if stage=="historical":
                    base=np.load(PROJECT_ROOT/f"outputs/fixed_weight_confirmation_20261029/repeat_{repeat}_fold_{fold}.npz")
                    np.testing.assert_array_equal(base["training"],ti);np.testing.assert_array_equal(base["validation"],vi)
                    ref[vi]=base["probability"]@np.array(plan["weights"])
                else:
                    q=np.zeros(len(vi));names=[]
                    for j,(weight,component) in enumerate(zip(plan["weights"],plan["members"])):
                        for model_seed in component["seeds"]:
                            stem=f"reference_{tag}_component_{j}_seed_{model_seed}";names.append(stem)
                            z=np.load(out/f"{stem}.npz");np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
                            modelpath=models/f"{stem}.joblib";record=read(report/f"{stem}.json");assert sha(modelpath)==record["model_sha256"]
                            assert record["config"]==dict(component["config"],seed=model_seed)
                            m=joblib.load(modelpath);cfg=component["config"]
                            assert m["builder"].feature_set==cfg["feature_set"] and m["builder"].strategy==cfg["strategy"]
                            params=m["model"].get_params()
                            for key,value in cfg["params"].items():
                                actual_key="n_estimators" if key=="iterations" and cfg["family"]=="lightgbm" else key
                                assert params[actual_key]==value,key
                            assert params["random_seed" if cfg["family"]=="catboost" else "random_state"]==model_seed
                            predicted=m["model"].predict_proba(m["encoder"].transform(m["builder"].transform(old.features.iloc[vi])))[:,1]
                            np.testing.assert_allclose(predicted,z["probability"],atol=1e-14,rtol=0)
                            q+=weight/len(component["seeds"])*z["probability"];reference_models+=1
                    assert names==read(report/f"reference_{tag}.json")["members"]
                    ref[vi]=q
            alt=.5*ref+.5*p
            for col,value in [("reference",ref),("component",p),("alternative",alt),("fold",folds)]:np.testing.assert_allclose(table[col],value,atol=1e-14,rtol=0)
            row=dict(repeat=repeat,**stats(old.y,ref,alt,folds));assert_stats({k:v for k,v in row.items() if k!="repeat"},saved_row);rows.append(row)
            if stage=="fresh":fresh_deltas.append(((alt>=.5)==old.y).astype(float)-((ref>=.5)==old.y))
        rows_by_stage[stage]=rows
        for actual,saved in zip(rows,read(report/f"{stage}_results.json")):assert_stats({k:v for k,v in actual.items() if k!="repeat"},saved)
    qualified=all(gate(rows_by_stage.get(stage,[])) for stage in ["historical","fresh"])
    assert qualified==(summary["status"]=="candidate_generated")
    candidate=None
    if qualified:
        bundle=joblib.load(models/"ensemble.joblib");assert len(bundle["members"])==50
        np.testing.assert_allclose(sum(m["weight"] for m in bundle["members"]),1.,atol=1e-12,rtol=0)
        original=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
        for before,after in zip(original["members"],bundle["members"][:45]):np.testing.assert_allclose(after["weight"],.5*before["weight"],atol=1e-15,rtol=0)
        old_probability,_=predict_ensemble(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
        test=.5*old_probability
        canonical=pd.read_csv(old.report/"oof_predictions.csv",float_precision="round_trip")
        oof=.5*canonical.probability.to_numpy()
        for fold in range(1,6):
            ti,vi=np.flatnonzero(old.fold_of!=fold),np.flatnonzero(old.fold_of==fold)
            oof[vi]+=.5*context(f"final_fold_{fold}",ti,vi)
            m=bundle["members"][44+fold];np.testing.assert_allclose(m["weight"],.1,atol=1e-15,rtol=0)
            q=m["model"].predict_proba(old.test_features)[:,1];test+=.1*q
            m["model"]._model=None;gc.collect();torch.cuda.empty_cache()
        saved=pd.read_csv(out/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
        np.testing.assert_array_equal(saved.PassengerId,old.raw_test.PassengerId)
        np.testing.assert_allclose(test,saved.probability,atol=2e-6,rtol=0);np.testing.assert_array_equal(test>=.5,saved.prediction)
        stored_oof=pd.read_csv(report/"canonical_oof.csv",dtype={"PassengerId":str},float_precision="round_trip")
        np.testing.assert_array_equal(stored_oof.PassengerId,old.raw.PassengerId)
        np.testing.assert_allclose(oof,stored_oof.probability,atol=2e-6,rtol=0)
        np.testing.assert_allclose(((oof>=.5)==old.y).mean(),summary["canonical_accuracy"],atol=1e-14,rtol=0)
        submission=pd.read_csv(out/"submission_tabicl_probability_mean4690.csv",dtype={"PassengerId":str})
        template=pp.load_submission_template()
        assert list(submission)==["PassengerId","Transported"] and len(submission)==4277 and submission.PassengerId.is_unique
        np.testing.assert_array_equal(submission.PassengerId,template.PassengerId);assert submission.Transported.dtype==bool
        np.testing.assert_array_equal(submission.Transported,test>=.5)
        cli=out/"cli_reproduced_20261030"
        reproduced=pd.read_csv(cli/"submission.csv",dtype={"PassengerId":str});pd.testing.assert_frame_equal(submission,reproduced)
        cli_probability=pd.read_csv(cli/"test_predictions.csv",float_precision="round_trip")
        np.testing.assert_allclose(cli_probability.probability,saved.probability,atol=2e-6,rtol=0)
        candidate=dict(rows=4277,sha256=sha(out/"submission_tabicl_probability_mean4690.csv"),
                       unique_ids=True,boolean_labels=True,template_order=True,cli_reproduced=True,
                       independent_test_recombination_max_error=float(np.max(np.abs(test-saved.probability.to_numpy()))))
    else:
        assert not (models/"ensemble.joblib").exists() and not (out/"submission_tabicl_probability_mean4690.csv").exists()
    uncertainty=None
    if len(fresh_deltas)==2:
        delta=np.mean(fresh_deltas,axis=0);codes,groups=pd.factorize(old.features.GroupId)
        sums=np.bincount(codes,weights=delta);sizes=np.bincount(codes);rng=np.random.default_rng(20481030);boot=[]
        for _ in range(20):
            draw=rng.integers(0,len(groups),size=(100,len(groups)))
            boot.extend((sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)).tolist())
        uncertainty=dict(mean_gain=float(delta.mean()),group_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
                         groups=len(groups),resamples=2000,seed=20481030,
                         caveat="Conditional fixed-prediction interval, averaging repeated outcomes per passenger before group resampling. Does not include model-selection or refitting uncertainty; not a promotion criterion.")
    logpath=PROJECT_ROOT/"outputs/tabicl_input_followup_20261030_tests.log";log=logpath.read_text(encoding="utf-8",errors="replace")
    match=re.search(r"Ran (\d+) tests",log);assert match and re.search(r"\nOK\s*$",log)
    ledger=read(EXPERIMENTS_DIR/"kaggle_submissions.json")
    assert ledger["current_platform_best"]["model_run"]=="optimization_20261025" and ledger["current_platform_best"]["score"]==.81108
    audit=dict(passed=True,arithmetic_contexts_reloaded=contexts,fresh_reference_models_reloaded=reference_models,
                max_validation_reload_error=max_error,source_data_checkpoint_hashes_match=True,group_isolation_verified=True,
                all_eight_views_retained=True,only_internal_aggregation_changed=True,gates_recomputed=True,gate_passed=bool(qualified),
                tests_passed=int(match.group(1)),test_log_sha256=sha(logpath),incumbent_submission_sha256=expected,
                candidate=candidate,conditional_uncertainty=uncertainty,platform_best_retained=True,
                caveat="Hypothesis selected from historical failures. Fresh partitions use the same known rows; Kaggle improvement remains unobserved.")
    _write_json(PROJECT_ROOT/"reports/checks/tabicl_probability_mean_20261030.json",audit)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
