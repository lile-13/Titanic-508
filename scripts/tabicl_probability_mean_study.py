"""One aggregation change, all views retained, with paired and fresh split gates."""
import gc
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from scripts.tabicl_study import compare
from src import preprocess as pp
from src.frozen_reference import fit_reference
from src.optimization import SearchRunner, predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.probability_calibration import IdentityTransform
from src.tabicl_aggregation import from_context,fit_probability_mean
from src.tabicl_models import PARAMS,CHECKPOINT,CHECKPOINT_SHA256
from src.train import _write_csv,_write_json

NAME="tabicl_probability_mean_20261030"
FRESH_SEEDS=[20261030,20261107]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def passes(rows):
    return (len(rows)==2 and all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows)
            and np.mean([r["gain"] for r in rows])>=.001 and sum(r["improved_folds"] for r in rows)>=6)


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/NAME;out=PROJECT_ROOT/"outputs"/NAME;models=PROJECT_ROOT/"models"/NAME
    for path in [report,out,models]:path.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    diagnostic=EXPERIMENTS_DIR/"tabicl_failure_diagnosis_20261030/per_view_summary.json"
    protocol=dict(name=NAME,params=dict(PARAMS,average_logits=False),weight=.5,threshold=.5,
        hypothesis="A selected failure fold exposed an almost-zero-probability internal view; logit averaging amplifies its influence. Test the author's arithmetic probability aggregation, keeping every one of the same eight views. No label-based view removal, clipping, routing, weight or threshold search.",
        selection_history="Aggregation hypothesis chosen after inspecting repeat2/fold3 and learning-curve contexts. Diagnostic arithmetic mean was inspected only for that cohort. All prior failed input/model studies remain failed; historical paired checks are adaptive development.",
        historical_gate="Two existing grouped repetitions: both gains nonnegative, mean >=.001, >=6/10 improved folds, each logloss deterioration <=.002. First negative/loss/zero-improved repeat stops. No changed settings or replacement splits.",
        fresh_gate="Only after historical gate passes: newly assigned GroupKFold seeds 20261030/20261107, same gain/fold/loss gate against freshly fitted frozen original references. Fresh split outcomes not read at protocol freeze. Same historical rows, so still sensitivity evidence, not a new independent holdout.",
        original_plan_sha256=sha(old.report/"frozen_plan.json"),diagnostic_sha256=sha(diagnostic),
        parent_protocol_sha256=sha(EXPERIMENTS_DIR/"tabicl_paired_20261030/protocol.json"),
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"],
        checkpoint_sha256=CHECKPOINT_SHA256,
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/tabicl_probability_mean_study.py","src/tabicl_aggregation.py","src/frozen_reference.py",
            "src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py","scripts/tabicl_study.py","features/folds.csv","requirements-tabicl.txt"]},
        environment={p:importlib.metadata.version(p) for p in ["tabicl","torch","catboost","lightgbm","numpy","pandas","scikit-learn"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Frozen aggregation study changed")
    else:_write_json(path,protocol)
    for name,digest in protocol["data_hashes"].items():assert sha(pp.find_data_dir()/name)==digest
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT)==CHECKPOINT_SHA256
    if (report/"summary.json").exists():print("Completed aggregation study; audit instead",flush=True);return
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)

    def fitted(ti,vi,tag,original_path=None):
        cache,modelpath,recordpath=out/f"{tag}.npz",models/f"{tag}.joblib",report/f"{tag}.json"
        if cache.exists() and modelpath.exists() and recordpath.exists():
            z=np.load(cache);np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
            assert sha(modelpath)==json.loads(recordpath.read_text(encoding="utf-8"))["model_sha256"]
            return z["probability"]
        assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
        member=(from_context(joblib.load(original_path)) if original_path is not None else fit_probability_mean(old.features.iloc[ti],old.y[ti],public))
        np.testing.assert_array_equal(member.training_y,old.y[ti])
        p=member.predict_proba(old.features.iloc[vi])[:,1];assert np.isfinite(p).all()
        assert member._model.get_params()["average_logits"] is False
        joblib.dump(member,modelpath,compress=3);np.savez_compressed(cache,training=ti,validation=vi,probability=p)
        _write_json(recordpath,dict(model_sha256=sha(modelpath),tag=tag,training_rows=len(ti),validation_rows=len(vi),
                    original_context_sha256=sha(original_path) if original_path is not None else None,
                    columns=list(member.training_X),all_eight_views_retained=True))
        del member;gc.collect();torch.cuda.empty_cache()
        print(f"Arithmetic TFM {tag} ready",flush=True)
        return p

    stages={}
    for stage in ["historical","fresh"]:
        rows=[]
        for repeat in [1,2]:
            ref,p,original_tfm=[np.full(len(old.y),np.nan) for _ in range(3)];folds=np.zeros(len(old.y),int)
            if stage=="historical":
                old_table=pd.read_csv(EXPERIMENTS_DIR/f"tabicl_paired_20261030/repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
                np.testing.assert_array_equal(old_table.PassengerId,old.raw.PassengerId)
                split=[]
                for fold in range(1,6):
                    z=np.load(PROJECT_ROOT/f"outputs/fixed_weight_confirmation_20261029/repeat_{repeat}_fold_{fold}.npz")
                    split.append((z["training"],z["validation"]))
            else:
                split=list(GroupKFold(5,shuffle=True,random_state=FRESH_SEEDS[repeat-1]).split(old.features,groups=old.features.GroupId))
            for fold,(ti,vi) in enumerate(split,1):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                original=(PROJECT_ROOT/f"models/tabicl_paired_20261030/repeat_{repeat}_fold_{fold}.joblib" if stage=="historical" else None)
                if stage=="historical":
                    ref[vi]=old_table.reference.to_numpy()[vi];original_tfm[vi]=old_table.tabicl.to_numpy()[vi]
                else:
                    ref[vi]=fit_reference(old,plan,ti,vi,tag,out,models,report)
                p[vi]=fitted(ti,vi,tag,original);folds[vi]=fold
            alt=.5*ref+.5*p
            r=dict(repeat=repeat,**compare(old.y,ref,alt,folds));rows.append(r)
            _write_json(report/f"{stage}_results.json",rows)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=folds,reference=ref,component=p,
                                        original_logit_component=original_tfm,alternative=alt)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            print(f"Arithmetic {stage} repeat {repeat}: gain {r['gain']:+.6f}, {r['improved_folds']}/5 folds",flush=True)
            if r["gain"]<0 or r["log_loss"]>r["reference_log_loss"]+.002 or (repeat==1 and r["improved_folds"]==0):break
        stages[stage]=rows
        if not passes(rows):
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None,
                        reason="Fixed all-view probability-mean gain/fold/loss gate failed; no post-result retuning"))
            return
    _write_json(report/"qualified.json",dict(stages=stages,no_retuning=True))
    original=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=[dict(m,weight=.5*m["weight"]) for m in original["members"]];identity=IdentityTransform()
    canonical=pd.read_csv(old.report/"oof_predictions.csv",float_precision="round_trip");oof=.5*canonical.probability.to_numpy()
    for fold in range(1,6):
        ti,vi=np.flatnonzero(old.fold_of!=fold),np.flatnonzero(old.fold_of==fold)
        oof[vi]+=.5*fitted(ti,vi,f"final_fold_{fold}")
        member=joblib.load(models/f"final_fold_{fold}.joblib")
        members.append(dict(weight=.1,builder=identity,encoder=identity,model=member))
    modelpath=models/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5),data_hashes=original["data_hashes"]),modelpath,compress=3)
    p,submission=predict_ensemble(modelpath);again,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(p,again,atol=2e-6,rtol=0);pd.testing.assert_frame_equal(submission,check)
    _write_csv(submission,out/"submission_tabicl_probability_mean4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=p,prediction=p>=.5)),out/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,probability=oof)),report/"canonical_oof.csv")
    old_test=pd.read_csv(old.output/"test_predictions.csv")
    _write_json(report/"summary.json",dict(status="candidate_generated",stages=stages,kaggle_score=None,models=50,
                changed_test_labels=int(np.sum((p>=.5)!=old_test.prediction.to_numpy())),
                canonical_accuracy=float(((oof>=.5)==old.y).mean()),reload_max_error=float(np.max(np.abs(p-again)))))


if __name__=="__main__":main()
