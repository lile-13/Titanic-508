"""One nested logit stack, with held-out-group teacher probabilities only."""
import gc
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.calibration_study import holdout_calibration_data
from scripts.tabicl_study import compare
from src import preprocess as pp
from src.logit_stacking import LogitStacker, StackedTabICLFold
from src.optimization import SearchRunner, predict_ensemble
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.probability_calibration import IdentityTransform
from src.tabicl_models import fit_member, CHECKPOINT_SHA256, CHECKPOINT
from src.train import _write_json, _write_csv, make_submission

NAME="tabicl_stacking_20261030"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=SearchRunner()
    report,out,modeldir=EXPERIMENTS_DIR/NAME,PROJECT_ROOT/"outputs"/NAME,PROJECT_ROOT/"models"/NAME
    for directory in [report,out,modeldir]:directory.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    parent=EXPERIMENTS_DIR/"tabicl_20261030"
    paired=EXPERIMENTS_DIR/"tabicl_paired_20261030"
    assert json.loads((parent/"summary.json").read_text(encoding="utf-8"))["status"]=="no_candidate"
    assert json.loads((paired/"summary.json").read_text(encoding="utf-8"))["status"]=="no_candidate"
    protocol=dict(name=NAME,ridge=.01,weight_bound=[0.,1.],intercept_shrinkage_target=0.,weight_shrinkage_target=.5,threshold=.5,
        method="Fit intercept and constrained TFM weight on inner 25% group holdout: logit(p)=b+(1-w)logit(old)+w logit(TFM). Both temporary teachers exclude every calibration group. Transfer fitted parameters to full outer-training teachers. No cohort features or fixed-weight search.",
        control="Same held-out rows, but TFM weight fixed to zero; only old-model intercept fitted with same ridge. Not selected by outer labels.",
        gate="Development accuracy gain >=.001, >=3/4 folds improve, global logloss deterioration <=.002, accuracy >= reference-only fitted-intercept control +.0005. Then two paired grouped five-fold repeats: both nonnegative, mean gain >=.001, >=6/10 improved folds, mean gain over matched control >=.0005, loss deterioration <=.002 each. Stop at any failed stage; no further stacking configuration search.",
        rationale="Raw fixed TFM blend failed development and separate paired checks despite lower logloss on every partition. This is a new adaptive learned-combination hypothesis; both prior failures remain unchanged.",
        caveat="Repeated historical rows and source splits; adaptive development, not independent unbiased confirmation. No guarantee of Kaggle improvement.",
        inner_sources=dict(development="calibration_study_20261029/inner_fold_{fold}.npz, seed 20261029+fold",
                           repeat_1="g_cohort_routing_20261029/repeat_1_fold_{fold}_inner.npz, seed 690001+10+fold",
                           repeat_2="new cache, seed 690001+20+fold",final="new cache, seed 740001+fold"),
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"],
        checkpoint_sha256=CHECKPOINT_SHA256,
        parent_protocols={name:sha(EXPERIMENTS_DIR/name/"protocol.json") for name in ["tabicl_20261030","tabicl_paired_20261030","calibration_study_20261029","g_cohort_routing_20261029"]},
        parent_results={name:sha(EXPERIMENTS_DIR/name/"summary.json") for name in ["tabicl_20261030","tabicl_paired_20261030"]},
        original_plan_sha256=sha(old.report/"frozen_plan.json"),
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/tabicl_stacking_study.py","src/logit_stacking.py","scripts/calibration_study.py",
            "scripts/tabicl_study.py","src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py","features/folds.csv","requirements-tabicl.txt"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Nested stacking protocol changed")
    else:_write_json(path,protocol)
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT)==CHECKPOINT_SHA256
    for name,digest in protocol["data_hashes"].items():assert sha(pp.find_data_dir()/name)==digest
    if (report/"summary.json").exists():
        print("Completed frozen nested stacking; use audit",flush=True);return
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)

    def tfm_cached(ti,vi,tag):
        cache,modelpath=out/f"{tag}_tfm.npz",modeldir/f"{tag}_tfm.joblib"
        if cache.exists() and modelpath.exists():
            z=np.load(cache);np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
            return z["probability"]
        m=fit_member(old.features.iloc[ti],old.y[ti],public);p=m.predict_proba(old.features.iloc[vi])[:,1]
        joblib.dump(m,modelpath,compress=3);np.savez_compressed(cache,training=ti,validation=vi,probability=p)
        _write_json(report/f"{tag}_tfm.json",dict(model_sha256=sha(modelpath),rows=len(ti),validation_rows=len(vi)))
        del m;gc.collect();torch.cuda.empty_cache()
        return p

    def calibrate(ti,vi,tag,inner_source,inner_seed,ref,tfm):
        ci,teacher,_=holdout_calibration_data(old,plan,ti,inner_seed,inner_source)
        z=np.load(inner_source);fit=z["fit_indices"]
        assert not set(np.r_[fit,ci])&set(vi)
        assert not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[ci])
        learned_tfm=tfm_cached(fit,ci,tag+"_inner")
        calibrators=[LogitStacker(tfm_enabled=True).fit(teacher,learned_tfm,old.y[ci]),
                     LogitStacker(tfm_enabled=False).fit(teacher,learned_tfm,old.y[ci])]
        _write_json(report/f"{tag}_calibrators.json",dict(inner_source=str(inner_source.relative_to(PROJECT_ROOT)),
                    inner_source_sha256=sha(inner_source),stack=calibrators[0].audit_,control=calibrators[1].audit_))
        joblib.dump(calibrators[0],modeldir/f"{tag}_calibrator.joblib",compress=3)
        return calibrators[0].predict(ref,tfm),calibrators[1].predict(ref,tfm)

    dev=old.fold_of!=5
    parent_table=pd.read_csv(parent/"development_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(parent_table.PassengerId,old.raw.PassengerId)
    ref,tfm=parent_table.reference.to_numpy(),parent_table.tabicl.to_numpy()
    alt,control=np.full(len(old.y),np.nan),np.full(len(old.y),np.nan)
    for fold,(ti,vi) in enumerate(old.splits,1):
        alt[vi],control[vi]=calibrate(ti,vi,f"development_fold_{fold}",
            PROJECT_ROOT/f"outputs/calibration_study_20261029/inner_fold_{fold}.npz",20261029+fold,ref[vi],tfm[vi])
        print(f"Nested TFM stack development fold {fold} ready",flush=True)
    d=compare(old.y[dev],ref[dev],alt[dev],old.fold_of[dev])
    c=compare(old.y[dev],ref[dev],control[dev],old.fold_of[dev])
    passed=(d["gain"]>=.001 and d["improved_folds"]>=3 and d["log_loss"]<=d["reference_log_loss"]+.002 and d["accuracy"]>=c["accuracy"]+.0005)
    _write_json(report/"development.json",dict(passed=bool(passed),stack=d,control=c))
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,reference=ref,tfm=tfm,
                                alternative=alt,control=control)),report/"development_predictions.csv")
    print(f"Nested TFM stack development gain {d['gain']:+.6f}, improved {d['improved_folds']}/4",flush=True)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",stage="development",development=d,control=c,
                    completed_repeats=0,kaggle_score=None,reason="Single nested logit combination failed fixed development gate"));return
    rows=[]
    for repeat in [1,2]:
        source=pd.read_csv(paired/f"repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
        ref,tfm=source.reference.to_numpy(),source.tabicl.to_numpy();folds=source.fold.to_numpy()
        alt,control=np.full(len(old.y),np.nan),np.full(len(old.y),np.nan)
        for fold in range(1,6):
            z=np.load(PROJECT_ROOT/f"outputs/fixed_weight_confirmation_20261029/repeat_{repeat}_fold_{fold}.npz")
            ti,vi=z["training"],z["validation"]
            inner=(PROJECT_ROOT/f"outputs/g_cohort_routing_20261029/repeat_1_fold_{fold}_inner.npz" if repeat==1
                   else out/f"repeat_2_fold_{fold}_teacher.npz")
            alt[vi],control[vi]=calibrate(ti,vi,f"repeat_{repeat}_fold_{fold}",inner,690001+repeat*10+fold,ref[vi],tfm[vi])
            print(f"Nested TFM stack repeat {repeat} fold {fold} ready",flush=True)
        d_repeat=compare(old.y,ref,alt,folds);c_repeat=compare(old.y,ref,control,folds)
        rows.append(dict(repeat=repeat,stack=d_repeat,control=c_repeat))
        _write_json(report/"repeat_results.json",rows)
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=folds,reference=ref,tfm=tfm,
                                    alternative=alt,control=control)),report/f"repeat_{repeat}_predictions.csv")
        if d_repeat["gain"]<0 or d_repeat["log_loss"]>d_repeat["reference_log_loss"]+.002 or (repeat==1 and d_repeat["improved_folds"]==0):
            _write_json(report/"summary.json",dict(status="no_candidate",stage="sensitivity",development=d,control=c,repeats=rows,
                        completed_repeats=repeat,kaggle_score=None,reason="Nested stack fixed repeated non-regression/futility gate failed"));return
    passed=(np.mean([r["stack"]["gain"] for r in rows])>=.001 and sum(r["stack"]["improved_folds"] for r in rows)>=6 and
            np.mean([r["stack"]["accuracy"]-r["control"]["accuracy"] for r in rows])>=.0005)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",stage="sensitivity",development=d,control=c,repeats=rows,
                    completed_repeats=2,kaggle_score=None,reason="Nested stack fixed repeated mean/fold/control gate failed"));return
    _write_json(report/"qualified.json",dict(development=d,control=c,repeats=rows))
    original=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=[];oof=np.zeros(len(old.y));probability=np.zeros(len(old.raw_test));identity=IdentityTransform()
    for block,fold in enumerate([5,1,2,3,4]):
        ti,vi=np.flatnonzero(old.fold_of!=fold),np.flatnonzero(old.fold_of==fold)
        children=[dict(m,weight=5*m["weight"]) for m in original["members"][block*9:block*9+9]]
        outer_tfm=tfm_cached(ti,vi,f"final_fold_{fold}_outer")
        ref_fold=np.zeros(len(vi))
        for m in children:ref_fold+=m["weight"]*m["model"].predict_proba(m["encoder"].transform(m["builder"].transform(old.features.iloc[vi])))[:,1]
        calibrate(ti,vi,f"final_fold_{fold}",out/f"final_fold_{fold}_teacher.npz",740001+fold,ref_fold,outer_tfm)
        tfm_member=joblib.load(modeldir/f"final_fold_{fold}_outer_tfm.joblib")
        stacker=joblib.load(modeldir/f"final_fold_{fold}_calibrator.joblib")
        model=StackedTabICLFold(children,tfm_member,stacker)
        oof[vi]=model.predict_proba(old.features.iloc[vi])[:,1]
        probability+=model.predict_proba(old.test_features)[:,1]/5
        tfm_member._model=None;gc.collect();torch.cuda.empty_cache()
        members.append(dict(weight=.2,builder=identity,encoder=identity,model=model))
    modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5),data_hashes=original["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,probability,pp.load_submission_template(),.5)
    check,check_submission=predict_ensemble(modelpath)
    np.testing.assert_allclose(check,probability,atol=2e-6,rtol=0);pd.testing.assert_frame_equal(check_submission,submission)
    _write_csv(submission,out/"submission_tabicl_stack4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=probability,prediction=probability>=.5)),out/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,probability=oof)),report/"canonical_oof.csv")
    _write_json(report/"summary.json",dict(status="candidate_generated",stage="completed",development=d,control=c,repeats=rows,
                completed_repeats=2,kaggle_score=None,canonical_accuracy=float(((oof>=.5)==old.y).mean()),reload_max_error=float(np.max(np.abs(check-probability)))))


if __name__=="__main__":main()
