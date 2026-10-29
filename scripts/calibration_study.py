"""嵌套按组留出校准实验；不改动已冻结基模型或历史实验文件。"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import GroupShuffleSplit,GroupKFold

from src.optimization import SearchRunner,config_id,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.probability_calibration import ProbabilityCalibrator,CalibratedFoldModel,IdentityTransform
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="calibration_study_20261029"
PROFILES=[dict(mode="global",C=C) for C in [.1,1.,10.]]+[
    dict(mode="context",C=C) for C in [.01,.1,1.]]+[
    dict(mode="slope_context",C=C) for C in [.01,.1]]


def holdout_calibration_data(runner,plan,training,seed,cache):
    if cache.exists():
        data=np.load(cache)
        ti,ci,p=data["fit_indices"],data["cal_indices"],data["probability"]
        np.testing.assert_array_equal(np.sort(np.r_[ti,ci]),np.sort(training))
    else:
        splitter=GroupShuffleSplit(1,test_size=.25,random_state=seed)
        a,b=next(splitter.split(training,runner.y[training],runner.features.GroupId.iloc[training]))
        ti,ci=training[a],training[b]
        assert not set(runner.features.GroupId.iloc[ti])&set(runner.features.GroupId.iloc[ci])
        p,_,_=runner.fit_partition(plan,ti,ci,save_test=False)
        np.savez_compressed(cache,fit_indices=ti,cal_indices=ci,probability=p)
    assert not set(ti)&set(ci)
    assert not set(runner.features.GroupId.iloc[ti])&set(runner.features.GroupId.iloc[ci])
    return ci,p,dict(fit_rows=len(ti),calibration_rows=len(ci),group_overlap=0)


def main():
    old=SearchRunner(threads=4);report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    protocol=dict(name=NAME,profiles=PROFILES,calibration_weights=[.25,.5,1.],threshold=.5,
        development_folds=[1,2,3,4],excluded_fold=5,inner_holdout_fraction=.25,inner_seed_base=20261029,
        sensitivity_seed=20261103,
        gate="Accuracy gain >=.001, >=3/4 folds improved, logloss deterioration <=.002. Then freeze and changed-partition accuracy gain >=0, logloss deterioration <=.002. No retuning.",
        method="Calibration labels belong to outer training only; temporary teachers exclude entire calibration groups; calibrator then transfers to full outer-training teacher",
        caveat="Historical folds have been inspected. Changed partitions check sensitivity, not independent blind validation. Calibration transfer is itself tested by outer grouped validation.",
        source_plan_hash=hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest(),
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
                 ["scripts/calibration_study.py","src/probability_calibration.py","src/optimization.py","src/competition_features.py","src/preprocess.py","features/folds.csv"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("协议变化，不得混用缓存")
    else:_write_json(path,protocol)
    dev=old.fold_of!=5;reference=np.full(len(old.y),np.nan);reference[dev]=0
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            p=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            assert np.isnan(p[~dev]).all();reference[dev]+=weight/len(member["seeds"])*p[dev]
    base_acc=float(accuracy_score(old.y[dev],reference[dev]>=.5));base_loss=float(log_loss(old.y[dev],reference[dev]))
    np.testing.assert_allclose(base_acc,plan["development_accuracy"],atol=1e-14)
    predictions={i:np.full(len(old.y),np.nan) for i in range(len(PROFILES))};split_audits=[]
    for fold,(ti,vi) in enumerate(old.splits,1):
        assert not (old.fold_of[ti]==5).any()
        ci,p,audit=holdout_calibration_data(old,plan,ti,20261029+fold,output/f"inner_fold_{fold}.npz")
        assert not set(ci)&set(vi);assert not (old.fold_of[ci]==5).any()
        split_audits.append(dict(fold=fold,outer_validation_rows=len(vi),outer_validation_group_overlap=0,**audit))
        for index,config in enumerate(PROFILES):
            calibrator=ProbabilityCalibrator(config["mode"],config["C"]).fit(old.features.iloc[ci],p,old.y[ci])
            predictions[index][vi]=calibrator.predict(old.features.iloc[vi],reference[vi])
        print(f"Nested calibration development fold {fold}/4 finished",flush=True)
    _write_json(report/"split_audit.json",split_audits)
    rows=[]
    for index,pred in predictions.items():
        assert np.isnan(pred[~dev]).all()
        np.savez_compressed(output/f"profile_{index}.npz",probability=pred)
        for weight in protocol["calibration_weights"]:
            p=(1-weight)*reference[dev]+weight*pred[dev];gains=[]
            for fold in range(1,5):
                mask=old.fold_of[dev]==fold
                gains.append(float(accuracy_score(old.y[dev][mask],p[mask]>=.5)-accuracy_score(old.y[dev][mask],reference[dev][mask]>=.5)))
            rows.append(dict(profile=index,config=PROFILES[index],weight=weight,
                accuracy=float(accuracy_score(old.y[dev],p>=.5)),log_loss=float(log_loss(old.y[dev],p)),
                fold_gains=gains,improved_folds=int(np.sum(np.array(gains)>0))))
    rows.sort(key=lambda r:(-r["accuracy"],r["log_loss"],r["weight"]))
    best=rows[0];passed=best["accuracy"]>=base_acc+.001 and best["improved_folds"]>=3 and best["log_loss"]<=base_loss+.002
    frozen=dict(**best,incumbent_accuracy=base_acc,incumbent_log_loss=base_loss,initial_gate_passed=bool(passed))
    _write_json(report/"comparisons.json",rows);_write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,
            reason="Predeclared initial gate failed; preserve observed Kaggle best 0.81108"))
        print(f"Calibration gate failed: {best['accuracy']:.6f} vs {base_acc:.6f}",flush=True);return
    # 先冻结校准方式和混合比例，再检查另一种分组划分；不根据结果回调参数。
    ref=np.full(len(old.y),np.nan);alt=ref.copy();indices=np.flatnonzero(dev)
    splitter=GroupKFold(4,shuffle=True,random_state=protocol["sensitivity_seed"])
    for fold,(a,b) in enumerate(splitter.split(indices,old.y[dev],old.features.GroupId[dev]),1):
        ti,vi=indices[a],indices[b]
        reference_cache=output/f"sensitivity_reference_{fold}.npz"
        if reference_cache.exists():q=np.load(reference_cache)["probability"]
        else:
            q,_,_=old.fit_partition(plan,ti,vi,save_test=False);np.savez_compressed(reference_cache,probability=q)
        ci,p,_=holdout_calibration_data(old,plan,ti,20261103+fold,output/f"sensitivity_inner_{fold}.npz")
        c=ProbabilityCalibrator(best["config"]["mode"],best["config"]["C"]).fit(old.features.iloc[ci],p,old.y[ci])
        ref[vi]=q;alt[vi]=(1-best["weight"])*q+best["weight"]*c.predict(old.features.iloc[vi],q)
        print(f"Calibration changed-partition sensitivity {fold}/4 finished",flush=True)
    sensitivity=dict(reference_accuracy=float(accuracy_score(old.y[dev],ref[dev]>=.5)),
        alternative_accuracy=float(accuracy_score(old.y[dev],alt[dev]>=.5)),reference_log_loss=float(log_loss(old.y[dev],ref[dev])),
        alternative_log_loss=float(log_loss(old.y[dev],alt[dev])),interpretation="Sensitivity check on historical rows, not blind confirmation")
    passed=sensitivity["alternative_accuracy"]>=sensitivity["reference_accuracy"] and sensitivity["alternative_log_loss"]<=sensitivity["reference_log_loss"]+.002
    sensitivity["passed"]=bool(passed);_write_json(report/"sensitivity.json",sensitivity)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,reason="Changed-partition gate failed"));return
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report/"oof_predictions.csv",dtype={"PassengerId":str})
    oof=np.zeros(len(old.y));tp=np.zeros(len(old.raw_test));members=[]
    for block,fold in enumerate([5,1,2,3,4]):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        ci,p,_=holdout_calibration_data(old,plan,ti,20261201+fold,output/f"final_inner_{fold}.npz")
        c=ProbabilityCalibrator(best["config"]["mode"],best["config"]["C"]).fit(old.features.iloc[ci],p,old.y[ci])
        children=[dict(m,weight=m["weight"]*5) for m in bundle["members"][block*9:block*9+9]]
        raw=CalibratedFoldModel(children,c,0.).predict_proba(old.features.iloc[vi])[:,1]
        np.testing.assert_allclose(raw,canonical.probability.to_numpy()[vi],atol=1e-12,rtol=0)
        model=CalibratedFoldModel(children,c,best["weight"])
        oof[vi]=model.predict_proba(old.features.iloc[vi])[:,1];tp+=model.predict_proba(old.test_features)[:,1]/5
        members.append(dict(weight=.2,builder=IdentityTransform(),encoder=IdentityTransform(),model=model))
        print(f"Frozen calibration fold {fold}/5 finished",flush=True)
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5,frozen_choice=frozen),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,tp,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_calibrated4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=tp,prediction=tp>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    reload,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reload,tp,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    original=pd.read_csv(old.output/"submission.csv")
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,sensitivity=sensitivity,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),kaggle_score=None,
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=original.Transported.to_numpy())),
        reload_max_error=float(np.max(np.abs(reload-tp))),teacher_models=45,calibrators=5,
        interpretation="Post-selection historical OOF; platform score unknown until actual submission"))


if __name__=="__main__":main()
