"""Nested G-only probability offsets: training-holdout roles versus context control."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from scripts.calibration_study import holdout_calibration_data
from scripts.g_deck_study import metrics
from src import preprocess as pp
from src.g_deck_models import g_mask
from src.g_offset_models import PROFILES,GOffsetCalibrator,corrected_probability,GOffsetFoldModel
from src.optimization import SearchRunner,config_id,predict_ensemble
from src.probability_calibration import IdentityTransform
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission

NAME="g_deck_offset_20261029"
WEIGHTS=[.25,.5,1.]


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    protocol=dict(name=NAME,profiles=PROFILES,weights=WEIGHTS,threshold=.5,
        method="Fix teacher logit coefficient at 1; ridge correction uses CryoSleep and Side, optionally other-group role variables and CryoSleep interactions; all scales fixed",
        fitting="Calibrate on 25% GroupShuffleSplit holdout inside outer training, with teacher excluding whole calibration groups. Only G holdout labels used to fit correction. Public features never use labels.",
        selection="Best enriched profile and blend by development accuracy, logloss, smaller blend; compare same-ridge/same-weight no-role context control",
        gate="Global gain >=.0015; G gain >=.005, >=3/4 G folds improve; role-versus-control G gain >=.003; G logloss deterioration <=.005 and global <=.002. Then fixed two repetitions mean global gain >=.001, both nonnegative, >=6/10 G folds improve, mean role-versus-control G gain >=.003; same logloss limits. No retuning.",
        development_folds=[1,2,3,4],excluded_fold=5,repeat_seeds=[20261029,20261106],
        caveat="Adaptive follow-up after historical G-only model experiments; original rows and folds already inspected. Not blind confirmation or a Kaggle guarantee.",
        source_plan_hash=hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest(),
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
            ["scripts/g_offset_study.py","scripts/g_deck_study.py","scripts/calibration_study.py","src/g_offset_models.py",
             "src/g_deck_models.py","src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py","features/folds.csv","requirements.txt"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Protocol changed")
    else:_write_json(path,protocol)
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    dev=old.fold_of!=5;g=g_mask(old.features);reference=np.full(len(old.y),np.nan);reference[dev]=0
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            reference[dev]+=weight/len(member["seeds"])*np.load(old.output/"search"/f"{trial}.npz")["probability"][dev]
    predictions={i:np.full(len(old.y),np.nan) for i in range(len(PROFILES))}
    for fold,(ti,vi) in enumerate(old.splits,1):
        source=PROJECT_ROOT/"outputs/calibration_study_20261029"/f"inner_fold_{fold}.npz"
        ci,p,audit=holdout_calibration_data(old,plan,ti,20261029+fold,source)
        assert not set(ci)&set(vi) and not (old.fold_of[ci]==5).any()
        for index,profile in enumerate(PROFILES):
            calibrator=GOffsetCalibrator(**profile).fit(old.features.iloc[ci],p,old.y[ci],public)
            predictions[index][vi]=calibrator.predict(old.features.iloc[vi],reference[vi])
            _write_json(report/f"profile_{index}_fold_{fold}.json",dict(**calibrator.audit_,**audit,source_cache_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        print(f"Nested G offset fold {fold}/4",flush=True)
    rows=[]
    for index,p in predictions.items():
        np.savez_compressed(output/f"profile_{index}.npz",probability=p)
        for weight in WEIGHTS:
            mixed=reference[dev].copy();mixed[g[dev]]=(1-weight)*mixed[g[dev]]+weight*p[dev][g[dev]]
            rows.append(dict(profile=index,weight=weight,**metrics(old.y[dev],reference[dev],mixed,old.features.loc[dev],old.fold_of[dev])))
    rows.sort(key=lambda r:(-r["accuracy"],r["log_loss"],r["weight"]))
    winner=next(r for r in rows if PROFILES[r["profile"]]["roles"]);control=winner["profile"]-1
    matched=next(r for r in rows if r["profile"]==control and r["weight"]==winner["weight"])
    added=winner["g_accuracy"]-matched["g_accuracy"]
    passed=winner["gain"]>=.0015 and winner["g_gain"]>=.005 and winner["g_improved_folds"]>=3 and added>=.003 and \
        winner["g_log_loss"]<=winner["g_reference_log_loss"]+.005 and winner["log_loss"]<=winner["reference_log_loss"]+.002
    frozen=dict(profile=winner["profile"],control_profile=control,config=PROFILES[winner["profile"]],weight=winner["weight"],
        development=winner,matched_control=matched,added_roles_g_gain=float(added),development_gate_passed=bool(passed))
    _write_json(report/"comparisons.json",rows);_write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,
            reason="Nested G-offset development/ablation gate failed"));print("G offset development gate failed",flush=True);return
    base=PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029";repeats=[];w=winner["weight"]
    for repeat in [1,2]:
        ref=np.full(len(old.y),np.nan);alt=ref.copy();ctl=ref.copy();fold_of=np.zeros(len(old.y),int)
        for fold in range(1,6):
            z=np.load(base/f"repeat_{repeat}_fold_{fold}.npz");ti,vi=z["training"],z["validation"]
            ref[vi]=z["probability"]@np.array(plan["weights"]);fold_of[vi]=fold
            ci,p,_=holdout_calibration_data(old,plan,ti,20261220+repeat*10+fold,output/f"repeat_{repeat}_inner_{fold}.npz")
            for index,dest in [(winner["profile"],alt),(control,ctl)]:
                c=GOffsetCalibrator(**PROFILES[index]).fit(old.features.iloc[ci],p,old.y[ci],public)
                dest[vi]=corrected_probability(c,old.features.iloc[vi],ref[vi],w)
            print(f"G offset repeat {repeat} fold {fold}/5",flush=True)
        r=metrics(old.y,ref,alt,old.features,fold_of);c=metrics(old.y,ref,ctl,old.features,fold_of)
        repeats.append(dict(repeat=repeat,**r,control=c,added_roles_g_gain=r["g_accuracy"]-c["g_accuracy"]))
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=fold_of,reference=ref,alternative=alt,control=ctl)),report/f"repeat_{repeat}_oof.csv")
    passed=np.mean([r["gain"] for r in repeats])>=.001 and sum(r["g_improved_folds"] for r in repeats)>=6 and \
        np.mean([r["added_roles_g_gain"] for r in repeats])>=.003 and all(r["gain"]>=0 and r["g_log_loss"]<=r["g_reference_log_loss"]+.005 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in repeats)
    sensitivity=dict(passed=bool(passed),repeats=repeats,mean_gain=float(np.mean([r["gain"] for r in repeats])))
    _write_json(report/"sensitivity.json",sensitivity)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
            reason="Frozen G-offset repeated-partition gate failed"));return
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib");members=[];oof=np.zeros(len(old.y));test=np.zeros(len(old.raw_test));identity=IdentityTransform()
    for block,fold in enumerate([5,1,2,3,4]):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        ci,p,_=holdout_calibration_data(old,plan,ti,20261320+fold,output/f"final_inner_{fold}.npz")
        c=GOffsetCalibrator(**frozen["config"]).fit(old.features.iloc[ci],p,old.y[ci],public)
        children=[dict(m,weight=m["weight"]*5) for m in bundle["members"][block*9:block*9+9]]
        model=GOffsetFoldModel(children,c,w)
        oof[vi]=model.predict_proba(old.features.iloc[vi])[:,1];test+=model.predict_proba(old.test_features)[:,1]/5
        members.append(dict(weight=.2,builder=identity,encoder=identity,model=model))
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5,frozen_choice=frozen),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,test,pp.load_submission_template(),.5);_write_csv(submission,output/"submission_goffset4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test,prediction=test>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof)),report/"oof_predictions.csv")
    original=pd.read_csv(old.output/"test_predictions.csv",float_precision="round_trip");gt=g_mask(old.test_features)
    np.testing.assert_allclose(test[~gt],original.probability.to_numpy()[~gt],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(test[~gt]>=.5,original.prediction.to_numpy()[~gt])
    reload,check=predict_ensemble(modelpath);np.testing.assert_allclose(reload,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    from sklearn.metrics import accuracy_score
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),g_canonical_accuracy=float(accuracy_score(old.y[g],oof[g]>=.5)),
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=original.prediction.to_numpy())),non_g_test_labels_unchanged=True,
        fold_wrappers=5,original_members=45,calibrators=5,reload_max_error=float(np.max(np.abs(reload-test)))))


if __name__=="__main__":main()
