"""复用新分组基准，检验两个预先固定的正则化替代；不再调参。"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss

from src.domain_models import PROFILES,fit_member
from src.optimization import SearchRunner,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="paired_regularization_20261029"
CHOICES=[2,9]
WEIGHT=.25


def predict(member,frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:,1]


def main():
    old=SearchRunner(threads=4);report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    baseline_report=EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029"
    baseline_output=PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029"
    base_protocol=json.loads((baseline_report/"protocol.json").read_text(encoding="utf-8"))
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    protocol=dict(name=NAME,source_choices=CHOICES,profiles=[PROFILES[i] for i in CHOICES],blend_weight=WEIGHT,
        threshold=.5,seeds=[42,2026,103],split_seeds=base_protocol["split_seeds"],folds=5,
        choice_rationale="Two previously evaluated fixed alternatives: strongly regularized depth-3 monotone XGBoost and G/E regional LightGBM with improved historical probability loss. Old development gates remain failed; this follow-up checks paired repartition stability without tuning.",
        gate="Mean repeated OOF gain >=.001, both repeats nonnegative, >=6/10 folds improved, logloss deterioration <=.002 on each repeat. If two pass, highest mean gain has priority; one submission only.",
        caveat="All rows historically inspected. New partitions do not create an independent blind holdout.",
        baseline_protocol_sha256=hashlib.sha256((baseline_report/"protocol.json").read_bytes()).hexdigest(),
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
            ["scripts/paired_regularization_check.py","scripts/confirm_fixed_weights.py","src/domain_models.py","src/focused_features.py","src/optimization.py","src/competition_features.py","src/preprocess.py"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("协议改变，拒绝混用缓存")
    else:_write_json(path,protocol)
    if not all((baseline_output/f"repeat_{r}_fold_{f}.npz").exists() for r in [1,2] for f in range(1,6)):
        print("Protocol frozen; paired checks will run after all baseline partitions finish",flush=True);return
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    rows=[];comparison=[]
    for choice in CHOICES:
        repetitions=[];folds=[]
        for repeat in [1,2]:
            ref=np.full(len(old.y),np.nan);alt=ref.copy()
            for fold in range(1,6):
                baseline=np.load(baseline_output/f"repeat_{repeat}_fold_{fold}.npz")
                ti,vi=baseline["training"],baseline["validation"]
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                reference=baseline["probability"]@np.array(plan["weights"])
                cache=output/f"choice_{choice}_repeat_{repeat}_fold_{fold}.npz"
                if cache.exists():
                    data=np.load(cache);np.testing.assert_array_equal(data["training"],ti);np.testing.assert_array_equal(data["validation"],vi)
                    p=data["probability"]
                else:
                    p=np.mean([predict(fit_member(PROFILES[choice],seed,old.features.iloc[ti],old.y[ti],public),old.features.iloc[vi]) for seed in protocol["seeds"]],axis=0)
                    np.savez_compressed(cache,training=ti,validation=vi,probability=p)
                ref[vi]=reference;alt[vi]=(1-WEIGHT)*reference+WEIGHT*p
                delta=float(accuracy_score(old.y[vi],alt[vi]>=.5)-accuracy_score(old.y[vi],reference>=.5))
                folds.append(delta)
                comparison.append(dict(choice=choice,repeat=repeat,fold=fold,gain=delta,group_overlap=0))
                print(f"Paired fixed alternative {choice} repeat {repeat} fold {fold}: delta {delta:+.4f}",flush=True)
            assert np.isfinite(ref).all() and np.isfinite(alt).all()
            r=dict(repeat=repeat,reference_accuracy=float(accuracy_score(old.y,ref>=.5)),alternative_accuracy=float(accuracy_score(old.y,alt>=.5)),
                reference_log_loss=float(log_loss(old.y,ref)),alternative_log_loss=float(log_loss(old.y,alt)))
            r["gain"]=r["alternative_accuracy"]-r["reference_accuracy"];repetitions.append(r)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,reference=ref,alternative=alt)),report/f"choice_{choice}_repeat_{repeat}_oof.csv")
        gain=float(np.mean([r["gain"] for r in repetitions]));improved=int(np.sum(np.array(folds)>0))
        passed=gain>=.001 and improved>=6 and all(r["gain"]>=0 and r["alternative_log_loss"]<=r["reference_log_loss"]+.002 for r in repetitions)
        rows.append(dict(choice=choice,config=PROFILES[choice],mean_gain=gain,improved_folds=improved,repeats=repetitions,passed=bool(passed)))
    _write_json(report/"comparisons.json",rows);_write_json(report/"fold_differences.json",comparison)
    qualified=sorted([r for r in rows if r["passed"]],key=lambda r:-r["mean_gain"])
    if not qualified:
        _write_json(report/"summary.json",dict(status="no_candidate",comparisons=rows,kaggle_score=None,
            reason="Fixed regularized alternatives did not pass paired repeated-partition gate"));return
    frozen=qualified[0];_write_json(report/"frozen_choice.json",frozen)
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report/"oof_predictions.csv");old_test=pd.read_csv(old.output/"test_predictions.csv")
    members=[dict(m,weight=m["weight"]*(1-WEIGHT)) for m in bundle["members"]]
    oof=(1-WEIGHT)*canonical.probability.to_numpy();test=(1-WEIGHT)*old_test.probability.to_numpy()
    for fold in range(1,6):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        for seed in protocol["seeds"]:
            m=fit_member(frozen["config"],seed,old.features.iloc[ti],old.y[ti],public)
            oof[vi]+=WEIGHT/3*predict(m,old.features.iloc[vi]);test+=WEIGHT/15*predict(m,old.test_features)
            members.append(dict(m,weight=WEIGHT/15))
        print(f"Paired regularized frozen refit {fold}/5",flush=True)
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5,frozen_choice=frozen),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,test,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_regularized4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test,prediction=test>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    reload,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reload,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,comparisons=rows,kaggle_score=None,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),models=len(members),
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=old_test.prediction.to_numpy())),reload_max_error=float(np.max(np.abs(reload-test)))))


if __name__=="__main__":main()
