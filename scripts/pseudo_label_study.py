"""严格按组验证的高置信软伪标签对照；教师排除整个外层验证组。"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import GroupKFold

from src.optimization import SearchRunner,config_id,predict_ensemble
from src.pseudo_labels import fit_student
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="pseudo_label_study_20261029"


def predict(member,frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:,1]


def main():
    old=SearchRunner(threads=4);report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    profiles=[dict(feature_set=m["config"]["feature_set"],confidence=c,pseudo_weight=.25,teacher_config=m["config"])
              for m in plan["members"][:2] for c in [.95,.99]]
    protocol=dict(name=NAME,profiles=profiles,weights=[.25,.5,1.],threshold=.5,seeds=[42,2026,103],
        development_folds=[1,2,3,4],excluded_fold=5,sensitivity_seed=20261105,
        gate="Three-seed dev gain >=.001, >=3/4 folds improved, logloss deterioration <=.002. Then changed-partition gain >=0 and logloss deterioration <=.002. No retuning.",
        pseudo_target="Unlabeled public test features only. Soft teacher probabilities selected by fixed confidence and weight .25; no actual test labels or external solutions",
        teacher="Original frozen 9-model blend fitted only on allowed outer training groups; validation groups are wholly excluded",
        caveat="Historical folds, not blind validation. Changed partitions check sensitivity. CrossEntropy plus augmentation is evaluated as a whole; no causal attribution to pseudo-labeling alone",
        source_plan_hash=hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest(),
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
                 ["scripts/pseudo_label_study.py","src/pseudo_labels.py","src/optimization.py","src/competition_features.py","src/preprocess.py","features/folds.csv"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("协议变化，不混用缓存")
    else:_write_json(path,protocol)
    dev=old.fold_of!=5;reference=np.full(len(old.y),np.nan);reference[dev]=0
    for weight,m in zip(plan["weights"],plan["members"]):
        for seed in m["seeds"]:
            trial=config_id(dict(m["config"],seed=seed),"confirmation",[1,2,3,4])
            p=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            assert np.isnan(p[~dev]).all();reference[dev]+=weight/len(m["seeds"])*p[dev]
    base_acc=float(accuracy_score(old.y[dev],reference[dev]>=.5));base_loss=float(log_loss(old.y[dev],reference[dev]))
    np.testing.assert_allclose(base_acc,plan["development_accuracy"],atol=1e-14)

    def teacher(ti,vi,name):
        path=output/f"teacher_{name}.npz"
        assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
        if path.exists():
            d=np.load(path);np.testing.assert_array_equal(d["training"],ti);np.testing.assert_array_equal(d["validation"],vi)
            return d["validation_probability"],d["test_probability"]
        v,t,_=old.fit_partition(plan,ti,vi)
        np.savez_compressed(path,training=ti,validation=vi,validation_probability=v,test_probability=t)
        return v,t

    def evaluate(index,seed):
        path=output/f"profile_{index}_seed_{seed}.npz";record=report/f"profile_{index}_seed_{seed}.json"
        if path.exists() and record.exists():return np.load(path)["probability"]
        result=np.full(len(old.y),np.nan);audits=[];scores=[]
        for fold,(ti,vi) in enumerate(old.splits,1):
            assert not (old.fold_of[ti]==5).any()
            reference_check,tp=teacher(ti,vi,f"development_{fold}")
            np.testing.assert_allclose(reference_check,reference[vi],atol=1e-12,rtol=0)
            member=fit_student(profiles[index],seed,old.features.iloc[ti],old.y[ti],old.test_features,tp)
            result[vi]=predict(member,old.features.iloc[vi]);audits.append(dict(fold=fold,**member["pseudo_audit"]))
            scores.append(float(accuracy_score(old.y[vi],result[vi]>=.5)))
            print(f"Pseudo profile {index} seed {seed} fold {fold}: {scores[-1]:.4f}, pseudo rows {audits[-1]['pseudo_rows']}",flush=True)
        assert np.isnan(result[~dev]).all()
        np.savez_compressed(path,probability=result)
        _write_json(record,dict(profile=index,config=profiles[index],seed=seed,fold_scores=scores,
            accuracy=float(accuracy_score(old.y[dev],result[dev]>=.5)),log_loss=float(log_loss(old.y[dev],result[dev])),pseudo_audits=audits))
        return result

    def choices(predictions):
        rows=[]
        for index,pred in predictions.items():
            for w in protocol["weights"]:
                p=(1-w)*reference[dev]+w*pred[dev];gains=[]
                for fold in range(1,5):
                    mask=old.fold_of[dev]==fold
                    gains.append(float(accuracy_score(old.y[dev][mask],p[mask]>=.5)-accuracy_score(old.y[dev][mask],reference[dev][mask]>=.5)))
                rows.append(dict(profile=index,weight=w,accuracy=float(accuracy_score(old.y[dev],p>=.5)),
                    log_loss=float(log_loss(old.y[dev],p)),fold_gains=gains,improved_folds=int(np.sum(np.array(gains)>0))))
        return sorted(rows,key=lambda r:(-r["accuracy"],r["log_loss"],r["weight"]))
    first={i:evaluate(i,42) for i in range(len(profiles))};initial=choices(first)
    top=[]
    for row in initial:
        if row["profile"] not in top:top.append(row["profile"])
        if len(top)==2:break
    repeat={i:np.mean([first[i],evaluate(i,2026),evaluate(i,103)],axis=0) for i in top}
    scans=choices(repeat);best=scans[0]
    passed=best["accuracy"]>=base_acc+.001 and best["improved_folds"]>=3 and best["log_loss"]<=base_loss+.002
    frozen=dict(**best,config=profiles[best["profile"]],seeds=protocol["seeds"],incumbent_accuracy=base_acc,
        incumbent_log_loss=base_loss,initial_gate_passed=bool(passed))
    _write_json(report/"single_seed_mixtures.json",initial);_write_json(report/"three_seed_mixtures.json",scans)
    _write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,reason="Initial gate failed"));return
    indices=np.flatnonzero(dev);ref=np.full(len(old.y),np.nan);alt=ref.copy()
    splitter=GroupKFold(4,shuffle=True,random_state=protocol["sensitivity_seed"])
    for fold,(a,b) in enumerate(splitter.split(indices,old.y[dev],old.features.GroupId[dev]),1):
        ti,vi=indices[a],indices[b];q,tp=teacher(ti,vi,f"sensitivity_{fold}")
        students=[fit_student(frozen["config"],seed,old.features.iloc[ti],old.y[ti],old.test_features,tp) for seed in frozen["seeds"]]
        p=np.mean([predict(m,old.features.iloc[vi]) for m in students],axis=0)
        ref[vi]=q;alt[vi]=(1-best["weight"])*q+best["weight"]*p
        print(f"Pseudo changed-partition sensitivity {fold}/4",flush=True)
    sensitivity=dict(reference_accuracy=float(accuracy_score(old.y[dev],ref[dev]>=.5)),alternative_accuracy=float(accuracy_score(old.y[dev],alt[dev]>=.5)),
        reference_log_loss=float(log_loss(old.y[dev],ref[dev])),alternative_log_loss=float(log_loss(old.y[dev],alt[dev])))
    passed=sensitivity["alternative_accuracy"]>=sensitivity["reference_accuracy"] and sensitivity["alternative_log_loss"]<=sensitivity["reference_log_loss"]+.002
    sensitivity["passed"]=bool(passed);_write_json(report/"sensitivity.json",sensitivity)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,reason="Changed-partition gate failed"));return
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report/"oof_predictions.csv")
    old_test=pd.read_csv(old.output/"test_predictions.csv")
    w=best["weight"];members=[dict(m,weight=m["weight"]*(1-w)) for m in bundle["members"] if w<1]
    oof=(1-w)*canonical.probability.to_numpy();test=(1-w)*old_test.probability.to_numpy()
    for block,fold in enumerate([5,1,2,3,4]):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        children=bundle["members"][block*9:block*9+9]
        q=sum(m["weight"]*5*predict(m,old.features.iloc[vi]) for m in children)
        np.testing.assert_allclose(q,canonical.probability.to_numpy()[vi],atol=1e-12,rtol=0)
        tp=sum(m["weight"]*5*predict(m,old.test_features) for m in children)
        for seed in frozen["seeds"]:
            student=fit_student(frozen["config"],seed,old.features.iloc[ti],old.y[ti],old.test_features,tp)
            oof[vi]+=w/3*predict(student,old.features.iloc[vi]);test+=w/15*predict(student,old.test_features)
            members.append(dict(student,weight=w/15))
        print(f"Pseudo frozen refit fold {fold}/5",flush=True)
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5,frozen_choice=frozen),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,test,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_pseudo4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test,prediction=test>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    reload,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reload,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),models=len(members),
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=old_test.prediction.to_numpy())),reload_max_error=float(np.max(np.abs(reload-test)))))


if __name__=="__main__":main()
