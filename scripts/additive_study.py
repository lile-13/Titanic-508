"""Finite EBM screen, repeated-group verification and conditional candidate refit."""
import hashlib
import json
import os
import time

for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(name,"4")

import interpret
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss

from src.additive_models import PROFILES,PARAMETERS,fit_member
from src.optimization import SearchRunner,config_id,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="additive_study_20261029"
SEEDS=[42,2026,103]
WEIGHTS=[.25,.5,.75,1.]


def predict(member,frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:,1]


def metrics(y,reference,alternative,folds):
    changes=[float(accuracy_score(y[folds==f],alternative[folds==f]>=.5)-
        accuracy_score(y[folds==f],reference[folds==f]>=.5)) for f in np.unique(folds)]
    return dict(reference_accuracy=float(accuracy_score(y,reference>=.5)),
        accuracy=float(accuracy_score(y,alternative>=.5)),
        gain=float(accuracy_score(y,alternative>=.5)-accuracy_score(y,reference>=.5)),
        reference_log_loss=float(log_loss(y,reference)),log_loss=float(log_loss(y,alternative)),
        fold_gains=changes,improved_folds=int(np.sum(np.array(changes)>0)))


def main():
    old=SearchRunner(threads=4)
    report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    baseline_report=EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029"
    baseline_output=PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029"
    base_protocol=json.loads((baseline_report/"protocol.json").read_text(encoding="utf-8"))
    protocol=dict(name=NAME,profiles=PROFILES,parameters=PARAMETERS,interpret_version=interpret.__version__,
        development_folds=[1,2,3,4],excluded_fold=5,seeds=SEEDS,weights=WEIGHTS,threshold=.5,
        selection="Four single-seed profiles; top two by best blend accuracy then logloss repeated with three seeds. Freeze best three-seed blend once.",
        gate="Development gain >=.0015, >=3/4 folds improved, logloss deterioration <=.002. Then two cached repeated GroupKFold checks: mean gain >=.001, both nonnegative, >=6/10 folds improved, logloss deterioration <=.002 per repeat. No adjustment after failure.",
        internal_validation="Six GroupShuffleSplit bags of outer training rows only; entire groups held out at fraction .15; no row-random early stopping",
        repeated_split_seeds=base_protocol["split_seeds"],
        caveat="All historical rows inspected. Screening and repartition checks are adaptive development, not independent blind validation. A passed local gate does not guarantee a Kaggle increase.",
        public_context="Combined public unlabeled features for counts and unique HomePlanet lookups; no Transported in context",
        source_plan_hash=hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest(),
        baseline_protocol_sha256=hashlib.sha256((baseline_report/"protocol.json").read_bytes()).hexdigest(),
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"],
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
            ["scripts/additive_study.py","src/additive_models.py","src/focused_features.py","src/competition_features.py",
             "src/optimization.py","src/preprocess.py","features/folds.csv","requirements-ebm.txt","requirements.txt"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Protocol changed; use a new run")
    else:_write_json(path,protocol)
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public
    dev=old.fold_of!=5
    incumbent=np.full(len(old.y),np.nan);incumbent[dev]=0
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            p=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            assert np.isnan(p[~dev]).all()
            incumbent[dev]+=weight/len(member["seeds"])*p[dev]
    np.testing.assert_allclose(accuracy_score(old.y[dev],incumbent[dev]>=.5),plan["development_accuracy"],atol=1e-14)

    def evaluate(index,seed):
        cache=output/f"profile_{index}_seed_{seed}.npz";record=report/f"profile_{index}_seed_{seed}.json"
        if cache.exists() and record.exists():return np.load(cache)["probability"]
        started=time.perf_counter();p=np.full(len(old.y),np.nan);audits=[]
        for fold,(ti,vi) in enumerate(old.splits,1):
            foldcache=output/f"profile_{index}_seed_{seed}_fold_{fold}.npz"
            foldrecord=report/f"profile_{index}_seed_{seed}_fold_{fold}.json"
            assert not (old.fold_of[ti]==5).any() and not (old.fold_of[vi]==5).any()
            assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
            if foldcache.exists() and foldrecord.exists():
                data=np.load(foldcache)
                np.testing.assert_array_equal(data["training"],ti);np.testing.assert_array_equal(data["validation"],vi)
                p[vi]=data["probability"];audit=json.loads(foldrecord.read_text(encoding="utf-8"))
            else:
                member=fit_member(PROFILES[index],seed,old.features.iloc[ti],old.y[ti],public)
                p[vi]=predict(member,old.features.iloc[vi])
                audit=dict(inner_validation=member["model"].group_validation_,
                    best_iteration=member["model"].best_iteration_.tolist())
                np.savez_compressed(foldcache,training=ti,validation=vi,probability=p[vi])
                _write_json(foldrecord,audit)
            audits.append(audit)
            print(f"EBM profile {index} seed {seed} fold {fold}/4: {accuracy_score(old.y[vi],p[vi]>=.5):.4f}",flush=True)
        assert np.isfinite(p[dev]).all() and np.isnan(p[~dev]).all()
        np.savez_compressed(cache,probability=p)
        _write_json(record,dict(profile=index,seed=seed,config=PROFILES[index],
            accuracy=float(accuracy_score(old.y[dev],p[dev]>=.5)),log_loss=float(log_loss(old.y[dev],p[dev])),
            seconds=time.perf_counter()-started,fold_audits=audits))
        return p

    def choices(predictions):
        rows=[]
        for index,p in predictions.items():
            for weight in WEIGHTS:
                row=metrics(old.y[dev],incumbent[dev],(1-weight)*incumbent[dev]+weight*p[dev],old.fold_of[dev])
                rows.append(dict(profile=index,new_weight=weight,**row))
        return sorted(rows,key=lambda r:(-r["accuracy"],r["log_loss"],r["new_weight"]))

    first={i:evaluate(i,42) for i in range(len(PROFILES))};initial=choices(first)
    _write_json(report/"single_seed_mixtures.json",initial)
    top=list(dict.fromkeys(r["profile"] for r in initial))[:2]
    repeated={i:np.mean([first[i],evaluate(i,2026),evaluate(i,103)],axis=0) for i in top}
    rows=choices(repeated);winner=rows[0]
    passed=winner["gain"]>=.0015 and winner["improved_folds"]>=3 and winner["log_loss"]<=winner["reference_log_loss"]+.002
    frozen=dict(profile=winner["profile"],config=PROFILES[winner["profile"]],weight=winner["new_weight"],
        development=winner,development_gate_passed=bool(passed),threshold=.5,seeds=SEEDS)
    _write_json(report/"three_seed_mixtures.json",rows);_write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,
            reason="Development gate failed; retain observed Kaggle best 0.81108"))
        print("EBM development gate failed; original ensemble retained",flush=True);return
    index=frozen["profile"];w=frozen["weight"];repetitions=[]
    for repeat in [1,2]:
        reference=np.full(len(old.y),np.nan);alternative=reference.copy();fold_of=np.zeros(len(old.y),int)
        for fold in range(1,6):
            baseline=np.load(baseline_output/f"repeat_{repeat}_fold_{fold}.npz")
            ti,vi=baseline["training"],baseline["validation"]
            assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
            cache=output/f"repeat_{repeat}_fold_{fold}.npz"
            if cache.exists():
                data=np.load(cache)
                np.testing.assert_array_equal(data["training"],ti);np.testing.assert_array_equal(data["validation"],vi)
                probability=data["probability"]
            else:
                probability=np.mean([predict(fit_member(frozen["config"],seed,old.features.iloc[ti],old.y[ti],public),old.features.iloc[vi]) for seed in SEEDS],axis=0)
                np.savez_compressed(cache,training=ti,validation=vi,probability=probability)
            reference[vi]=baseline["probability"]@np.array(plan["weights"])
            alternative[vi]=(1-w)*reference[vi]+w*probability;fold_of[vi]=fold
            print(f"EBM frozen sensitivity repeat {repeat} fold {fold}/5",flush=True)
        assert np.isfinite(reference).all() and np.isfinite(alternative).all()
        repetitions.append(dict(repeat=repeat,**metrics(old.y,reference,alternative,fold_of)))
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=fold_of,reference=reference,alternative=alternative)),report/f"repeat_{repeat}_oof.csv")
    passed=np.mean([r["gain"] for r in repetitions])>=.001 and sum(r["improved_folds"] for r in repetitions)>=6 and all(
        r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in repetitions)
    sensitivity=dict(passed=bool(passed),repeats=repetitions,mean_gain=float(np.mean([r["gain"] for r in repetitions])),
        interpretation="Paired changed-partition sensitivity on historical rows, not an independent test")
    _write_json(report/"sensitivity.json",sensitivity)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
            reason="Frozen repeated-partition gate failed; retain incumbent"))
        print("EBM repeated-partition gate failed; original ensemble retained",flush=True);return
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=[dict(m,weight=m["weight"]*(1-w)) for m in bundle["members"] if w<1]
    canonical=pd.read_csv(old.report/"oof_predictions.csv",dtype={"PassengerId":str})
    old_test=pd.read_csv(old.output/"test_predictions.csv",dtype={"PassengerId":str})
    np.testing.assert_array_equal(canonical.PassengerId,old.raw.PassengerId)
    np.testing.assert_array_equal(old_test.PassengerId,old.raw_test.PassengerId)
    oof=(1-w)*canonical.probability.to_numpy();test=(1-w)*old_test.probability.to_numpy()
    for fold in range(1,6):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        for seed in SEEDS:
            member=fit_member(frozen["config"],seed,old.features.iloc[ti],old.y[ti],public)
            oof[vi]+=w/3*predict(member,old.features.iloc[vi]);test+=w/15*predict(member,old.test_features)
            members.append(dict(member,weight=w/15))
        print(f"EBM frozen refit fold {fold}/5",flush=True)
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5,frozen_choice=frozen),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,test,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_additive4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test,prediction=test>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof)),report/"oof_predictions.csv")
    reload,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reload,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),members=len(members),
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=old_test.prediction.to_numpy())),
        reload_max_error=float(np.max(np.abs(reload-test)))))


if __name__=="__main__":main()
