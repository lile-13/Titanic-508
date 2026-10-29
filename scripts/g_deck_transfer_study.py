"""G-upweighted global learners; G-only routing and role-feature ablation."""
import hashlib
import json
import os
import time

for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ.setdefault(name,"4")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss

from src import preprocess as pp
from src.g_transfer_models import PROFILES,PARAMETERS,fit_member
from src.g_deck_models import predict_specialist,blend_g,g_mask,GOnlyFoldModel
from src.optimization import SearchRunner,config_id,predict_ensemble
from src.probability_calibration import IdentityTransform
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission

NAME="g_deck_transfer_20261029"
SEEDS=[42,2026,103]
WEIGHTS=[.25,.5,.75,1.]


def metrics(y,reference,alternative,frame,folds):
    mask=g_mask(frame);outside=~mask
    np.testing.assert_array_equal(reference[outside],alternative[outside])
    gain=[]
    for fold in np.unique(folds):
        f=(folds==fold)&mask
        gain.append(float(accuracy_score(y[f],alternative[f]>=.5)-accuracy_score(y[f],reference[f]>=.5)))
    return dict(accuracy=float(accuracy_score(y,alternative>=.5)),reference_accuracy=float(accuracy_score(y,reference>=.5)),
        gain=float(accuracy_score(y,alternative>=.5)-accuracy_score(y,reference>=.5)),
        log_loss=float(log_loss(y,alternative)),reference_log_loss=float(log_loss(y,reference)),
        g_rows=int(mask.sum()),g_accuracy=float(accuracy_score(y[mask],alternative[mask]>=.5)),
        g_reference_accuracy=float(accuracy_score(y[mask],reference[mask]>=.5)),
        g_gain=float(accuracy_score(y[mask],alternative[mask]>=.5)-accuracy_score(y[mask],reference[mask]>=.5)),
        g_log_loss=float(log_loss(y[mask],alternative[mask])),g_reference_log_loss=float(log_loss(y[mask],reference[mask])),
        g_fold_gains=gain,g_improved_folds=int(np.sum(np.array(gain)>0)),
        non_g_probabilities_unchanged=True)


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    base_report=EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029";base_output=PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029"
    protocol=dict(name=NAME,profiles=PROFILES,parameters=PARAMETERS,seeds=SEEDS,weights=WEIGHTS,threshold=.5,
        scope="Fit on all allowed outer training rows with G sample weight 2 and other sample weight 1; route corrections to raw observed Deck G only. All non-G predictions retain incumbent. No unsafe group Deck propagation.",
        role_features="Other-known CryoSleep share/support, other-adult mean/support, other-child count, age gap and child/adult role; exclude self, use combined public unlabeled features only",
        development_folds=[1,2,3,4],excluded_fold=5,
        selection="Four profiles at seed 42. Select best enriched family by global blend accuracy, logloss, then smaller new weight; repeat that enriched profile and its paired no-role control with all three seeds. Freeze best enriched three-seed weight once; no family fallback after failure.",
        development_gate="Global gain >=.0015; G gain >=.005 and >=3/4 G folds improve; G logloss deterioration <=.005 and global <=.002; enriched G accuracy exceeds matched control at same weight by >=.003. Non-G predictions identical.",
        repeat_gate="Two fixed cached GroupKFold repetitions; mean global gain >=.001, both global gains nonnegative, G improves in >=6/10 folds; each G logloss deterioration <=.005 and global <=.002. Mean G enriched-versus-control gain >=.003. No retuning.",
        repeat_seeds=[20261029,20261106],
        caveat="Follow-up to the G-only fitting study: limited G sample size motivates shared training. All rows and previous repartitions already inspected. This is adaptive development and stability checking, not a new blind test or guarantee of Kaggle gain.",
        source_plan_hash=hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest(),
        baseline_protocol_sha256=hashlib.sha256((base_report/"protocol.json").read_bytes()).hexdigest(),
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"],
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
            ["scripts/g_deck_transfer_study.py","src/g_transfer_models.py","src/g_deck_models.py","src/focused_features.py","src/competition_features.py",
             "src/optimization.py","src/preprocess.py","src/probability_calibration.py","features/folds.csv","requirements.txt"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Protocol changed; use a new run")
    else:_write_json(path,protocol)
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public
    dev=old.fold_of!=5;g=g_mask(old.features);reference=np.full(len(old.y),np.nan);reference[dev]=0
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            reference[dev]+=weight/len(member["seeds"])*np.load(old.output/"search"/f"{trial}.npz")["probability"][dev]
    np.testing.assert_allclose(accuracy_score(old.y[dev],reference[dev]>=.5),plan["development_accuracy"],atol=1e-14)

    def evaluate(index,seed):
        cache=output/f"profile_{index}_seed_{seed}.npz";record=report/f"profile_{index}_seed_{seed}.json"
        if cache.exists() and record.exists():return np.load(cache)["probability"]
        started=time.perf_counter();p=np.full(len(old.y),np.nan)
        for fold,(ti,vi) in enumerate(old.splits,1):
            part=output/f"profile_{index}_seed_{seed}_fold_{fold}.npz"
            assert not (old.fold_of[ti]==5).any() and not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
            if part.exists():
                data=np.load(part);np.testing.assert_array_equal(data["training"],ti);np.testing.assert_array_equal(data["validation"],vi)
                p[vi]=data["probability"]
            else:
                m=fit_member(PROFILES[index],seed,old.features.iloc[ti],old.y[ti],public)
                p[vi]=predict_specialist(m,old.features.iloc[vi])
                np.savez_compressed(part,training=ti,validation=vi,probability=p[vi])
            print(f"G profile {index} seed {seed} fold {fold}/4: {accuracy_score(old.y[vi][g[vi]],p[vi][g[vi]]>=.5):.4f}",flush=True)
        assert np.isnan(p[~(dev&g)]).all() and np.isfinite(p[dev&g]).all()
        np.savez_compressed(cache,probability=p)
        _write_json(record,dict(profile=index,config=PROFILES[index],seed=seed,g_rows=int((dev&g).sum()),
            g_accuracy=float(accuracy_score(old.y[dev&g],p[dev&g]>=.5)),g_log_loss=float(log_loss(old.y[dev&g],p[dev&g])),seconds=time.perf_counter()-started))
        return p

    def choices(predictions):
        rows=[]
        for index,p in predictions.items():
            for weight in WEIGHTS:
                mixed=blend_g(reference[dev],p[dev],old.features.loc[dev],weight)
                rows.append(dict(profile=index,new_weight=weight,**metrics(old.y[dev],reference[dev],mixed,old.features.loc[dev],old.fold_of[dev])))
        return sorted(rows,key=lambda r:(-r["accuracy"],r["log_loss"],r["new_weight"]))

    first={i:evaluate(i,42) for i in range(4)};initial=choices(first)
    _write_json(report/"single_seed_mixtures.json",initial)
    chosen=next(r["profile"] for r in initial if PROFILES[r["profile"]]["roles"]);control=chosen-1
    seeded={i:np.mean([first[i],evaluate(i,2026),evaluate(i,103)],axis=0) for i in [chosen,control]}
    comparisons=choices(seeded);winner=next(r for r in comparisons if r["profile"]==chosen)
    matched=next(r for r in comparisons if r["profile"]==control and r["new_weight"]==winner["new_weight"])
    added=winner["g_accuracy"]-matched["g_accuracy"]
    passed=winner["gain"]>=.0015 and winner["g_gain"]>=.005 and winner["g_improved_folds"]>=3 and \
        winner["g_log_loss"]<=winner["g_reference_log_loss"]+.005 and winner["log_loss"]<=winner["reference_log_loss"]+.002 and added>=.003
    frozen=dict(profile=chosen,control_profile=control,config=PROFILES[chosen],weight=winner["new_weight"],seeds=SEEDS,
        development=winner,matched_control=matched,added_roles_g_gain=float(added),development_gate_passed=bool(passed),threshold=.5)
    _write_json(report/"three_seed_mixtures.json",comparisons);_write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,
            reason="G-only role-feature development/ablation gate failed; retain original ensemble"))
        print("G-only development gate failed; original ensemble retained",flush=True);return
    w=frozen["weight"];repetitions=[]
    for repeat in [1,2]:
        ref=np.full(len(old.y),np.nan);alternative=ref.copy();ablation=ref.copy();fold_of=np.zeros(len(old.y),int)
        for fold in range(1,6):
            base=np.load(base_output/f"repeat_{repeat}_fold_{fold}.npz");ti,vi=base["training"],base["validation"]
            assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
            ref[vi]=base["probability"]@np.array(plan["weights"])
            for index,destination in [(chosen,alternative),(control,ablation)]:
                cache=output/f"profile_{index}_repeat_{repeat}_fold_{fold}.npz"
                if cache.exists():
                    data=np.load(cache);np.testing.assert_array_equal(data["training"],ti);np.testing.assert_array_equal(data["validation"],vi);q=data["probability"]
                else:
                    q=np.mean([predict_specialist(fit_member(PROFILES[index],seed,old.features.iloc[ti],old.y[ti],public),old.features.iloc[vi]) for seed in SEEDS],axis=0)
                    np.savez_compressed(cache,training=ti,validation=vi,probability=q)
                destination[vi]=blend_g(ref[vi],q,old.features.iloc[vi],w)
            fold_of[vi]=fold
            print(f"G-only frozen repeat {repeat} fold {fold}/5",flush=True)
        r=metrics(old.y,ref,alternative,old.features,fold_of);c=metrics(old.y,ref,ablation,old.features,fold_of)
        repetitions.append(dict(repeat=repeat,**r,control=c,added_roles_g_gain=r["g_accuracy"]-c["g_accuracy"]))
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=fold_of,reference=ref,alternative=alternative,control=ablation)),report/f"repeat_{repeat}_oof.csv")
    passed=np.mean([r["gain"] for r in repetitions])>=.001 and sum(r["g_improved_folds"] for r in repetitions)>=6 and \
        np.mean([r["added_roles_g_gain"] for r in repetitions])>=.003 and all(r["gain"]>=0 and r["g_log_loss"]<=r["g_reference_log_loss"]+.005 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in repetitions)
    sensitivity=dict(passed=bool(passed),repeats=repetitions,mean_gain=float(np.mean([r["gain"] for r in repetitions])),
        interpretation="Historical repeated-partition sensitivity and same-weight role ablation; not independent blind confirmation")
    _write_json(report/"sensitivity.json",sensitivity)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
            reason="Frozen G-only repeated-partition/ablation gate failed"));print("G-only repeat gate failed",flush=True);return
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report/"oof_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    old_test=pd.read_csv(old.output/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(canonical.PassengerId,old.raw.PassengerId);np.testing.assert_array_equal(old_test.PassengerId,old.raw_test.PassengerId)
    oof=np.zeros(len(old.y));test=np.zeros(len(old.raw_test));members=[];identity=IdentityTransform()
    for block,fold in enumerate([5,1,2,3,4]):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        children=[dict(m,weight=m["weight"]*5) for m in bundle["members"][block*9:block*9+9]]
        specialists=[fit_member(PROFILES[chosen],seed,old.features.iloc[ti],old.y[ti],public) for seed in SEEDS]
        model=GOnlyFoldModel(children,specialists,w)
        oof[vi]=model.predict_proba(old.features.iloc[vi])[:,1];test+=model.predict_proba(old.test_features)[:,1]/5
        members.append(dict(weight=.2,builder=identity,encoder=identity,model=model))
        print(f"G-only final frozen fold {fold}/5",flush=True)
    np.testing.assert_allclose(oof[~g],canonical.probability.to_numpy()[~g],atol=1e-12,rtol=0)
    gt=g_mask(old.test_features);np.testing.assert_allclose(test[~gt],old_test.probability.to_numpy()[~gt],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(test[~gt]>=.5,old_test.prediction.to_numpy()[~gt])
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5,frozen_choice=frozen),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,test,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_gtransfer4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test,prediction=test>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof)),report/"oof_predictions.csv")
    reloaded,check=predict_ensemble(modelpath);np.testing.assert_allclose(reloaded,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),g_canonical_accuracy=float(accuracy_score(old.y[g],oof[g]>=.5)),
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=old_test.prediction.to_numpy())),
        non_g_test_labels_unchanged=True,fold_wrappers=5,original_members=45,specialist_members=15,
        reload_max_error=float(np.max(np.abs(reloaded-test)))))


if __name__=="__main__":main()
