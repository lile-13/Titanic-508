"""One frozen cohort-routing hypothesis, with paired repeated checks and futility stop."""
import hashlib
import json
import os

for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ.setdefault(key,"4")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import GroupShuffleSplit

from src import preprocess as pp
from src.competition_features import FeatureBuilder,InputEncoder
from src.g_deck_models import fit_member,predict_specialist,g_mask
from src.g_offset_models import GOffsetCalibrator
from src.g_routing import cohort_masks,route_probability,GRoutedFoldModel
from src.optimization import SearchRunner,estimator,fit_model,predict_ensemble
from src.probability_calibration import IdentityTransform
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission

NAME="g_cohort_routing_20261029"
SEEDS=[42,2026,103]
CHILD=dict(family="catboost",roles=True)
CONTROL=dict(family="catboost",roles=False)
OFFSET=dict(roles=True,ridge=.1)
OFFSET_CONTROL=dict(roles=False,ridge=.1)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y,ref,alt,frame,folds):
    cold,child=cohort_masks(frame);g=g_mask(frame);untouched=~(cold|child)
    np.testing.assert_array_equal(ref[untouched],alt[untouched])
    correct=(ref>=.5)==y;new=(alt>=.5)==y
    gains=[float(new[(folds==f)&g].mean()-correct[(folds==f)&g].mean()) for f in np.unique(folds)]
    return dict(reference_accuracy=float(correct.mean()),accuracy=float(new.mean()),gain=float(new.mean()-correct.mean()),
        net_correct=int(new.sum()-correct.sum()),corrected=int((~correct&new).sum()),new_errors=int((correct&~new).sum()),
        reference_log_loss=float(log_loss(y,ref)),log_loss=float(log_loss(y,alt)),
        g_rows=int(g.sum()),g_reference_accuracy=float(correct[g].mean()),g_accuracy=float(new[g].mean()),
        g_reference_log_loss=float(log_loss(y[g],ref[g])),g_log_loss=float(log_loss(y[g],alt[g])),
        cold_rows=int(cold.sum()),cold_net_correct=int(new[cold].sum()-correct[cold].sum()),
        child_rows=int(child.sum()),child_net_correct=int(new[child].sum()-correct[child].sum()),
        g_fold_gains=gains,g_improved_folds=int(np.sum(np.array(gains)>0)),untouched_predictions_identical=True)


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/NAME;out=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    baseline_report=EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029"
    baseline_out=PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029"
    protocol=dict(name=NAME,child_profile=CHILD,control_child_profile=CONTROL,cold_offset=OFFSET,
        control_cold_offset=OFFSET_CONTROL,seeds=SEEDS,cold_weight=.25,child_weight=.25,threshold=.5,
        branches="Observed G + CryoSleep=True: offset; observed G + CryoSleep=False + known Age<=12: specialist; all other passengers retain reference",
        source_hypothesis="Cohort routing selected from previous developmental error tradeoffs. Constituent studies failed their original gates and remain failed. No new weight, profile, threshold or branch search.",
        evaluation="Reuse two fixed repeated GroupKFold baselines, seeds 20261029/20261106. New routing predictions not used before this protocol; rows, partitions and baseline outcomes historically inspected. This is adaptive paired stability checking, not new independent or blind validation.",
        inner_teacher="25% GroupShuffleSplit inside each outer training set, seed 690001+repeat*10+fold; full frozen three-component/three-seed teacher excludes all calibration groups. Ridge offsets fit only G calibration labels.",
        gate="Both repeats global gain nonnegative; cold and child net correct nonnegative on each repeat; mean global gain >=.001; >=6/10 G folds improve; mean enriched-control G accuracy difference >=.003; global logloss deterioration <=.002 and G <=.005 on each repeat. No retuning.",
        futility="Stop after repeat 1 if global, cold or child net correct is negative, loss limits fail, or zero G folds improve (then 6/10 impossible). No selection of a replacement split.",
        source_plan_hash=sha(old.report/"frozen_plan.json"),baseline_protocol_sha256=sha(baseline_report/"protocol.json"),
        source_choices={n:sha(EXPERIMENTS_DIR/n/"frozen_choice.json") for n in ["g_deck_roles_20261029_v2","g_deck_offset_20261029"]},
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"],
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/g_routing_study.py","src/g_routing.py","src/g_deck_models.py",
            "src/g_offset_models.py","src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py","features/folds.csv","requirements.txt"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Routing protocol changed; use a new run")
    else:_write_json(path,protocol)
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public

    def inner_teacher(training,seed,tag):
        cache=out/f"{tag}_inner.npz"
        if cache.exists():
            z=np.load(cache);fit,cal,p=z["fit_indices"],z["cal_indices"],z["probability"]
        else:
            a,b=next(GroupShuffleSplit(1,test_size=.25,random_state=seed).split(training,groups=old.features.GroupId.iloc[training]))
            fit,cal=training[a],training[b];p=np.zeros(len(cal))
            assert not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[cal])
            for j,(weight,component) in enumerate(zip(plan["weights"],plan["members"])):
                cfg=component["config"];builder=FeatureBuilder(cfg["feature_set"],cfg["strategy"]).fit(old.features.iloc[fit])
                X=builder.transform(old.features.iloc[fit]);V=builder.transform(old.features.iloc[cal])
                encoder=InputEncoder(cfg["family"]).fit(X);X=encoder.transform(X);V=encoder.transform(V)
                assert pp.TARGET not in X
                for model_seed in component["seeds"]:
                    part=out/f"{tag}_teacher_{j}_{model_seed}.npz"
                    if part.exists():
                        z=np.load(part);np.testing.assert_array_equal(z["fit_indices"],fit);np.testing.assert_array_equal(z["cal_indices"],cal);q=z["probability"]
                    else:
                        model=estimator(dict(cfg,seed=model_seed),encoder.cat_columns_,4)
                        fit_model(model,cfg["family"],X,old.y[fit]);q=model.predict_proba(V)[:,1]
                        np.savez_compressed(part,fit_indices=fit,cal_indices=cal,probability=q)
                    p+=weight/len(component["seeds"])*q
                    print(f"{tag} inner teacher {j+1}/3 seed {model_seed} ready",flush=True)
            np.savez_compressed(cache,fit_indices=fit,cal_indices=cal,probability=p)
        np.testing.assert_array_equal(np.sort(np.r_[fit,cal]),np.sort(training))
        assert not set(fit)&set(cal) and not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[cal])
        return cal,p

    rows=[]
    for repeat in [1,2]:
        ref=np.full(len(old.y),np.nan);alt=ref.copy();ctl=ref.copy();fold_of=np.zeros(len(old.y),int)
        for fold in range(1,6):
            z=np.load(baseline_out/f"repeat_{repeat}_fold_{fold}.npz");ti,vi=z["training"],z["validation"]
            assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
            reference=z["probability"]@np.array(plan["weights"]);tag=f"repeat_{repeat}_fold_{fold}"
            ci,p=inner_teacher(ti,690001+repeat*10+fold,tag)
            assert not set(ci)&set(vi)
            predictions={}
            for label,profile,offset_profile in [("roles",CHILD,OFFSET),("control",CONTROL,OFFSET_CONTROL)]:
                specialist=[]
                for seed in SEEDS:
                    part=out/f"{tag}_{label}_{seed}.npz"
                    if part.exists():
                        a=np.load(part);np.testing.assert_array_equal(a["training"],ti);np.testing.assert_array_equal(a["validation"],vi);q=a["probability"]
                    else:
                        member=fit_member(profile,seed,old.features.iloc[ti],old.y[ti],public)
                        q=predict_specialist(member,old.features.iloc[vi]);np.savez_compressed(part,training=ti,validation=vi,probability=q)
                    specialist.append(q)
                calibrator=GOffsetCalibrator(**offset_profile).fit(old.features.iloc[ci],p,old.y[ci],public)
                cold_p=calibrator.predict(old.features.iloc[vi],reference)
                predictions[label]=route_probability(old.features.iloc[vi],reference,cold_p,np.mean(specialist,axis=0))
                _write_json(report/f"{tag}_{label}_calibrator.json",calibrator.audit_)
            ref[vi]=reference;alt[vi]=predictions["roles"];ctl[vi]=predictions["control"];fold_of[vi]=fold
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId.iloc[vi],reference=reference,
                alternative=predictions["roles"],control=predictions["control"])),report/f"{tag}_predictions.csv")
            print(f"{tag} routing and matched control ready",flush=True)
        row=score(old.y,ref,alt,old.features,fold_of);control=score(old.y,ref,ctl,old.features,fold_of)
        rows.append(dict(repeat=repeat,**row,control=control,added_roles_g_gain=row["g_accuracy"]-control["g_accuracy"]))
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=fold_of,
            reference=ref,alternative=alt,control=ctl)),report/f"repeat_{repeat}_oof.csv")
        _write_json(report/"repeat_results.json",rows)
        hard_failure=(row["net_correct"]<0 or row["cold_net_correct"]<0 or row["child_net_correct"]<0 or
            row["log_loss"]>row["reference_log_loss"]+.002 or row["g_log_loss"]>row["g_reference_log_loss"]+.005 or
            (repeat==1 and row["g_improved_folds"]==0))
        if hard_failure:
            _write_json(report/"summary.json",dict(status="no_candidate",completed_repeats=repeat,early_stop=repeat==1,
                reason="Predeclared cohort non-regression/loss/futility rule failed",repeats=rows,kaggle_score=None))
            print("Frozen routing fails non-regression; stop without changing the rule",flush=True);return
    passed=np.mean([r["gain"] for r in rows])>=.001 and sum(r["g_improved_folds"] for r in rows)>=6 and np.mean([r["added_roles_g_gain"] for r in rows])>=.003
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",completed_repeats=2,early_stop=False,
            reason="Predeclared mean gain/fold/ablation gate failed",repeats=rows,kaggle_score=None));return
    _write_json(report/"frozen_pass.json",dict(passed=True,repeats=rows,no_retuning=True))
    original=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report/"oof_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    old_test=pd.read_csv(old.output/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    members=[];oof=np.zeros(len(old.y));test=np.zeros(len(old.raw_test));identity=IdentityTransform()
    for block,fold in enumerate([5,1,2,3,4]):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        ci,p=inner_teacher(ti,700001+fold,f"final_fold_{fold}")
        c=GOffsetCalibrator(**OFFSET).fit(old.features.iloc[ci],p,old.y[ci],public)
        specialists=[fit_member(CHILD,s,old.features.iloc[ti],old.y[ti],public) for s in SEEDS]
        children=[dict(m,weight=m["weight"]*5) for m in original["members"][block*9:block*9+9]]
        model=GRoutedFoldModel(children,c,specialists)
        oof[vi]=model.predict_proba(old.features.iloc[vi])[:,1];test+=model.predict_proba(old.test_features)[:,1]/5
        members.append(dict(weight=.2,builder=identity,encoder=identity,model=model))
        print(f"Final routing fold {fold}/5 ready",flush=True)
    cold,child=cohort_masks(old.test_features);untouched=~(cold|child)
    np.testing.assert_allclose(test[untouched],old_test.probability.to_numpy()[untouched],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(test[untouched]>=.5,old_test.prediction.to_numpy()[untouched])
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5),data_hashes=original["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,test,pp.load_submission_template(),.5)
    _write_csv(submission,out/"submission_grouted4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test,prediction=test>=.5)),out/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof)),report/"oof_predictions.csv")
    reloaded,check=predict_ensemble(modelpath);np.testing.assert_allclose(reloaded,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",completed_repeats=2,repeats=rows,kaggle_score=None,
        canonical_accuracy=float(accuracy_score(old.y,oof>=.5)),changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=old_test.prediction.to_numpy())),
        untouched_test_labels_unchanged=True,original_members=45,specialists=15,calibrators=5,reload_max_error=float(np.max(np.abs(reloaded-test)))))


if __name__=="__main__":main()
