"""精简位置和消费特征与XGBoost实验；固定协议、可恢复、可复现。"""
import hashlib
import json
import os
import time

for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ.setdefault(name,"4")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import GroupKFold
import xgboost

from src.competition_features import InputEncoder
from src.focused_features import FocusedBuilder
from src.optimization import SearchRunner,config_id,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="focused_study_20261028"
MODES=["compact","location","spend_location","physical"]
PROFILES=[dict(family=family,mode=mode,variant=variant,resolve_planet=True)
          for family in ["xgboost","catboost"] for mode in MODES for variant in [0,1]]
PROFILES += [dict(family="lightgbm",mode=mode,variant=variant,resolve_planet=True)
             for mode in ["location","physical"] for variant in [0,1]]
# 单独对照HomePlanet补缺，避免把位置增益和新补缺混为一谈。
PROFILES += [dict(family="xgboost",mode="location",variant=1,resolve_planet=False),
             dict(family="catboost",mode="location",variant=0,resolve_planet=False)]


def estimator(profile,seed,categorical):
    v=profile["variant"];family=profile["family"]
    if family=="xgboost":
        return xgboost.XGBClassifier(n_estimators=[850,600][v],learning_rate=.035,max_depth=[3,5][v],
            min_child_weight=[8,12][v],reg_lambda=[8.,15.][v],reg_alpha=.1,
            subsample=.85,colsample_bytree=.9,max_bin=256,tree_method="hist",enable_categorical=True,
            eval_metric="logloss",random_state=seed,n_jobs=4)
    if family=="catboost":
        return CatBoostClassifier(iterations=[800,1000][v],depth=[5,6][v],learning_rate=[.04,.035][v],
            l2_leaf_reg=[6.,10.][v],max_ctr_complexity=2,random_seed=seed,thread_count=4,
            cat_features=categorical,loss_function="Logloss",verbose=False,allow_writing_files=False)
    return LGBMClassifier(n_estimators=[800,450][v],learning_rate=.035,num_leaves=[7,15][v],
        min_child_samples=35,reg_lambda=5.,reg_alpha=.1,subsample=.9,subsample_freq=1,colsample_bytree=.9,
        random_state=seed,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)


def fit_member(profile,seed,frame,y,context):
    builder=FocusedBuilder(profile["mode"],profile["resolve_planet"]).fit(frame,context)
    X=builder.transform(frame)
    encoder=InputEncoder("catboost" if profile["family"]=="catboost" else "lightgbm").fit(X)
    model=estimator(profile,seed,encoder.cat_columns_)
    model.fit(encoder.transform(X),y)
    return dict(builder=builder,encoder=encoder,model=model)


def predict(member,frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:,1]


def main():
    old=SearchRunner(threads=4);report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    dev=old.fold_of!=5
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of)),report/"fold_assignments.csv")
    params=[estimator(p,42,[]).get_params() for p in PROFILES]
    # XGB的参数有NaN作为缺失标志，转成明确字符串以支持严格JSON比较。
    params=json.loads(json.dumps(params,default=str).replace('NaN','"NaN"'))
    protocol=dict(name=NAME,profiles=PROFILES,parameters=params,xgboost_version=xgboost.__version__,
        development_folds=[1,2,3,4],excluded_fold=5,seeds=[42,2026,103],weights=[.25,.5,.75,1.],threshold=.5,
        gate="Three-seed development accuracy gain >= .001 and logloss deterioration <= .002; improvement in >=3/4 fixed folds. Then changed-partition sensitivity check: gain >=0 and logloss deterioration <= .002. No adjustments after either check.",
        sensitivity_seed=20261029,sensitivity_folds=4,
        context="Public unlabeled train+test pool for consistent surname/deck HomePlanet lookup; target never enters lookup",
        caveat="All folds are historically inspected. Repartitioning checks sensitivity, not independent blind validation. Initial diagnostic included historical full OOF.",
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
                 ["scripts/focused_study.py","src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py","features/folds.csv","requirements.txt"]},
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"])
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("实验协议或源码变化，拒绝混用缓存")
    else:_write_json(path,protocol)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    incumbent=np.full(len(old.y),np.nan);incumbent[dev]=0
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            p=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            assert np.isnan(p[~dev]).all()
            incumbent[dev]+=weight/len(member["seeds"])*p[dev]
    base_acc=float(accuracy_score(old.y[dev],incumbent[dev]>=.5));base_loss=float(log_loss(old.y[dev],incumbent[dev]))
    np.testing.assert_allclose(base_acc,plan["development_accuracy"],atol=1e-14)

    def evaluate(index,seed):
        cache=output/f"profile_{index}_seed_{seed}.npz";record=report/f"profile_{index}_seed_{seed}.json"
        if cache.exists() and record.exists():return np.load(cache)["probability"]
        started=time.perf_counter();pred=np.full(len(old.y),np.nan);foldscores=[]
        for fold,(ti,vi) in enumerate(old.splits,1):
            assert not (old.fold_of[ti]==5).any()
            member=fit_member(PROFILES[index],seed,old.features.iloc[ti],old.y[ti],public)
            pred[vi]=predict(member,old.features.iloc[vi]);foldscores.append(float(accuracy_score(old.y[vi],pred[vi]>=.5)))
            print(f"Focused {index} {PROFILES[index]['family']}/{PROFILES[index]['mode']} seed {seed} fold {fold}: {foldscores[-1]:.4f}",flush=True)
        assert np.isnan(pred[~dev]).all()
        np.savez_compressed(cache,probability=pred)
        _write_json(record,dict(profile=index,config=PROFILES[index],seed=seed,fold_scores=foldscores,
            accuracy=float(accuracy_score(old.y[dev],pred[dev]>=.5)),log_loss=float(log_loss(old.y[dev],pred[dev])),seconds=time.perf_counter()-started))
        return pred

    def choices(predictions):
        rows=[]
        for index,pred in predictions.items():
            for weight in protocol["weights"]:
                mixed=(1-weight)*incumbent[dev]+weight*pred[dev]
                fold_scores=[];fold_gains=[]
                for fold in range(1,5):
                    mask=old.fold_of[dev]==fold
                    score=accuracy_score(old.y[dev][mask],mixed[mask]>=.5)
                    fold_scores.append(float(score));fold_gains.append(float(score-accuracy_score(old.y[dev][mask],incumbent[dev][mask]>=.5)))
                rows.append(dict(profile=index,new_weight=weight,accuracy=float(accuracy_score(old.y[dev],mixed>=.5)),
                                 log_loss=float(log_loss(old.y[dev],mixed)),fold_scores=fold_scores,fold_gains=fold_gains,
                                 improved_folds=int(np.sum(np.array(fold_gains)>0))))
        return sorted(rows,key=lambda r:(-r["accuracy"],r["log_loss"],r["new_weight"]))

    first={i:evaluate(i,42) for i in range(len(PROFILES))};initial=choices(first)
    top=[]
    for row in initial:
        if row["profile"] not in top:top.append(row["profile"])
        if len(top)==2:break
    repeated={i:np.mean([first[i],evaluate(i,2026),evaluate(i,103)],axis=0) for i in top}
    scans=choices(repeated);winner=scans[0]
    _write_json(report/"single_seed_mixtures.json",initial)
    _write_json(report/"three_seed_mixtures.json",scans)
    passed=winner["accuracy"]>=base_acc+.001 and winner["log_loss"]<=base_loss+.002 and winner["improved_folds"]>=3
    frozen=dict(profile=winner["profile"],config=PROFILES[winner["profile"]],weight=winner["new_weight"],
        threshold=.5,seeds=protocol["seeds"],development=winner,incumbent_accuracy=base_acc,incumbent_log_loss=base_loss,
        initial_gate_passed=bool(passed))
    _write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,
                   reason="Predeclared initial gate not met; preserve observed Kaggle best 0.81108"))
        print("Focused initial gate failed; original ensemble retained",flush=True);return
    w=frozen["weight"]
    # 冻结选择后，只检查更换分组划分的敏感性；不根据结果修改参数或权重。
    sensitivity_cache=output/"sensitivity.npz"
    if sensitivity_cache.exists():
        cache=np.load(sensitivity_cache);reference,alternative=cache["reference"],cache["alternative"]
    else:
        reference=np.full(len(old.y),np.nan);alternative=np.full(len(old.y),np.nan)
        indices=np.flatnonzero(dev)
        splitter=GroupKFold(4,shuffle=True,random_state=protocol["sensitivity_seed"])
        for n,(ta,va) in enumerate(splitter.split(indices,old.y[dev],old.features.GroupId[dev]),1):
            ti,vi=indices[ta],indices[va]
            reference[vi],_,_=old.fit_partition(plan,ti,vi)
            new=np.mean([predict(fit_member(frozen["config"],seed,old.features.iloc[ti],old.y[ti],public),old.features.iloc[vi]) for seed in frozen["seeds"]],axis=0)
            alternative[vi]=(1-w)*reference[vi]+w*new
            print(f"Focused changed-partition sensitivity {n}/4",flush=True)
        assert np.isnan(reference[~dev]).all() and np.isnan(alternative[~dev]).all()
        np.savez_compressed(sensitivity_cache,reference=reference,alternative=alternative)
    sensitivity=dict(reference_accuracy=float(accuracy_score(old.y[dev],reference[dev]>=.5)),
        alternative_accuracy=float(accuracy_score(old.y[dev],alternative[dev]>=.5)),
        reference_log_loss=float(log_loss(old.y[dev],reference[dev])),alternative_log_loss=float(log_loss(old.y[dev],alternative[dev])),
        interpretation="Changed-partition sensitivity after selection on the same historical rows, not independent validation")
    passed=sensitivity["alternative_accuracy"]>=sensitivity["reference_accuracy"] and sensitivity["alternative_log_loss"]<=sensitivity["reference_log_loss"]+.002
    sensitivity["passed"]=bool(passed);_write_json(report/"sensitivity.json",sensitivity)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,reason="Changed-partition sensitivity gate failed"))
        print("Focused sensitivity gate failed; original ensemble retained",flush=True);return
    incumbent_bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=[dict(m,weight=m["weight"]*(1-w)) for m in incumbent_bundle["members"] if w<1]
    canonical=pd.read_csv(old.report/"oof_predictions.csv",dtype={"PassengerId":str})
    old_test=pd.read_csv(old.output/"test_predictions.csv",dtype={"PassengerId":str})
    np.testing.assert_array_equal(canonical.PassengerId,old.raw.PassengerId)
    np.testing.assert_array_equal(old_test.PassengerId,old.raw_test.PassengerId)
    oof=(1-w)*canonical.probability.to_numpy();tp=(1-w)*old_test.probability.to_numpy()
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True)
    for fold in range(1,6):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        for seed in frozen["seeds"]:
            member=fit_member(frozen["config"],seed,old.features.iloc[ti],old.y[ti],public)
            oof[vi]+=w/3*predict(member,old.features.iloc[vi]);tp+=w/15*predict(member,old.test_features)
            members.append(dict(member,weight=w/15))
            if frozen["config"]["family"]=="xgboost":
                member["model"].save_model(modeldir/f"fold_{fold}_seed_{seed}.ubj")
        print(f"Focused frozen refit fold {fold}/5",flush=True)
    modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(threshold=.5,name=NAME,frozen_choice=frozen),data_hashes=incumbent_bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,tp,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_focused4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=tp,prediction=tp>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    reloaded,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reloaded,tp,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,sensitivity=sensitivity,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=old_test.prediction.to_numpy())),
        models=len(members),reload_max_error=float(np.max(np.abs(reloaded-tp))),kaggle_score=None,
        interpretation="Historical post-selection OOF; platform improvement unknown until submission"))


if __name__=="__main__":main()
