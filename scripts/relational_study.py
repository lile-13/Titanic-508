"""有限的分组交叉拟合关系特征对照；固定0.5阈值，不用榜分选择参数。"""
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,log_loss

from src.competition_features import InputEncoder
from src.relational_features import RelationalBuilder
from src.optimization import SearchRunner,config_id
from src.train import _write_json,_write_csv,make_submission
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src import preprocess as pp


NAME = "relational_study_20261028"
PROFILES = [("base","rule_group",None,False),("base","native",None,False),
            ("geo","native",None,False),("base","rule_group",10.,False),
            ("base","rule_group",30.,False),("base","native",10.,False),
            ("geo","native",10.,False),("geo","rule_group",10.,True)]


def model(seed):
    return LGBMClassifier(n_estimators=600,learning_rate=.035,num_leaves=15,
                          min_child_samples=35,reg_alpha=.1,reg_lambda=5.,
                          colsample_bytree=.9,subsample=.85,subsample_freq=1,
                          random_state=seed,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)


def main():
    old = SearchRunner(threads=4)
    report=EXPERIMENTS_DIR/NAME;report.mkdir(parents=True,exist_ok=True)
    output=PROJECT_ROOT/"outputs"/NAME;output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    protocol=dict(name=NAME,profiles=PROFILES,params=model(42).get_params(),threshold=.5,
                  development_folds=[1,2,3,4],excluded_from_development=5,
                  caveat="All historical folds have been inspected previously; this is additional development, not a new blind test",
                  promotion_rule="3-seed blend dev gain >=0.0015 over frozen incumbent and logloss deterioration <=0.003; later full five-fold is descriptive",
                  source_hashes={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in ["src/relational_features.py","scripts/relational_study.py"]})
    protocol_path=report/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8"))!=protocol:raise ValueError("试验实现已改变，不能混合缓存")
    else:_write_json(protocol_path,protocol)
    dev=old.fold_of!=5
    incumbent=np.zeros(len(old.y))
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            c=dict(member["config"],seed=seed)
            trial=config_id(c,"confirmation",[1,2,3,4])
            p=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            assert np.isnan(p[~dev]).all()
            incumbent[dev]+=weight/len(member["seeds"])*p[dev]
    base_acc=float(accuracy_score(old.y[dev],incumbent[dev]>=.5))
    base_loss=float(log_loss(old.y[dev],incumbent[dev]))
    np.testing.assert_allclose(base_acc,plan["development_accuracy"],atol=1e-14)
    def evaluate(index,seed):
        path=output/f"profile_{index}_seed_{seed}.npz"
        recpath=report/f"profile_{index}_seed_{seed}.json"
        if path.exists() and recpath.exists():return np.load(path)["probability"]
        started=time.perf_counter();pred=np.full(len(old.y),np.nan);folds=[]
        for fold,(ti,vi) in enumerate(old.splits,1):
            builder=RelationalBuilder(*PROFILES[index])
            X=builder.fit_transform(old.features.iloc[ti],old.y[ti]);V=builder.transform(old.features.iloc[vi])
            encoder=InputEncoder("lightgbm").fit(X)
            learner=model(seed).fit(encoder.transform(X),old.y[ti])
            pred[vi]=learner.predict_proba(encoder.transform(V))[:,1]
            folds.append(float(accuracy_score(old.y[vi],pred[vi]>=.5)))
        assert np.isnan(pred[~dev]).all()
        np.savez_compressed(path,probability=pred)
        record=dict(profile=index,definition=PROFILES[index],seed=seed,fold_scores=folds,
                    accuracy=float(accuracy_score(old.y[dev],pred[dev]>=.5)),log_loss=float(log_loss(old.y[dev],pred[dev])),seconds=time.perf_counter()-started)
        _write_json(recpath,record)
        print(f"Relational profile {index} seed {seed}: {record['accuracy']:.4f} ({record['seconds']:.1f}s)",flush=True)
        return pred
    def choices(predictions):
        rows=[]
        for index,pred in predictions.items():
            for weight in [.25,.5,1.]:
                mix=(1-weight)*incumbent[dev]+weight*pred[dev]
                rows.append(dict(profile=index,new_weight=weight,accuracy=float(accuracy_score(old.y[dev],mix>=.5)),log_loss=float(log_loss(old.y[dev],mix))))
        return sorted(rows,key=lambda r:(-r["accuracy"],r["log_loss"],r["new_weight"]))
    first={i:evaluate(i,42) for i in range(len(PROFILES))}
    initial=choices(first)
    top=[]
    for row in initial:
        if row["profile"] not in top:top.append(row["profile"])
        if len(top)==2:break
    seeded={i:np.mean([first[i],evaluate(i,2026),evaluate(i,103)],axis=0) for i in top}
    scans=choices(seeded);winner=scans[0]
    _write_csv(pd.DataFrame(initial),report/"single_seed_mixtures.csv")
    _write_csv(pd.DataFrame(scans),report/"three_seed_mixtures.csv")
    passed=winner["accuracy"]>=base_acc+.0015 and winner["log_loss"]<=base_loss+.003
    frozen=dict(profile=winner["profile"],definition=PROFILES[winner["profile"]],weight=winner["new_weight"],threshold=.5,seeds=[42,2026,103],development=winner,incumbent_accuracy=base_acc,incumbent_log_loss=base_loss,generate_candidate=bool(passed))
    _write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,reason="Predeclared development improvement gate not met; preserve 0.81108 incumbent"))
        print("No new candidate: incumbent retained.",flush=True);return
    # 此后只按冻结方案重训；不再根据历史第5折或榜分调整选择。
    old_bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report/"oof_predictions.csv",dtype={"PassengerId":str})
    old_test=pd.read_csv(old.output/"test_predictions.csv",dtype={"PassengerId":str})
    w=frozen["weight"];members=[dict(m,weight=m["weight"]*(1-w)) for m in old_bundle["members"] if w<1]
    oof=(1-w)*canonical.probability.to_numpy();test_pred=(1-w)*old_test.probability.to_numpy()
    for fold in range(1,6):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        builder=RelationalBuilder(*frozen["definition"])
        X=builder.fit_transform(old.features.iloc[ti],old.y[ti]);V=builder.transform(old.features.iloc[vi]);T=builder.transform(old.test_features)
        encoder=InputEncoder("lightgbm").fit(X);X,V,T=encoder.transform(X),encoder.transform(V),encoder.transform(T)
        for seed in frozen["seeds"]:
            learner=model(seed).fit(X,old.y[ti]);oof[vi]+=w/3*learner.predict_proba(V)[:,1];test_pred+=w/15*learner.predict_proba(T)[:,1]
            members.append(dict(weight=w/15,builder=builder,encoder=encoder,model=learner))
        print(f"Relational frozen refit fold {fold}/5",flush=True)
    model_dir=PROJECT_ROOT/"models"/NAME;model_dir.mkdir(parents=True,exist_ok=True)
    bundle=dict(members=members,plan=dict(threshold=.5,name=NAME,frozen_choice=frozen),data_hashes=old_bundle["data_hashes"])
    joblib.dump(bundle,model_dir/"ensemble.joblib",compress=3)
    submission=make_submission(old.raw_test.PassengerId,test_pred,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_relational4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test_pred,prediction=test_pred>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),models=len(members),kaggle_score=None,interpretation="Post-selection descriptive OOF; additional candidate is not an observed Kaggle improvement"))


if __name__=="__main__":main()
