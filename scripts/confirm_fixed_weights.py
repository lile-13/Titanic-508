"""固定昨天的微弱权重线索，用两次新分组五折检验，不再搜索权重。"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import GroupKFold

from src.competition_features import FeatureBuilder,InputEncoder
from src.ensemble_weights import reweight_members
from src.optimization import SearchRunner,estimator,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="fixed_weight_confirmation_20261029"
WEIGHTS=[.4,.15,.45]


def predict(member,frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:,1]


def main():
    old=SearchRunner(threads=4);report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    protocol=dict(name=NAME,weights=WEIGHTS,threshold=.5,split_seeds=[20261029,20261106],folds=5,
        gate="Mean repeated OOF gain >=.001, both repeats nonnegative, >=6/10 folds improved; logloss deterioration <=.002 on each repeat. No selection or retuning.",
        historical_choice="Old 49-weight search had only +6 development correct predictions and failed its gate. This is a new fixed-choice follow-up, not a reversal of that result.",
        caveat="All original rows and folds historically inspected. Repeated repartitioning provides stability evidence, not independent blind evaluation.",
        source_choice_hash=hashlib.sha256((EXPERIMENTS_DIR/"weight_refinement_20261028/frozen_choice.json").read_bytes()).hexdigest(),
        source_plan_hash=hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest(),
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
                 ["scripts/confirm_fixed_weights.py","src/ensemble_weights.py","src/optimization.py","src/competition_features.py","src/preprocess.py","features/folds.csv"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("协议改变，不能混用缓存")
    else:_write_json(path,protocol)
    _write_json(report/"frozen_choice.json",dict(weights=WEIGHTS,threshold=.5,source="weight_refinement_20261028",frozen_before_new_partitions=True))
    summaries=[];metrics=[];assignments=[]
    for repeat,seed in enumerate(protocol["split_seeds"],1):
        splitter=GroupKFold(5,shuffle=True,random_state=seed)
        matrix=np.full((len(old.y),3),np.nan)
        for fold,(ti,vi) in enumerate(splitter.split(old.features,old.y,old.features.GroupId),1):
            assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
            cache=output/f"repeat_{repeat}_fold_{fold}.npz"
            if cache.exists():
                data=np.load(cache);np.testing.assert_array_equal(data["training"],ti);np.testing.assert_array_equal(data["validation"],vi)
                values=data["probability"]
            else:
                values=np.zeros((len(vi),3))
                for col,choice in enumerate(plan["members"]):
                    c=choice["config"];builder=FeatureBuilder(c["feature_set"],c["strategy"]).fit(old.features.iloc[ti])
                    X=builder.transform(old.features.iloc[ti]);V=builder.transform(old.features.iloc[vi])
                    encoder=InputEncoder(c["family"]).fit(X);X,V=encoder.transform(X),encoder.transform(V)
                    for model_seed in choice["seeds"]:
                        model=estimator(dict(c,seed=model_seed),encoder.cat_columns_,threads=4)
                        model.fit(X,old.y[ti]);values[:,col]+=model.predict_proba(V)[:,1]/len(choice["seeds"])
                np.savez_compressed(cache,training=ti,validation=vi,probability=values)
            matrix[vi]=values
            before=values@np.array(plan["weights"]);after=values@np.array(WEIGHTS)
            row=dict(repeat=repeat,seed=seed,fold=fold,n=len(vi),reference_accuracy=float(accuracy_score(old.y[vi],before>=.5)),
                alternative_accuracy=float(accuracy_score(old.y[vi],after>=.5)),group_overlap=0)
            row["gain"]=row["alternative_accuracy"]-row["reference_accuracy"];metrics.append(row)
            assignments.extend(dict(PassengerId=old.raw.PassengerId.iloc[i],repeat=repeat,fold=fold) for i in vi)
            _write_json(report/"completed_folds.json",metrics)
            print(f"Fixed weights repeat {repeat}/2 fold {fold}/5: {row['reference_accuracy']:.4f} -> {row['alternative_accuracy']:.4f}",flush=True)
        assert np.isfinite(matrix).all()
        before=matrix@np.array(plan["weights"]);after=matrix@np.array(WEIGHTS)
        summary=dict(repeat=repeat,seed=seed,reference_accuracy=float(accuracy_score(old.y,before>=.5)),alternative_accuracy=float(accuracy_score(old.y,after>=.5)),
            reference_log_loss=float(log_loss(old.y,before)),alternative_log_loss=float(log_loss(old.y,after)))
        summary["gain"]=summary["alternative_accuracy"]-summary["reference_accuracy"];summaries.append(summary)
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,reference=before,alternative=after)),report/f"repeat_{repeat}_oof.csv")
    mean_gain=float(np.mean([r["gain"] for r in summaries]));improved=sum(r["gain"]>0 for r in metrics)
    passed=mean_gain>=.001 and all(r["gain"]>=0 for r in summaries) and improved>=6 and all(r["alternative_log_loss"]<=r["reference_log_loss"]+.002 for r in summaries)
    _write_csv(pd.DataFrame(assignments),report/"new_fold_assignments.csv")
    _write_json(report/"repeated_evaluation.json",dict(repeats=summaries,mean_gain=mean_gain,improved_folds=improved,passed=bool(passed)))
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",weights=WEIGHTS,repeats=summaries,mean_gain=mean_gain,
            improved_folds=improved,kaggle_score=None,reason="Fixed-choice repeated-partition gate failed; previous rejection remains unchanged"));return
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=reweight_members(bundle["members"],plan,WEIGHTS)
    canonical=pd.read_csv(old.report/"oof_predictions.csv")
    oof=np.zeros(len(old.y));reference=oof.copy();test=np.zeros(len(old.raw_test))
    for block,fold in enumerate([5,1,2,3,4]):
        vi=np.flatnonzero(old.fold_of==fold)
        for original,changed in zip(bundle["members"][block*9:block*9+9],members[block*9:block*9+9]):
            p=predict(original,old.features.iloc[vi]);reference[vi]+=original["weight"]*5*p;oof[vi]+=changed["weight"]*5*p
            test+=changed["weight"]*predict(changed,old.test_features)
    np.testing.assert_allclose(reference,canonical.probability.to_numpy(),atol=1e-12,rtol=0)
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True);modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(plan,name=NAME,weights=WEIGHTS,confirmation=protocol),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,test,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_fixedweights4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=test,prediction=test>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    reload,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reload,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    original=pd.read_csv(old.output/"submission.csv")
    _write_json(report/"summary.json",dict(status="candidate_generated",weights=WEIGHTS,repeats=summaries,mean_gain=mean_gain,
        improved_folds=improved,kaggle_score=None,canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),
        models=45,changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=original.Transported.to_numpy())),
        reload_max_error=float(np.max(np.abs(reload-test))),interpretation="Fixed-choice historical sensitivity evidence, not a blind validation or observed Kaggle improvement"))


if __name__=="__main__":main()
