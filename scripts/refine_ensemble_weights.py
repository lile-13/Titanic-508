"""只细化既有三组件的固定融合权重；冻结后检查换分组划分敏感性。"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import GroupKFold

from src.optimization import SearchRunner,config_id,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="weight_refinement_20261028"


def member_prediction(member,frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:,1]


def main():
    runner=SearchRunner(threads=4);report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME
    report.mkdir(exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    oldplan=json.loads((runner.report/"frozen_plan.json").read_text(encoding="utf-8"))
    grid=[]
    for a in range(2,9):
        for b in range(2,9):
            grid.append([a/20,b/20,1-(a+b)/20])
    protocol=dict(name=NAME,weights=grid,threshold=.5,seeds=[42,2026,103],development_folds=[1,2,3,4],excluded_fold=5,
        gate="Development gain >=.001 and >=3/4 folds improved; logloss deterioration <=.002. Freeze, then changed-partition gain >=0 and logloss deterioration <=.002. No later retuning.",
        sensitivity_seed=20261029,parameters="Frozen original model parameters unchanged; only 49 predeclared convex weight combinations",
        caveat="Historical development folds; repartition is a sensitivity check, not independent blind validation",
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
                 ["scripts/refine_ensemble_weights.py","src/optimization.py","src/competition_features.py","src/preprocess.py","features/folds.csv"]},
        source_plan_hash=hashlib.sha256((runner.report/"frozen_plan.json").read_bytes()).hexdigest())
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("协议变化，不能复用缓存")
    else:_write_json(path,protocol)
    dev=runner.fold_of!=5;components=[]
    for member in oldplan["members"]:
        values=[]
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            p=np.load(runner.output/"search"/f"{trial}.npz")["probability"]
            assert np.isnan(p[~dev]).all();values.append(p[dev])
        components.append(np.mean(values,axis=0))
    matrix=np.column_stack(components);original=matrix@np.array(oldplan["weights"])
    base_acc=float(accuracy_score(runner.y[dev],original>=.5));base_loss=float(log_loss(runner.y[dev],original))
    np.testing.assert_allclose(base_acc,oldplan["development_accuracy"],atol=1e-14)
    rows=[]
    for weights in grid:
        p=matrix@np.array(weights);gains=[]
        for fold in range(1,5):
            mask=runner.fold_of[dev]==fold
            gains.append(float(accuracy_score(runner.y[dev][mask],p[mask]>=.5)-accuracy_score(runner.y[dev][mask],original[mask]>=.5)))
        rows.append(dict(weights=weights,accuracy=float(accuracy_score(runner.y[dev],p>=.5)),
            log_loss=float(log_loss(runner.y[dev],p)),fold_gains=gains,improved_folds=int(np.sum(np.array(gains)>0)),
            distance_from_original=float(np.linalg.norm(np.array(weights)-np.array(oldplan["weights"])))))
    rows.sort(key=lambda r:(-r["accuracy"],r["distance_from_original"],r["log_loss"]))
    best=rows[0]
    passed=best["accuracy"]>=base_acc+.001 and best["improved_folds"]>=3 and best["log_loss"]<=base_loss+.002
    frozen=dict(weights=best["weights"],threshold=.5,development=best,incumbent_accuracy=base_acc,
                incumbent_log_loss=base_loss,initial_gate_passed=bool(passed))
    _write_json(report/"weight_comparisons.json",rows);_write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,
                    reason="Predeclared development gate not met"))
        print(f"Weight refinement gate failed: {best['accuracy']:.6f} vs {base_acc:.6f}",flush=True);return
    weights=np.array(best["weights"])
    indices=np.flatnonzero(dev);ref=np.full(len(runner.y),np.nan);alt=ref.copy()
    split=GroupKFold(4,shuffle=True,random_state=protocol["sensitivity_seed"])
    for n,(a,b) in enumerate(split.split(indices,runner.y[dev],runner.features.GroupId[dev]),1):
        ti,vi=indices[a],indices[b]
        reference,_,members=runner.fit_partition(oldplan,ti,vi)
        assert len(members)==9
        values=np.column_stack([np.mean([member_prediction(m,runner.features.iloc[vi]) for m in members[i*3:i*3+3]],axis=0) for i in range(3)])
        np.testing.assert_allclose(values@np.array(oldplan["weights"]),reference,atol=1e-12,rtol=0)
        ref[vi]=reference;alt[vi]=values@weights
        print(f"Weight frozen sensitivity {n}/4",flush=True)
    sensitivity=dict(reference_accuracy=float(accuracy_score(runner.y[dev],ref[dev]>=.5)),
        alternative_accuracy=float(accuracy_score(runner.y[dev],alt[dev]>=.5)),reference_log_loss=float(log_loss(runner.y[dev],ref[dev])),
        alternative_log_loss=float(log_loss(runner.y[dev],alt[dev])),interpretation="Changed-partition sensitivity on historically inspected rows")
    passed=sensitivity["alternative_accuracy"]>=sensitivity["reference_accuracy"] and sensitivity["alternative_log_loss"]<=sensitivity["reference_log_loss"]+.002
    sensitivity["passed"]=bool(passed);_write_json(report/"sensitivity.json",sensitivity)
    np.savez_compressed(output/"sensitivity.npz",reference=ref,alternative=alt)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,sensitivity=sensitivity,kaggle_score=None,
                   reason="Changed-partition gate failed"));return
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    assert len(bundle["members"])==45
    oof=np.zeros(len(runner.y));ref_oof=oof.copy();test=np.zeros(len(runner.raw_test));members=[]
    for block,fold in enumerate([5,1,2,3,4]):
        vi=np.flatnonzero(runner.fold_of==fold)
        for component in range(3):
            children=bundle["members"][block*9+component*3:block*9+component*3+3]
            for m in children:
                ratio=weights[component]/oldplan["weights"][component]
                new=dict(m,weight=m["weight"]*ratio);members.append(new)
                pred=member_prediction(m,runner.features.iloc[vi])
                ref_oof[vi]+=m["weight"]*5*pred;oof[vi]+=new["weight"]*5*pred
                test+=new["weight"]*member_prediction(m,runner.test_features)
    original_oof=pd.read_csv(runner.report/"oof_predictions.csv")
    np.testing.assert_allclose(ref_oof,original_oof.probability,atol=1e-12,rtol=0)
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(parents=True,exist_ok=True)
    newplan=dict(oldplan,name=NAME,weights=weights.tolist(),threshold=.5,frozen_choice=frozen)
    modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=newplan,data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(runner.raw_test.PassengerId,test,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_weights4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=runner.raw_test.PassengerId,probability=test,prediction=test>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=runner.raw.PassengerId,fold=runner.fold_of,Transported=runner.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    reload,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reload,test,atol=1e-12,rtol=0);pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,sensitivity=sensitivity,
        kaggle_score=None,canonical_oof=float(accuracy_score(runner.y,oof>=.5)),canonical_log_loss=float(log_loss(runner.y,oof)),
        models=45,changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=pd.read_csv(runner.output/"submission.csv").Transported.to_numpy())),
        reload_max_error=float(np.max(np.abs(reload-test))),interpretation="Reweighted existing frozen models; descriptive post-selection OOF, no observed platform improvement"))


if __name__=="__main__":main()
