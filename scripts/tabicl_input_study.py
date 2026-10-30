"""Frozen input factorial, matched CatBoost controls, and new grouped partitions."""
import gc
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from scripts.tabicl_study import compare
from src import preprocess as pp
from src.competition_features import FeatureBuilder, InputEncoder
from src.optimization import SearchRunner, config_id, estimator, predict_ensemble
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.probability_calibration import IdentityTransform
from src.tabicl_inputs import VIEWS, TabICLInputBuilder, fit_tfm, fit_cat
from src.tabicl_models import PARAMS, CHECKPOINT, CHECKPOINT_SHA256
from src.train import _write_csv, _write_json

NAME="tabicl_input_audit_20261030"
SPLIT_SEEDS=[20261030,20261107]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction(m,family,frame):
    if family=="tfm":return m.predict_proba(frame)[:,1]
    return m["model"].predict_proba(m["encoder"].transform(m["builder"].transform(frame)))[:,1]


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/NAME;out=PROJECT_ROOT/"outputs"/NAME;modeldir=PROJECT_ROOT/"models"/NAME
    for p in [report,out,modeldir]:p.mkdir(parents=True,exist_ok=True)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    params=plan["members"][1]["config"]["params"]
    protocol=dict(name=NAME,views=VIEWS,tfm_params=PARAMS,cat_params=params,cat_seed=42,weight=.5,threshold=.5,
        feature_design="2x2 fixed factorial: compact 18 / expanded 34, each with / without GroupIdNum (17/33). Expanded adds existing GroupPos, GroupSurnameCount, CabinRelative, eight public group aggregates and five per-service logs. Public unique GroupId->Side fills missing only; compact parity required on complete folds.",
        selection="All three new TFM inputs evaluated on development folds 1..4. Qualify if ensemble gain >=.001, >=3/4 folds improve, loss deterioration <=.002, and accuracy >= compact-control blend +.0005. Select highest qualifying accuracy, then lowest logloss, then listed order. If none qualify stop; no new inputs, thresholds, weights or fallback algorithms.",
        matched_cat_control="Same four matrices, fixed frozen rich_names CatBoost parameters and seed 42; descriptive comparison only, no CatBoost candidate selection and no tuning.",
        sensitivity="After selection only: GroupKFold five-fold split seeds 20261030/20261107, unseen assignments at protocol freeze but historical rows. Fresh frozen 9-member original references and compact TFM control. Both selected-blend gains nonnegative, mean >=.001, >=6/10 improved folds, loss deterioration <=.002 each, mean gain over compact blend >=.0005. Stop after first negative/loss/zero-improved repeat.",
        diagnostic_boundary="Historical repeat2/fold3 learning curve is descriptive only and not a promotion gate. No route is learned from that fold's labels.",
        checkpoint_sha256=CHECKPOINT_SHA256,original_plan_sha256=sha(old.report/"frozen_plan.json"),
        compact_protocol_sha256=sha(EXPERIMENTS_DIR/"tabicl_20261030/protocol.json"),
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"],
        caveat="Adaptive input hypotheses after historical error inspection. New partitions do not create independent rows or a new blind holdout; candidate Kaggle score remains unknown.",
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/tabicl_input_study.py","src/tabicl_inputs.py","src/tabicl_models.py",
            "src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py","scripts/tabicl_study.py","features/folds.csv","requirements-tabicl.txt"]},
        environment={p:importlib.metadata.version(p) for p in ["tabicl","torch","catboost","lightgbm","numpy","pandas","scikit-learn"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Input factorial protocol changed")
    else:_write_json(path,protocol)
    for name,digest in protocol["data_hashes"].items():assert sha(pp.find_data_dir()/name)==digest
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT)==CHECKPOINT_SHA256
    if (report/"summary.json").exists():print("Completed frozen input study; audit instead",flush=True);return
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    columns={v:list(TabICLInputBuilder(v).fit(old.features,public).transform(old.features.iloc[:10])) for v in VIEWS}
    _write_json(report/"feature_columns.json",columns)

    def fitted(view,family,ti,vi,tag):
        stem=f"{family}_{view}_{tag}"
        cache,modelpath,recordpath=out/f"{stem}.npz",modeldir/f"{stem}.joblib",report/f"{stem}.json"
        if cache.exists() and modelpath.exists() and recordpath.exists():
            z=np.load(cache);np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
            assert sha(modelpath)==json.loads(recordpath.read_text(encoding="utf-8"))["model_sha256"]
            return z["probability"]
        assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
        m=(fit_tfm(view,old.features.iloc[ti],old.y[ti],public) if family=="tfm" else
           fit_cat(view,old.features.iloc[ti],old.y[ti],public,params))
        p=prediction(m,family,old.features.iloc[vi]);assert np.isfinite(p).all()
        joblib.dump(m,modelpath,compress=3);np.savez_compressed(cache,training=ti,validation=vi,probability=p)
        _write_json(recordpath,dict(view=view,family=family,tag=tag,model_sha256=sha(modelpath),
                    training_rows=len(ti),validation_rows=len(vi),columns=columns[view]))
        del m;gc.collect();torch.cuda.empty_cache()
        print(f"Input {stem} ready",flush=True)
        return p

    dev=old.fold_of!=5;reference=np.full(len(old.y),np.nan);reference[dev]=0
    for weight,component in zip(plan["weights"],plan["members"]):
        for seed in component["seeds"]:
            trial=config_id(dict(component["config"],seed=seed),"confirmation",[1,2,3,4])
            q=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            reference[dev]+=weight/len(component["seeds"])*q[dev]
    tfm={v:np.full(len(old.y),np.nan) for v in VIEWS};cat={v:np.full(len(old.y),np.nan) for v in VIEWS}
    for fold,(ti,vi) in enumerate(old.splits,1):
        original=joblib.load(PROJECT_ROOT/f"models/tabicl_20261030/development_fold_{fold}.joblib")
        control=TabICLInputBuilder("compact").fit(old.features.iloc[ti],public)
        pd.testing.assert_frame_equal(control.transform(old.features.iloc[ti]),original.training_X)
        pd.testing.assert_frame_equal(control.transform(old.features.iloc[vi]),original.builder.transform(old.features.iloc[vi]))
        z=np.load(PROJECT_ROOT/f"outputs/tabicl_20261030/development_fold_{fold}.npz")
        np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
        tfm["compact"][vi]=z["probability"]
        for v in VIEWS:
            if v!="compact":tfm[v][vi]=fitted(v,"tfm",ti,vi,f"development_fold_{fold}")
            cat[v][vi]=fitted(v,"cat",ti,vi,f"development_fold_{fold}")
    rows=[];compact_mix=.5*reference+.5*tfm["compact"]
    for v in VIEWS:
        for family,prob in [("tfm",tfm[v]),("cat",cat[v])]:
            mixed=.5*reference+.5*prob
            d=compare(old.y[dev],reference[dev],mixed[dev],old.fold_of[dev])
            standalone=float(((prob[dev]>=.5)==old.y[dev]).mean())
            rows.append(dict(view=v,family=family,standalone_accuracy=standalone,**d))
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,
                                        reference=reference,component=prob,alternative=mixed)),report/f"development_{family}_{v}.csv")
    _write_json(report/"development_comparison.json",rows)
    control_acc=float(((compact_mix[dev]>=.5)==old.y[dev]).mean())
    qualified=[r for r in rows if r["family"]=="tfm" and r["view"]!="compact" and r["gain"]>=.001 and r["improved_folds"]>=3
               and r["log_loss"]<=r["reference_log_loss"]+.002 and r["accuracy"]>=control_acc+.0005]
    if not qualified:
        _write_json(report/"summary.json",dict(status="no_candidate",stage="development",development=rows,
                    completed_repeats=0,kaggle_score=None,reason="No prespecified input passed the full development gate"))
        print("Input development gate failed; original retained",flush=True);return
    winner=sorted(qualified,key=lambda r:(-r["accuracy"],r["log_loss"],VIEWS.index(r["view"])))[0]
    view=winner["view"];_write_json(report/"selected_view.json",winner)

    def original_reference(ti,vi,tag):
        total=np.zeros(len(vi));reference_paths=[]
        for j,(weight,component) in enumerate(zip(plan["weights"],plan["members"])):
            cfg=component["config"];b=FeatureBuilder(cfg["feature_set"],cfg["strategy"]).fit(old.features.iloc[ti])
            X,V=b.transform(old.features.iloc[ti]),b.transform(old.features.iloc[vi])
            encoder=InputEncoder(cfg["family"]).fit(X);X,V=encoder.transform(X),encoder.transform(V)
            for seed in component["seeds"]:
                stem=f"reference_{tag}_component_{j}_seed_{seed}"
                cache,modelpath,recordpath=out/f"{stem}.npz",modeldir/f"{stem}.joblib",report/f"{stem}.json"
                if cache.exists() and modelpath.exists() and recordpath.exists():
                    z=np.load(cache);np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
                    assert sha(modelpath)==json.loads(recordpath.read_text(encoding="utf-8"))["model_sha256"];p=z["probability"]
                else:
                    model=estimator(dict(cfg,seed=seed),encoder.cat_columns_,4);model.fit(X,old.y[ti]);p=model.predict_proba(V)[:,1]
                    joblib.dump(dict(builder=b,encoder=encoder,model=model),modelpath,compress=3)
                    np.savez_compressed(cache,training=ti,validation=vi,probability=p)
                    _write_json(recordpath,dict(model_sha256=sha(modelpath),component=j,seed=seed,tag=tag))
                    print(f"New split {stem} ready",flush=True)
                total+=weight/len(component["seeds"])*p;reference_paths.append(stem)
        _write_json(report/f"reference_{tag}.json",dict(members=reference_paths))
        return total

    repeats=[]
    for repeat,seed in enumerate(SPLIT_SEEDS,1):
        ref,new,ctl=[np.full(len(old.y),np.nan) for _ in range(3)];folds=np.zeros(len(old.y),int)
        for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(old.features,groups=old.features.GroupId),1):
            tag=f"repeat_{repeat}_fold_{fold}"
            assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
            ref[vi]=original_reference(ti,vi,tag)
            new[vi]=fitted(view,"tfm",ti,vi,tag);ctl[vi]=fitted("compact","tfm",ti,vi,tag);folds[vi]=fold
        alternative=.5*ref+.5*new;compact=.5*ref+.5*ctl
        a,c=compare(old.y,ref,alternative,folds),compare(old.y,ref,compact,folds)
        repeats.append(dict(repeat=repeat,selected=a,compact=c))
        _write_json(report/"repeat_results.json",repeats)
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=folds,reference=ref,
                                    component=new,compact_component=ctl,alternative=alternative,compact_alternative=compact)),report/f"repeat_{repeat}_predictions.csv")
        if a["gain"]<0 or a["log_loss"]>a["reference_log_loss"]+.002 or (repeat==1 and a["improved_folds"]==0):
            _write_json(report/"summary.json",dict(status="no_candidate",stage="sensitivity",view=view,development=rows,repeats=repeats,
                        completed_repeats=repeat,kaggle_score=None,reason="New-partition non-regression/loss/futility gate failed"));return
    passed=(np.mean([r["selected"]["gain"] for r in repeats])>=.001 and sum(r["selected"]["improved_folds"] for r in repeats)>=6 and
            np.mean([r["selected"]["accuracy"]-r["compact"]["accuracy"] for r in repeats])>=.0005)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",stage="sensitivity",view=view,development=rows,repeats=repeats,
                    completed_repeats=2,kaggle_score=None,reason="New-partition mean/fold/input-increment gate failed"));return
    _write_json(report/"qualified.json",dict(view=view,development=winner,repeats=repeats))
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=[dict(m,weight=.5*m["weight"]) for m in bundle["members"]];identity=IdentityTransform()
    canonical=pd.read_csv(old.report/"oof_predictions.csv",float_precision="round_trip");oof=.5*canonical.probability.to_numpy()
    for fold in range(1,6):
        ti,vi=np.flatnonzero(old.fold_of!=fold),np.flatnonzero(old.fold_of==fold)
        oof[vi]+=.5*fitted(view,"tfm",ti,vi,f"final_fold_{fold}")
        model=joblib.load(modeldir/f"tfm_{view}_final_fold_{fold}.joblib")
        members.append(dict(weight=.1,builder=identity,encoder=identity,model=model))
    modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    p,submission=predict_ensemble(modelpath);again,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(p,again,atol=2e-6,rtol=0);pd.testing.assert_frame_equal(submission,check)
    _write_csv(submission,out/"submission_tabicl_input4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=p,prediction=p>=.5)),out/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,probability=oof)),report/"canonical_oof.csv")
    _write_json(report/"summary.json",dict(status="candidate_generated",view=view,development=rows,repeats=repeats,completed_repeats=2,
                kaggle_score=None,models=50,canonical_accuracy=float(((oof>=.5)==old.y).mean()),reload_max_error=float(np.max(np.abs(p-again)))))


if __name__=="__main__":main()
