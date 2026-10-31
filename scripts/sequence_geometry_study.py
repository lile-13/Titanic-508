"""A controlled, target-free cabin-coordinate addition with fixed paired gates."""
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.audit_unlabeled_views import stats as compare
from src import preprocess as pp
from src.competition_features import InputEncoder
from src.optimization import SearchRunner, estimator
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.sequence_geometry import SequenceGeometryBuilder, COLUMNS
from src.train import _write_json, _write_csv


NAME = "sequence_geometry_20261031"
PARENT = "tabicl_probability_mean_20261030"
ADDITIONAL = "tabm_20261031"
WEIGHT = .25


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gate(rows):
    return (len(rows)==2 and all(r["gain"]>=0 and r["feature_gain_over_control"]>=0
             and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows)
        and np.mean([r["gain"] for r in rows])>=.001
        and np.mean([r["feature_gain_over_control"] for r in rows])>=.0005
        and sum(r["improved_folds"] for r in rows)>=6)


def main():
    old = SearchRunner()
    report, out, models = EXPERIMENTS_DIR/NAME, PROJECT_ROOT/"outputs"/NAME, PROJECT_ROOT/"models"/NAME
    for d in [report,out,models]:
        d.mkdir(parents=True,exist_ok=True)
    cfg = dict(read(old.report/"frozen_plan.json")["members"][2]["config"],seed=42)
    public = pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public
    protocol = dict(name=NAME,estimator_config=cfg,estimator_usage="Reuse only fixed LightGBM parameters; both arms use compact18 inputs, enabled arm adds six geometry columns",
        weight=WEIGHT,threshold=.5,extra_columns=COLUMNS,
        label_free_fit="All coordinates, linear slopes, median residual/MAD, same-cabin occupancy, nearest other cabin gap and passenger count within +/-10 cabins use the single combined public train/test feature pool. No Transported values or train/test source indicator enter the maps.",
        hypothesis="CabinNum and booking GroupId are almost perfectly monotonic within Deck/Side. An explicit diagonal residual may expose deviations more simply than axis-aligned tree splits; public local counts add cabin context. This is an unproven hypothesis, not a causal claim.",
        selection="One frozen model seed42, six fixed features and 25% addition. Matched compact-only control. No feature selection, radius, polynomial-degree, weight or threshold search.",
        gate="Both gains nonnegative, mean incumbent gain>=.001, >=6/10 strictly improved folds, per-repeat loss deterioration<=.002, enriched accuracy>=control each repeat and mean enriched-control>=.0005. First negative gain, negative enriched-control, bad loss or first zero-improved repeat stops stage. Both historical stages must pass for production.",
        stages=dict(paired=[20261030,20261107],additional_paired=[20261031,20261108]),
        caveat="Existing historical partitions and known labels; adaptive paired evidence, not an independent holdout. Increasing view count and prior TabM failures remain failed.",
        references={f"{run}/fresh_repeat_{r}_predictions.csv":sha(EXPERIMENTS_DIR/run/f"fresh_repeat_{r}_predictions.csv") for run in [PARENT,ADDITIONAL] for r in [1,2]},
        reference_audits={run:sha(PROJECT_ROOT/f"reports/checks/{run}.json") for run in [PARENT,ADDITIONAL]},
        original_plan_sha256=sha(old.report/"frozen_plan.json"),
        data_hashes=read(old.report/"protocol.json")["data_hashes"],
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/sequence_geometry_study.py","src/sequence_geometry.py","src/tabicl_inputs.py",
            "src/focused_features.py","src/competition_features.py","src/preprocess.py","src/optimization.py","scripts/audit_unlabeled_views.py"]},
        environment={p:importlib.metadata.version(p) for p in ["lightgbm","numpy","pandas","scikit-learn"]})
    path = report/"protocol.json"
    if path.exists():
        assert read(path)==protocol,"Frozen geometry study changed"
    else:
        _write_json(path,protocol)
    for p,digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==digest
    if (report/"summary.json").exists():
        print("Completed geometry study; audit rather than retrain",flush=True)
        return
    stats = SequenceGeometryBuilder(True).fit(old.features,public).geometry_.regions_
    _write_json(report/"public_coordinate_maps.json",{key:dict(slope=s["slope"],intercept=s["intercept"],residual_center=s["center"],
        residual_scale=s["scale"],public_passengers=len(s["passengers"]),distinct_cabins=len(s["unique"])) for key,s in stats.items()})
    stages = {}
    for stage,run,seeds in [("paired",PARENT,[20261030,20261107]),("additional_paired",ADDITIONAL,[20261031,20261108])]:
        rows = []
        for repeat,seed in enumerate(seeds,1):
            table = pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            reference = table.alternative.to_numpy() if stage=="paired" else table.reference.to_numpy()
            p = {False:np.full(len(old.y),np.nan),True:np.full(len(old.y),np.nan)}
            fold_of = np.zeros(len(old.y),int)
            for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(old.features,groups=old.features.GroupId),1):
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                assert (table.fold.iloc[vi]==fold).all()
                fold_of[vi] = fold
                for enabled in [False,True]:
                    tag = f"{stage}_repeat_{repeat}_fold_{fold}_{'geometry' if enabled else 'control'}"
                    cache,mp,rp = out/f"{tag}.npz",models/f"{tag}.joblib",report/f"{tag}.json"
                    if cache.exists() and mp.exists() and rp.exists():
                        z = np.load(cache)
                        np.testing.assert_array_equal(z["training"],ti)
                        np.testing.assert_array_equal(z["validation"],vi)
                        assert sha(mp)==read(rp)["model_sha256"]
                        p[enabled][vi] = z["probability"]
                        continue
                    builder = SequenceGeometryBuilder(enabled).fit(old.features.iloc[ti],public)
                    X = builder.transform(old.features.iloc[ti])
                    encoder = InputEncoder("lightgbm").fit(X)
                    model = estimator(cfg,encoder.cat_columns_,threads=4)
                    model.fit(encoder.transform(X),old.y[ti])
                    q = model.predict_proba(encoder.transform(builder.transform(old.features.iloc[vi])))[:,1]
                    assert np.isfinite(q).all()
                    p[enabled][vi] = q
                    joblib.dump(dict(builder=builder,encoder=encoder,model=model),mp,compress=3)
                    np.savez_compressed(cache,training=ti,validation=vi,probability=q)
                    _write_json(rp,dict(model_sha256=sha(mp),enabled=enabled,columns=list(X),training_rows=len(ti),validation_rows=len(vi)))
                print(f"Geometry {stage} repeat {repeat} fold {fold}: both arms completed",flush=True)
            control,alternative = (1-WEIGHT)*reference+WEIGHT*p[False],(1-WEIGHT)*reference+WEIGHT*p[True]
            row = dict(repeat=repeat,**compare(old.y,reference,alternative,fold_of),
                control_accuracy=float(((control>=.5)==old.y).mean()),
                feature_gain_over_control=float(((alternative>=.5)==old.y).mean()-((control>=.5)==old.y).mean()),
                standalone_control_accuracy=float(((p[False]>=.5)==old.y).mean()),standalone_geometry_accuracy=float(((p[True]>=.5)==old.y).mean()))
            rows.append(row)
            _write_json(report/f"{stage}_results.json",rows)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=fold_of,
                reference=reference,control_component=p[False],geometry_component=p[True],control=control,alternative=alternative)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            print(f"Geometry {stage} repeat {repeat}: net {row['corrected']-row['new_errors']:+d}, feature gain {row['feature_gain_over_control']:+.6f}, {row['improved_folds']}/5 improved",flush=True)
            if row["gain"]<0 or row["feature_gain_over_control"]<0 or row["log_loss"]>row["reference_log_loss"]+.002 or (repeat==1 and row["improved_folds"]==0):
                break
        stages[stage] = rows
        if not gate(rows):
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None,
                reason="Fixed public geometry addition failed incumbent/control gate; no post-result selection"))
            return
    _write_json(report/"qualified.json",dict(stages=stages))
    _write_json(report/"summary.json",dict(status="qualified_for_production",stages=stages,kaggle_score=None))


if __name__=="__main__":
    main()
