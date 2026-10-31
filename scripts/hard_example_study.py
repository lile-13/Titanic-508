"""Train-only difficulty mining and a matched G-deck factorial comparison."""
import hashlib
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.audit_unlabeled_views import stats
from src import preprocess as pp
from src.hard_example_training import ARMS, training_weights, fit_member, predict
from src.optimization import SearchRunner
from src.paths import EXPERIMENTS_DIR, PROJECT_ROOT
from src.train import _write_csv, _write_json

NAME = "hard_examples_g_20261031"
STAGES = [("paired", "tabicl_probability_mean_20261030", [20261030, 20261107]),
          ("additional", "tabm_20261031", [20261031, 20261108])]
INNER_SEED = 731026
WEIGHT = .25


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(y, reference, alternative, control, folds, g):
    return dict(**stats(y, reference, alternative, folds),
        control_accuracy=float(((control>=.5)==y).mean()),
        gain_over_control=float(((alternative>=.5)==y).mean()-((control>=.5)==y).mean()),
        g=stats(y[g],reference[g],alternative[g],folds[g]),
        non_g=stats(y[~g],reference[~g],alternative[~g],folds[~g]))


def passes(rows, arm):
    return (len(rows)==2 and all(r["gain"]>=0 and r["gain_over_control"]>=0
        and r["log_loss"]<=r["reference_log_loss"]+.002
        and r["g"]["gain"]>=0 and r["non_g"]["gain"]>=0 for r in rows)
        and np.mean([r["gain"] for r in rows])>=.001
        and np.mean([r["gain_over_control"] for r in rows])>=.0005
        and sum(r["improved_folds"] for r in rows)>=6
        and (arm!="hard_g" or np.mean([r["g"]["gain"] for r in rows])>=.003))


def main():
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,PROJECT_ROOT/"outputs"/NAME,PROJECT_ROOT/"models"/NAME
    for directory in [report,out,models]:
        directory.mkdir(parents=True,exist_ok=True)
    config=dict(read(old.report/"frozen_plan.json")["members"][1]["config"],seed=42)
    protocol=dict(name=NAME,config=config,inner_seed=INNER_SEED,inner_group_folds=3,
        arms=ARMS,weight=WEIGHT,threshold=.5,stages=STAGES,
        difficulty="Within each outer training fold, obtain 3-fold group-isolated unweighted CatBoost predictions. True-class probability defines bounded raw hard weight 2-p_true. G multiplier=1.5; normalize each arm to mean1. Raw weight ratio at most3.",
        candidates="Two predeclared candidates: hard vs control; hard_g vs g_control. Both compared with actual incumbent. All predictions use fixed .75 incumbent+.25 new component, threshold .5. Four arms form a matched 2x2 factorial; no post-result configuration or coefficient search.",
        gate="Two nonnegative repeat gains vs incumbent and matched control; mean incumbent gain>=.001, mean matched gain>=.0005, >=6/10 improved folds, logloss deterioration<=.002. G and non-G must not regress each repeat; hard_g mean G gain>=.003. A negative repeat or zero improved folds stops that candidate. Both historical stages required before production.",
        persistent_errors="The 1203 global historical error IDs are never read during fitting or weight creation; label-selected cohorts may only be used in post-fit descriptive summaries.",
        caveat="Adaptive hypotheses on the same historical passengers; multiple predeclared hypotheses, not an independent blind holdout. Reweighting need not create new information and may emphasize noise. Old G-weighted failures remain failed.",
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/hard_example_study.py","src/hard_example_training.py","src/competition_features.py","src/optimization.py","src/preprocess.py","scripts/audit_unlabeled_views.py"]},
        data_hashes=read(old.report/"protocol.json")["data_hashes"],
        references={f"{run}/fresh_repeat_{r}_predictions.csv":sha(EXPERIMENTS_DIR/run/f"fresh_repeat_{r}_predictions.csv") for _,run,_ in STAGES for r in [1,2]},
        environment={p:importlib.metadata.version(p) for p in ["catboost","numpy","pandas","scikit-learn"]})
    protocol=json.loads(json.dumps(protocol))
    if (report/"protocol.json").exists():
        assert read(report/"protocol.json")==protocol,"Frozen hard-example study changed"
    else:
        _write_json(report/"protocol.json",protocol)
    for p,digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==digest
    if (report/"summary.json").exists():
        print("Completed study; audit rather than retrain")
        return
    stages={}
    active={"hard","hard_g"}
    for stage,run,seeds in STAGES:
        rows={arm:[] for arm in active}
        stage_active=active.copy()
        for repeat,seed in enumerate(seeds,1):
            table=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            reference=table.alternative.to_numpy() if stage=="paired" else table.reference.to_numpy()
            probabilities={arm:np.full(len(old.y),np.nan) for arm in ARMS}
            folds=np.zeros(len(old.y),int)
            for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(old.features,groups=old.features.GroupId),1):
                assert (table.fold.iloc[vi]==fold).all()
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                folds[vi]=fold
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                difficulty=np.full(len(ti),np.nan)
                inner_of=np.zeros(len(ti),int)
                for inner,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=INNER_SEED).split(ti,groups=old.features.GroupId.iloc[ti]),1):
                    stem=f"{tag}_teacher_{inner}"
                    cache,mp,rp=out/f"{stem}.npz",models/f"{stem}.joblib",report/f"{stem}.json"
                    if rp.exists():
                        assert sha(mp)==read(rp)["model_sha256"]
                        z=np.load(cache)
                        np.testing.assert_array_equal(z["training"],ti[it])
                        np.testing.assert_array_equal(z["validation"],ti[iv])
                        p=z["probability"]
                    else:
                        start=time.perf_counter()
                        model=fit_member(old.features.iloc[ti[it]],old.y[ti[it]],config)
                        p=predict(model,old.features.iloc[ti[iv]])
                        joblib.dump(model,mp,compress=3)
                        np.savez_compressed(cache,training=ti[it],validation=ti[iv],probability=p)
                        _write_json(rp,dict(model_sha256=sha(mp),seconds=time.perf_counter()-start))
                    difficulty[iv]=p
                    inner_of[iv]=inner
                assert np.isfinite(difficulty).all()
                g=old.features.Deck.iloc[ti].eq("G").fillna(False).to_numpy(bool)
                saved_weights={}
                for arm in ARMS:
                    weights=training_weights(old.y[ti],difficulty,g,arm)
                    saved_weights[arm]=weights
                    stem=f"{tag}_{arm}"
                    cache,mp,rp=out/f"{stem}.npz",models/f"{stem}.joblib",report/f"{stem}.json"
                    if rp.exists():
                        assert sha(mp)==read(rp)["model_sha256"]
                        z=np.load(cache)
                        np.testing.assert_array_equal(z["weights"],weights)
                        p=z["probability"]
                    else:
                        start=time.perf_counter()
                        model=fit_member(old.features.iloc[ti],old.y[ti],config,weights)
                        p=predict(model,old.features.iloc[vi])
                        joblib.dump(model,mp,compress=3)
                        np.savez_compressed(cache,training=ti,validation=vi,weights=weights,probability=p)
                        _write_json(rp,dict(model_sha256=sha(mp),seconds=time.perf_counter()-start,
                            weight_min=float(weights.min()),weight_max=float(weights.max()),mean_weight_g=float(weights[g].mean()),mean_weight_non_g=float(weights[~g].mean())))
                    probabilities[arm][vi]=p
                np.savez_compressed(out/f"{tag}_difficulty.npz",training=ti,validation=vi,inner_fold=inner_of,probability=difficulty)
                print(f"Hard-example {tag}: 3 teachers and 4 matched arms completed",flush=True)
            g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
            mixtures={arm:(1-WEIGHT)*reference+WEIGHT*p for arm,p in probabilities.items()}
            for arm in sorted(stage_active):
                control=mixtures["control" if arm=="hard" else "g_control"]
                row=dict(repeat=repeat,**metrics(old.y,reference,mixtures[arm],control,folds,g))
                rows[arm].append(row)
                print(f"Hard-example {arm} repeat {repeat}: net {row['corrected']-row['new_errors']:+d}, G net {row['g']['corrected']-row['g']['new_errors']:+d}",flush=True)
                if row["gain"]<0 or row["gain_over_control"]<0 or row["g"]["gain"]<0 or row["non_g"]["gain"]<0 or row["log_loss"]>row["reference_log_loss"]+.002 or row["improved_folds"]==0:
                    stage_active.remove(arm)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=folds,reference=reference,
                **{f"component_{a}":p for a,p in probabilities.items()},**mixtures)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report/f"{stage}_results.json",rows)
            if not stage_active:
                break
        stages[stage]=rows
        active={arm for arm in stage_active if passes(rows[arm],arm)}
        if not active:
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None))
            return
    _write_json(report/"summary.json",dict(status="qualified_for_production",qualified_arms=sorted(active),stages=stages,kaggle_score=None))


if __name__=="__main__":
    main()
