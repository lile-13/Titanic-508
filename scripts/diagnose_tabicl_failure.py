"""Descriptive bad-fold audit and a fixed-validation grouped learning curve."""
import gc
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from scripts.tabicl_study import compare
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabicl_models import fit_member, PARAMS
from src.train import _write_json, _write_csv

NAME="tabicl_failure_diagnosis_20261030"
FRACTIONS=[.5,.75]
SEEDS=[5101,5102,5103]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def cohorts(frame):
    g=frame.Deck.eq("G").fillna(False).to_numpy(bool)
    cold=frame.CryoSleep.eq(True).fillna(False).to_numpy(bool)
    awake=frame.CryoSleep.eq(False).fillna(False).to_numpy(bool)
    child=frame.Age.le(12).fillna(False).to_numpy(bool)
    return dict(all=np.ones(len(frame),bool),G=g,non_G=~g,G_cold=g&cold,
                G_awake_child=g&awake&child,G_awake_other=g&awake&~child,G_unknown_sleep=g&~(cold|awake))


def summarize(y,ref,alt,mask):
    before,after=(ref[mask]>=.5)==y[mask],(alt[mask]>=.5)==y[mask]
    truth=y[mask];r=ref[mask]>=.5;a=alt[mask]>=.5
    return dict(rows=int(mask.sum()),reference_accuracy=float(before.mean()),accuracy=float(after.mean()),
                net_correct=int(after.sum()-before.sum()),corrected=int((~before&after).sum()),new_errors=int((before&~after).sum()),
                old_false_positive=int((r&~truth.astype(bool)).sum()),new_false_positive=int((a&~truth.astype(bool)).sum()),
                old_false_negative=int((~r&truth.astype(bool)).sum()),new_false_negative=int((~a&truth.astype(bool)).sum()),
                observed_rate=float(truth.mean()),reference_mean_probability=float(ref[mask].mean()),mean_probability=float(alt[mask].mean()))


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/NAME;out=PROJECT_ROOT/"outputs"/NAME;modeldir=PROJECT_ROOT/"models"/NAME
    for path in [report,out,modeldir]:path.mkdir(parents=True,exist_ok=True)
    parent=EXPERIMENTS_DIR/"tabicl_paired_20261030"
    protocol=dict(name=NAME,diagnostic_repeat=2,diagnostic_fold=3,chosen_from_historical_worst_fold=True,
        fractions=FRACTIONS,seeds=SEEDS,params=PARAMS,group_sampling="Permute unique outer-training GroupIds per seed; take first floor(fraction*n_groups). Fractions nested per seed; fixed validation unchanged.",
        interpretation="Selected-fold descriptive sensitivity only. No sampling fraction or seed is promoted; does not establish a causal explanation or independent gain. Full-training reference uses cached original ensemble; only TFM context size changes.",
        parent_protocol_sha256=sha(parent/"protocol.json"),parent_predictions_sha256=sha(parent/"repeat_2_predictions.csv"),
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/diagnose_tabicl_failure.py","src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py"]})
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("Diagnostic protocol changed")
    else:_write_json(path,protocol)
    if (report/"summary.json").exists():print("Completed fixed diagnostic",flush=True);return
    table=pd.read_csv(parent/"repeat_2_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
    ref,alt=table.reference.to_numpy(),table.alternative.to_numpy();folds=table.fold.to_numpy()
    rows=[];shift=[]
    for fold in range(1,6):
        z=np.load(PROJECT_ROOT/f"outputs/fixed_weight_confirmation_20261029/repeat_2_fold_{fold}.npz")
        ti,vi=z["training"],z["validation"]
        for name,mask in cohorts(old.features.iloc[vi]).items():
            rows.append(dict(fold=fold,cohort=name,**summarize(old.y[vi],ref[vi],alt[vi],mask)))
        train=old.features.iloc[ti];valid=old.features.iloc[vi]
        train=train[train.Deck.eq("G")];valid=valid[valid.Deck.eq("G")]
        for col in ["Age","GroupSize","GroupIdNum","CabinNum",*pp.SPEND_COLS]:
            a,b=train[col].to_numpy(dtype=float),valid[col].to_numpy(dtype=float)
            a,b=a[np.isfinite(a)],b[np.isfinite(b)]
            shift.append(dict(fold=fold,column=col,kind="numeric",train_n=len(a),valid_n=len(b),
                              train_mean=float(np.mean(a)),valid_mean=float(np.mean(b)),ks_statistic=float(ks_2samp(a,b).statistic)))
        for col in ["CryoSleep","Side","Destination","HomePlanet"]:
            a=train[col].astype("string").fillna("unknown").value_counts(normalize=True)
            b=valid[col].astype("string").fillna("unknown").value_counts(normalize=True)
            for value in sorted(set(a.index)|set(b.index)):
                shift.append(dict(fold=fold,column=col,kind="categorical",value=value,train_share=float(a.get(value,0)),valid_share=float(b.get(value,0))))
    _write_csv(pd.DataFrame(rows),report/"fold_cohorts.csv");_write_csv(pd.DataFrame(shift),report/"G_train_validation_shift.csv")
    base=np.load(PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029/repeat_2_fold_3.npz")
    ti,vi=base["training"],base["validation"]
    groups=np.unique(old.features.GroupId.iloc[ti]);curve=[]
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(groups)
        for fraction in FRACTIONS:
            selected=set(order[:int(np.floor(len(order)*fraction))])
            training=ti[old.features.GroupId.iloc[ti].isin(selected).to_numpy()]
            tag=f"seed_{seed}_fraction_{fraction}"
            cache,modelpath=out/f"{tag}.npz",modeldir/f"{tag}.joblib"
            if cache.exists() and modelpath.exists():
                z=np.load(cache);np.testing.assert_array_equal(z["training"],training);np.testing.assert_array_equal(z["validation"],vi);q=z["probability"]
            else:
                assert not set(old.features.GroupId.iloc[training])&set(old.features.GroupId.iloc[vi])
                m=fit_member(old.features.iloc[training],old.y[training],public);q=m.predict_proba(old.features.iloc[vi])[:,1]
                joblib.dump(m,modelpath,compress=3);np.savez_compressed(cache,training=training,validation=vi,probability=q)
                _write_json(report/f"{tag}.json",dict(model_sha256=sha(modelpath),training_rows=len(training),training_groups=len(selected)))
                del m;gc.collect();torch.cuda.empty_cache()
            mixed=.5*ref[vi]+.5*q
            for name,mask in cohorts(old.features.iloc[vi]).items():
                curve.append(dict(seed=seed,fraction=fraction,training_rows=len(training),training_groups=len(selected),cohort=name,
                                  **summarize(old.y[vi],ref[vi],mixed,mask)))
            print(f"Fixed-validation learning curve {tag} ready",flush=True)
    for name,mask in cohorts(old.features.iloc[vi]).items():
        curve.append(dict(seed=None,fraction=1.,training_rows=len(ti),training_groups=len(groups),cohort=name,
                          **summarize(old.y[vi],ref[vi],alt[vi],mask)))
    _write_csv(pd.DataFrame(curve),report/"fixed_validation_learning_curve.csv")
    _write_json(report/"summary.json",dict(status="diagnostic_only",selected_repeat=2,selected_fold=3,
        full_context=[r for r in rows if r["fold"]==3],models_fitted=6,sampling_fractions_not_selected=True,
        caveat="Same deliberately chosen difficult validation fold; subsamples overlap, not independent estimates; descriptive shift does not prove causation"))


if __name__=="__main__":main()
