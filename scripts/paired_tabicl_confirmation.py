"""A separately registered adaptive follow-up; the original failed gate stays failed."""
import gc
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.tabicl_study import compare
from src import preprocess as pp
from src.optimization import SearchRunner, predict_ensemble
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.probability_calibration import IdentityTransform
from src.tabicl_models import fit_member, PARAMS, CHECKPOINT, CHECKPOINT_SHA256
from src.train import _write_json, _write_csv

NAME = "tabicl_paired_20261030"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = SearchRunner()
    report, out, modeldir = EXPERIMENTS_DIR / NAME, PROJECT_ROOT / "outputs" / NAME, PROJECT_ROOT / "models" / NAME
    for directory in [report, out, modeldir]:
        directory.mkdir(parents=True, exist_ok=True)
    parent = EXPERIMENTS_DIR / "tabicl_20261030"
    original_result = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    assert original_result["status"] == "no_candidate" and original_result["completed_repeats"] == 0
    assert sha(PROJECT_ROOT / "models/tabicl_pretrained" / CHECKPOINT) == CHECKPOINT_SHA256
    plan = json.loads((old.report / "frozen_plan.json").read_text(encoding="utf-8"))
    protocol = dict(name=NAME, params=PARAMS, fixed_new_weight=.5, threshold=.5,
                    rationale="Original TabICL study failed its 3/4-fold consistency gate and remains failed. Its +14 development correct and logloss decrease about .008 motivate one separate, adaptive fixed-choice follow-up; no model, feature, weight, seed or threshold change.",
                    gate="Two paired grouped five-fold repeats: mean accuracy gain >=.001, both repeats nonnegative, >=6/10 folds improve, each logloss deterioration <=.002. Stop after repeat 1 if negative, loss failure, or no improved folds. No replacement split and no retuning.",
                    split_seeds=[20261029,20261106], folds=5,
                    caveat="Follow-up decided after inspecting the original failed development result and many previous studies. Same historical rows and partitions; sensitivity evidence only, not independent or unbiased confirmation. A candidate remains pending actual Kaggle score.",
                    parent_protocol_sha256=sha(parent / "protocol.json"), parent_summary_sha256=sha(parent / "summary.json"),
                    original_plan_sha256=sha(old.report / "frozen_plan.json"),
                    baseline_protocol_sha256=sha(EXPERIMENTS_DIR / "fixed_weight_confirmation_20261029/protocol.json"),
                    checkpoint_sha256=CHECKPOINT_SHA256,
                    data_hashes=json.loads((old.report / "protocol.json").read_text(encoding="utf-8"))["data_hashes"],
                    sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/paired_tabicl_confirmation.py","scripts/tabicl_study.py",
                        "src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/optimization.py","src/preprocess.py","features/folds.csv","requirements-tabicl.txt"]},
                    environment={p:importlib.metadata.version(p) for p in ["torch","tabicl","numpy","pandas","scikit-learn"]})
    path=report / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Separate follow-up protocol changed; use a new run")
    else:
        _write_json(path,protocol)
    if (report / "summary.json").exists():
        print("Completed fixed follow-up; use audit instead of retraining",flush=True)
        return
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)

    def fitted(ti,vi,tag):
        cache,modelpath,recordpath=out/f"{tag}.npz",modeldir/f"{tag}.joblib",report/f"{tag}.json"
        if cache.exists() and modelpath.exists() and recordpath.exists():
            saved=np.load(cache)
            np.testing.assert_array_equal(saved["training"],ti);np.testing.assert_array_equal(saved["validation"],vi)
            assert sha(modelpath)==json.loads(recordpath.read_text(encoding="utf-8"))["model_sha256"]
            return saved["probability"]
        assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
        m=fit_member(old.features.iloc[ti],old.y[ti],public)
        p=m.predict_proba(old.features.iloc[vi])[:,1]
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        joblib.dump(m,modelpath,compress=3)
        np.savez_compressed(cache,training=ti,validation=vi,probability=p)
        _write_json(recordpath,dict(tag=tag,model_sha256=sha(modelpath),training_rows=len(ti),validation_rows=len(vi),columns=list(m.training_X)))
        del m
        gc.collect();torch.cuda.empty_cache()
        print(f"Fixed paired TabICL {tag} ready",flush=True)
        return p

    rows=[]
    for repeat in [1,2]:
        ref,p,alt=[np.full(len(old.y),np.nan) for _ in range(3)]
        folds=np.zeros(len(old.y),int)
        for fold in range(1,6):
            tag=f"repeat_{repeat}_fold_{fold}"
            base=np.load(PROJECT_ROOT / f"outputs/fixed_weight_confirmation_20261029/{tag}.npz")
            ti,vi=base["training"],base["validation"]
            p[vi]=fitted(ti,vi,tag)
            ref[vi]=base["probability"] @ np.array(plan["weights"])
            alt[vi]=.5*ref[vi]+.5*p[vi]
            folds[vi]=fold
        r=dict(repeat=repeat,**compare(old.y,ref,alt,folds));rows.append(r)
        _write_json(report / "repeat_results.json",rows)
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=folds,reference=ref,alternative=alt,tabicl=p)),
                   report / f"repeat_{repeat}_predictions.csv")
        print(f"Paired TabICL repeat {repeat}: gain {r['gain']:+.6f}, improved {r['improved_folds']}/5",flush=True)
        if r["gain"] < 0 or r["log_loss"] > r["reference_log_loss"]+.002 or (repeat==1 and r["improved_folds"]==0):
            _write_json(report / "summary.json",dict(status="no_candidate",completed_repeats=repeat,repeats=rows,kaggle_score=None,
                        original_development_gate_still_failed=True,reason="Separate fixed-choice non-regression/futility gate failed"))
            return
    passed=np.mean([r["gain"] for r in rows])>=.001 and sum(r["improved_folds"] for r in rows)>=6
    if not passed:
        _write_json(report / "summary.json",dict(status="no_candidate",completed_repeats=2,repeats=rows,kaggle_score=None,
                    original_development_gate_still_failed=True,reason="Separate fixed-choice mean/fold-count gate failed"))
        return
    _write_json(report / "qualified.json",dict(repeats=rows,mean_gain=float(np.mean([r["gain"] for r in rows])),
                original_development_gate_still_failed=True,post_selection_sensitivity_only=True))
    original=joblib.load(PROJECT_ROOT / "models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report / "oof_predictions.csv",float_precision="round_trip")
    oof=.5*canonical.probability.to_numpy()
    members=[dict(m,weight=.5*m["weight"]) for m in original["members"]]
    identity=IdentityTransform()
    for fold in range(1,6):
        ti,vi=np.flatnonzero(old.fold_of!=fold),np.flatnonzero(old.fold_of==fold)
        oof[vi]+=.5*fitted(ti,vi,f"final_fold_{fold}")
        m=joblib.load(modeldir / f"final_fold_{fold}.joblib")
        members.append(dict(weight=.1,builder=identity,encoder=identity,model=m))
    modelpath=modeldir / "ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5),data_hashes=original["data_hashes"]),modelpath,compress=3)
    p,submission=predict_ensemble(modelpath)
    again,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(p,again,atol=2e-6,rtol=0)
    pd.testing.assert_frame_equal(submission,check)
    _write_csv(submission,out / "submission_tabicl4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=p,prediction=p>=.5)),out / "test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,probability=oof)),report / "canonical_oof.csv")
    old_test=pd.read_csv(old.output / "test_predictions.csv")
    _write_json(report / "summary.json",dict(status="candidate_generated",completed_repeats=2,repeats=rows,kaggle_score=None,
                original_development_gate_still_failed=True,mean_gain=float(np.mean([r["gain"] for r in rows])),
                improved_folds=sum(r["improved_folds"] for r in rows),models=50,tabicl_contexts=5,
                canonical_accuracy=float(((oof>=.5)==old.y).mean()),changed_test_labels=int(np.sum((p>=.5)!=old_test.prediction.to_numpy())),
                reload_max_error=float(np.max(np.abs(p-again))),interpretation="Adaptive pending candidate; original 0.81108 remains incumbent until platform evidence"))


if __name__=="__main__":
    main()
