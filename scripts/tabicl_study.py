"""One fixed local TabICLv2/old-ensemble blend with grouped paired checks."""
import gc
import hashlib
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss

from src import preprocess as pp
from src.optimization import SearchRunner, config_id, predict_ensemble
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.probability_calibration import IdentityTransform
from src.tabicl_models import PARAMS, CHECKPOINT, CHECKPOINT_SHA256, fit_member
from src.train import _write_json, _write_csv, make_submission

NAME = "tabicl_20261030"
WEIGHT = .5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(y, ref, alt, folds):
    correct, new = (ref >= .5) == y, (alt >= .5) == y
    gains = [float(new[folds == f].mean() - correct[folds == f].mean()) for f in np.unique(folds)]
    return dict(reference_accuracy=float(correct.mean()), accuracy=float(new.mean()), gain=float(new.mean()-correct.mean()),
                reference_log_loss=float(log_loss(y, ref)), log_loss=float(log_loss(y, alt)),
                corrected=int((~correct & new).sum()), new_errors=int((correct & ~new).sum()),
                fold_gains=gains, improved_folds=sum(v > 0 for v in gains))


def main():
    old = SearchRunner()
    report, output, modeldir = EXPERIMENTS_DIR / NAME, PROJECT_ROOT / "outputs" / NAME, PROJECT_ROOT / "models" / NAME
    for directory in [report, output, modeldir]:
        directory.mkdir(parents=True, exist_ok=True)
    source = json.loads((report / "pretrained_source.json").read_text(encoding="utf-8"))
    assert source["sha256"] == CHECKPOINT_SHA256
    assert sha(PROJECT_ROOT / "models/tabicl_pretrained" / CHECKPOINT) == CHECKPOINT_SHA256
    plan = json.loads((old.report / "frozen_plan.json").read_text(encoding="utf-8"))
    protocol = dict(name=NAME, params=PARAMS, weight=WEIGHT, threshold=.5, feature_builder="FocusedBuilder(compact, resolve_planet=True)",
                    selection="One pretrained model, one feature set, one fixed 50/50 blend. No hyperparameter, seed, threshold, weight or cohort search. No fine-tuning.",
                    gate="Development gain >=.001, >=3/4 folds improve, logloss deterioration <=.002. Then two existing grouped five-fold repetitions: both nonnegative, mean gain >=.001, >=6/10 folds improve, loss deterioration <=.002 each. Stop at any failed stage; first repeated negative/loss/zero improved folds stops repeat 2.",
                    development_folds=[1,2,3,4], excluded_from_development=5,
                    paired_splits="Reuse fixed_weight_confirmation_20261029 component caches and GroupKFold seeds 20261029/20261106; all historical rows/partitions seen previously. This is adaptive paired sensitivity, not a new blind holdout.",
                    pretrained_source_sha256=sha(report / "pretrained_source.json"), checkpoint_sha256=CHECKPOINT_SHA256,
                    original_plan_sha256=sha(old.report / "frozen_plan.json"),
                    baseline_protocol_sha256=sha(EXPERIMENTS_DIR / "fixed_weight_confirmation_20261029/protocol.json"),
                    data_hashes=json.loads((old.report / "protocol.json").read_text(encoding="utf-8"))["data_hashes"],
                    sources={p:sha(PROJECT_ROOT / p) for p in ["scripts/tabicl_study.py", "scripts/prepare_tabicl.py", "src/tabicl_models.py",
                        "src/focused_features.py", "src/competition_features.py", "src/preprocess.py", "src/optimization.py", "features/folds.csv", "requirements-tabicl.txt"]},
                    environment={p:importlib.metadata.version(p) for p in ["torch","tabicl","numpy","pandas","scikit-learn","scipy"]},
                    gpu=torch.cuda.get_device_name(0), context="Public unlabeled features only; entire validation groups absent from in-context training labels; all inference stays local")
    path = report / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Frozen TabICL protocol changed; use a new experiment")
    else:
        _write_json(path, protocol)
    for name, digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir() / name) == digest
    if (report / "summary.json").exists():
        print("Completed frozen TabICL study; audit instead of retraining", flush=True)
        return
    public = pd.concat([old.features.drop(columns=pp.TARGET), old.test_features], ignore_index=True)
    assert pp.TARGET not in public

    def cached(ti, vi, tag):
        cache, modelpath, recordpath = output / f"{tag}.npz", modeldir / f"{tag}.joblib", report / f"{tag}.json"
        if cache.exists() and modelpath.exists() and recordpath.exists():
            saved = np.load(cache)
            np.testing.assert_array_equal(saved["training"], ti)
            np.testing.assert_array_equal(saved["validation"], vi)
            assert sha(modelpath) == json.loads(recordpath.read_text(encoding="utf-8"))["model_sha256"]
            return saved["probability"]
        assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
        member = fit_member(old.features.iloc[ti], old.y[ti], public)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        p = member.predict_proba(old.features.iloc[vi])[:,1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        joblib.dump(member, modelpath, compress=3)
        np.savez_compressed(cache, training=ti, validation=vi, probability=p)
        _write_json(recordpath, dict(tag=tag, model_sha256=sha(modelpath), training_rows=len(ti), validation_rows=len(vi),
                                    columns=list(member.training_X), standalone_accuracy=float(accuracy_score(old.y[vi], p>=.5)),
                                    elapsed_seconds=time.perf_counter()-started, gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
        del member
        gc.collect()
        torch.cuda.empty_cache()
        print(f"TabICL {tag}: ready", flush=True)
        return p

    dev = old.fold_of != 5
    ref = np.full(len(old.y), np.nan)
    ref[dev] = 0.
    for weight, component in zip(plan["weights"], plan["members"]):
        for seed in component["seeds"]:
            trial = config_id(dict(component["config"], seed=seed), "confirmation", [1,2,3,4])
            q = np.load(old.output / "search" / f"{trial}.npz")["probability"]
            assert np.isnan(q[~dev]).all()
            ref[dev] += weight / len(component["seeds"]) * q[dev]
    p = np.full(len(old.y), np.nan)
    for fold, (ti, vi) in enumerate(old.splits,1):
        assert not (old.fold_of[ti] == 5).any()
        p[vi] = cached(ti, vi, f"development_fold_{fold}")
    alt = (1-WEIGHT)*ref + WEIGHT*p
    d = compare(old.y[dev],ref[dev],alt[dev],old.fold_of[dev])
    passed = d["gain"] >= .001 and d["improved_folds"] >= 3 and d["log_loss"] <= d["reference_log_loss"]+.002
    _write_json(report / "development.json",dict(passed=bool(passed),**d))
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,
                                reference=ref,alternative=alt,tabicl=p)),report / "development_predictions.csv")
    print(f"TabICL development: {d['reference_accuracy']:.6f} -> {d['accuracy']:.6f}; gain {d['gain']:+.6f}",flush=True)
    if not passed:
        _write_json(report / "summary.json",dict(status="no_candidate",stage="development",development=d,
                    completed_repeats=0,kaggle_score=None,reason="Fixed pretrained-model blend failed development gate"))
        return
    repetitions=[]
    for repeat in [1,2]:
        ref,alt,p = [np.full(len(old.y),np.nan) for _ in range(3)]
        folds=np.zeros(len(old.y),int)
        for fold in range(1,6):
            base=np.load(PROJECT_ROOT / f"outputs/fixed_weight_confirmation_20261029/repeat_{repeat}_fold_{fold}.npz")
            ti,vi=base["training"],base["validation"]
            p[vi]=cached(ti,vi,f"repeat_{repeat}_fold_{fold}")
            ref[vi]=base["probability"] @ np.array(plan["weights"])
            alt[vi]=(1-WEIGHT)*ref[vi]+WEIGHT*p[vi]
            folds[vi]=fold
        r=dict(repeat=repeat,**compare(old.y,ref,alt,folds))
        repetitions.append(r)
        _write_json(report / "repeat_results.json",repetitions)
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=folds,
                                    reference=ref,alternative=alt,tabicl=p)),report / f"repeat_{repeat}_predictions.csv")
        if r["gain"] < 0 or r["log_loss"] > r["reference_log_loss"]+.002 or (repeat==1 and r["improved_folds"]==0):
            _write_json(report / "summary.json",dict(status="no_candidate",stage="sensitivity",development=d,repeats=repetitions,
                        completed_repeats=repeat,kaggle_score=None,reason="Fixed pretrained-model repeated non-regression/futility gate failed"))
            return
    passed=np.mean([r["gain"] for r in repetitions]) >= .001 and sum(r["improved_folds"] for r in repetitions) >= 6
    if not passed:
        _write_json(report / "summary.json",dict(status="no_candidate",stage="sensitivity",development=d,repeats=repetitions,
                    completed_repeats=2,kaggle_score=None,reason="Fixed pretrained-model repeated mean/fold-count gate failed"))
        return
    _write_json(report / "qualified.json",dict(development=d,repeats=repetitions))
    original=joblib.load(PROJECT_ROOT / "models/optimization_20261025/ensemble.joblib")
    canonical=pd.read_csv(old.report / "oof_predictions.csv",float_precision="round_trip")
    oof=(1-WEIGHT)*canonical.probability.to_numpy()
    members=[dict(m,weight=(1-WEIGHT)*m["weight"]) for m in original["members"]]
    identity=IdentityTransform()
    for fold in range(1,6):
        ti,vi=np.flatnonzero(old.fold_of!=fold),np.flatnonzero(old.fold_of==fold)
        oof[vi]+=WEIGHT*cached(ti,vi,f"final_fold_{fold}")
        member=joblib.load(modeldir / f"final_fold_{fold}.joblib")
        members.append(dict(weight=WEIGHT/5,builder=identity,encoder=identity,model=member))
    modelpath=modeldir / "ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5),data_hashes=original["data_hashes"]),modelpath,compress=3)
    probability,submission=predict_ensemble(modelpath)
    again,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(again,probability,atol=2e-6,rtol=0)
    pd.testing.assert_frame_equal(check,submission)
    _write_csv(submission,output / "submission_tabicl4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=probability,prediction=probability>=.5)),output / "test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=old.fold_of,probability=oof)),report / "canonical_oof.csv")
    _write_json(report / "summary.json",dict(status="candidate_generated",stage="completed",development=d,repeats=repetitions,
                completed_repeats=2,kaggle_score=None,models=50,tabicl_contexts=5,internal_tabicl_estimators_per_context=8,
                canonical_accuracy=float(accuracy_score(old.y,oof>=.5)),reload_max_error=float(np.max(np.abs(again-probability)))))


if __name__ == "__main__":
    main()
