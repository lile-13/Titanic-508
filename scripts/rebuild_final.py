"""Rebuild the fixed 25% tree / 75% TabICL model without search caches."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src import preprocess as pp
from src.competition_features import build_pool, FeatureBuilder, InputEncoder
from src.optimization import estimator, predict_ensemble
from src.paths import PROJECT_ROOT as P
from src.probability_calibration import IdentityTransform
from src.tabicl_aggregation import fit_probability_mean
from src.tabicl_models import CHECKPOINT, CHECKPOINT_SHA256, PARAMS
from src.train import _data_hashes, _write_csv, _write_json, make_submission, validate_fold_map


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run_name, threads=4, data_dir=None, dry_run=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("Use a new alphanumeric run-name with optional underscores/hyphens")
    if threads < 1:
        raise ValueError("threads must be positive")
    directory = pp.find_data_dir(data_dir)
    source = P / "reports/experiments/optimization_20261025"
    plan = json.loads((source / "frozen_plan.json").read_text(encoding="utf-8"))
    historical = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if _data_hashes(directory) != historical["data_hashes"]:
        raise ValueError("Raw files differ from the fixed historical training data")
    checkpoint = P / "models/tabicl_pretrained" / CHECKPOINT
    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Author TabICL checkpoint hash mismatch")
    raw, test = pp.load_raw(directory)
    features, test_features = build_pool(raw, test)
    folds_path = P / "features/folds.csv"
    folds = validate_fold_map(features, pd.read_csv(folds_path, dtype={"PassengerId": str, "GroupId": str}))
    y = raw.Transported.astype(int).to_numpy()
    public = pd.concat([features.drop(columns="Transported"), test_features], ignore_index=True)
    report, output, models = [P / folder / run_name for folder in ["reports/experiments", "outputs", "models"]]
    if any(p.exists() for p in [report, output, models]):
        raise FileExistsError("Run directory exists; choose a new --run-name")
    protocol = dict(name=run_name, purpose="Reconstruct a fixed historical algorithm; no model selection or new platform score",
        tree_weight=.25, tabicl_weight=.75, threshold=.5, tabicl_params=dict(PARAMS, average_logits=False),
        tree_plan=plan, source_plan_sha256=sha(source / "frozen_plan.json"),
        data_hashes=_data_hashes(directory), folds_sha256=sha(folds_path), checkpoint_sha256=CHECKPOINT_SHA256,
        sources={s: sha(P / s) for s in ["scripts/rebuild_final.py", "src/competition_features.py", "src/preprocess.py",
            "src/optimization.py", "src/focused_features.py", "src/tabicl_models.py", "src/tabicl_aggregation.py"]},
        expected_tree_models=45, expected_tabicl_contexts=5,
        interpretation="OOF metrics reuse the already selected historical folds; descriptive, not independent blind validation")
    if dry_run:
        print(json.dumps(dict(checked=True, rows=len(raw), test_rows=len(test), folds=sorted(np.unique(folds).tolist()),
            tree_models=45, tabicl_contexts=5, weights=[.25, .75], outputs_created=False), indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed TabICL runtime requires CUDA")
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    for p in [report, output, models]:
        p.mkdir(parents=True, exist_ok=False)
    _write_json(report / "protocol.json", protocol)
    members, oof = [], np.full(len(raw), np.nan)
    # Keep the original tree fold order and weighted member layout.
    for fold in [5, 1, 2, 3, 4]:
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(features.GroupId.iloc[ti]) & set(features.GroupId.iloc[vi])
        pred = np.zeros(len(vi))
        for weight, spec in zip(plan["weights"], plan["members"]):
            cfg = spec["config"]
            builder = FeatureBuilder(cfg["feature_set"], cfg["strategy"]).fit(features.iloc[ti])
            X, V = builder.transform(features.iloc[ti]), builder.transform(features.iloc[vi])
            encoder = InputEncoder(cfg["family"]).fit(X)
            X, V = encoder.transform(X), encoder.transform(V)
            for seed in spec["seeds"]:
                model = estimator(dict(cfg, seed=seed), encoder.cat_columns_, threads)
                model.fit(X, y[ti])
                pred += weight / len(spec["seeds"]) * model.predict_proba(V)[:, 1]
                members.append(dict(weight=.25 * weight / len(spec["seeds"]) / 5,
                    builder=builder, encoder=encoder, model=model))
        oof[vi] = .25 * pred
        print(f"Tree fold {fold}: nine fixed models fitted", flush=True)
    identity = IdentityTransform()
    for fold in range(1, 6):
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        context = fit_probability_mean(features.iloc[ti], y[ti], public)
        oof[vi] += .75 * context.predict_proba(features.iloc[vi])[:, 1]
        assert context._model.get_params()["average_logits"] is False
        context._model = None
        members.append(dict(weight=.15, builder=identity, encoder=identity, model=context))
        gc.collect()
        torch.cuda.empty_cache()
        print(f"TabICL fold {fold}: fixed eight-view context fitted", flush=True)
    bundle_path = models / "ensemble.joblib"
    assert len(members) == 50 and abs(sum(m["weight"] for m in members) - 1) < 1e-12
    joblib.dump(dict(members=members, plan=dict(name=run_name, threshold=.5), data_hashes=protocol["data_hashes"]), bundle_path, compress=3)
    del members
    gc.collect()
    probability, submission = predict_ensemble(bundle_path, directory)
    _write_csv(submission, output / "submission.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=test.PassengerId, probability=probability, prediction=submission.Transported)), output / "test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=raw.PassengerId, fold=folds, Transported=y, probability=oof)), report / "oof_predictions.csv")
    _write_json(report / "summary.json", dict(status="fixed_model_rebuilt", tree_models=45, tabicl_contexts=5,
        oof_accuracy=float(accuracy_score(y, oof >= .5)), oof_log_loss=float(log_loss(y, oof)),
        model_sha256=sha(bundle_path), submission_sha256=sha(output / "submission.csv"), rows=len(submission),
        serialized_model_prediction_completed=True, kaggle_score=None,
        note="Historical .81318 belongs to the recorded submission; this rebuild has not been submitted"))
    print(f"Rebuilt fixed model: {bundle_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate fixed inputs and show the plan without fitting")
    args = parser.parse_args()
    main(args.run_name, args.threads, args.data_dir, args.dry_run)
