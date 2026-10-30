"""Bounded paired tests: replace only the frozen rich-name CatBoost component."""
import hashlib
import json
import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src import preprocess as pp
from src.competition_features import InputEncoder
from src.optimization import SearchRunner, config_id, estimator, predict_ensemble
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.train import _write_csv, _write_json, make_submission
from src.unlabeled_views import UnlabeledViewBuilder

NAME = "unlabeled_views_20261030"
VIEWS = ["names", "coordinates"]
SEEDS = [42, 2026, 103]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(member, frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:, 1]


def comparison(y, reference, alternative, folds):
    before, after = reference >= .5, alternative >= .5
    gains = [float(accuracy_score(y[folds == f], after[folds == f]) - accuracy_score(y[folds == f], before[folds == f]))
             for f in np.unique(folds)]
    return dict(reference_accuracy=float(accuracy_score(y, before)), accuracy=float(accuracy_score(y, after)),
                gain=float(accuracy_score(y, after) - accuracy_score(y, before)),
                reference_log_loss=float(log_loss(y, reference)), log_loss=float(log_loss(y, alternative)),
                corrected=int(((before != y) & (after == y)).sum()),
                new_errors=int(((before == y) & (after != y)).sum()),
                fold_gains=gains, improved_folds=sum(v > 0 for v in gains))


def main():
    old = SearchRunner()
    report = EXPERIMENTS_DIR / NAME
    out = PROJECT_ROOT / "outputs" / NAME
    modeldir = PROJECT_ROOT / "models" / NAME
    for path in [report, out, modeldir]:
        path.mkdir(parents=True, exist_ok=True)
    plan = json.loads((old.report / "frozen_plan.json").read_text(encoding="utf-8"))
    assert plan["weights"] == [.25, .25, .5]
    cfg = plan["members"][1]["config"]
    assert cfg["family"] == "catboost" and cfg["feature_set"] == "rich_names" and cfg["strategy"] == "native"
    protocol = dict(name=NAME, views=VIEWS, config=cfg, seeds=SEEDS, replacement_component=1,
                    fixed_component_weight=.25, threshold=.5,
                    hypothesis="Unseen name substrings may share statistical information; relative coordinate gaps may simplify axis-aligned splits. These are representations of existing features, not new independent data or causal facts.",
                    selection="Two seed-42 views on development folds 1..4; select one by highest ensemble accuracy, then lowest logloss, then listed view order. Only that view gets seeds 2026 and 103. No weight, threshold, parameter or branch search; no fallback to the other view after failure.",
                    gate="Three-seed development replacement gain >=.001, >=3/4 improved folds, logloss deterioration <=.002. If passed: two existing five-fold repartitions, both nonnegative, mean gain >=.001, >=6/10 improved folds, logloss deterioration <=.002 each. Stop before full refit if any stage fails.",
                    interpretation="Historical adaptive development and paired sensitivity only; neither rows nor partitions form a new blind holdout.",
                    context="Public train+test feature allowlist only; consistent counts and coordinates; no Transported labels in public context; queries independent of transform batch.",
                    source_plan_sha256=sha(old.report / "frozen_plan.json"),
                    baseline_protocol_sha256=sha(EXPERIMENTS_DIR / "fixed_weight_confirmation_20261029/protocol.json"),
                    data_hashes=json.loads((old.report / "protocol.json").read_text(encoding="utf-8"))["data_hashes"],
                    sources={p: sha(PROJECT_ROOT / p) for p in ["scripts/unlabeled_view_study.py", "src/unlabeled_views.py",
                        "src/competition_features.py", "src/preprocess.py", "src/optimization.py", "features/folds.csv", "requirements.txt"]},
                    references=["https://www.kaggle.com/c/spaceship-titanic/data",
                                "https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic"],
                    rejected_leads="Public combined data have zero cross-group CabinKey sharing and zero train/test CabinKey overlap; no cabin-mate label propagation. No inferred gender labels or external benchmark answers.")
    protocol_path = report / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Frozen source/data/protocol changed; use a new run name")
    else:
        _write_json(protocol_path, protocol)
    for name, digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir() / name) == digest
    if (report / "summary.json").exists():
        print("Completed frozen study; use audit_unlabeled_views to reconstruct results", flush=True)
        return
    public = pd.concat([old.features.drop(columns=pp.TARGET), old.test_features], ignore_index=True)
    assert pp.TARGET not in public
    diagnostic = {}
    for key in ["FirstName", "Surname", "CabinKey"]:
        counts = public.groupby(key, observed=True).GroupId.nunique()
        diagnostic[key] = dict(public_unique=int(public[key].nunique()), cross_group_keys=int((counts > 1).sum()),
                               test_seen_train_share=float(old.test_features[key].isin(old.features[key].dropna()).mean()))
    _write_json(report / "target_free_coverage.json", diagnostic)
    dev = old.fold_of != 5
    control_components = np.full((len(old.y), 3), np.nan)
    reference = np.zeros(len(old.y))
    for j, (weight, component) in enumerate(zip(plan["weights"], plan["members"])):
        parts = []
        for seed in component["seeds"]:
            trial = config_id(dict(component["config"], seed=seed), "confirmation", [1, 2, 3, 4])
            p = np.load(old.output / "search" / f"{trial}.npz")["probability"]
            assert np.isnan(p[~dev]).all()
            if j == 1:
                control_components[:, SEEDS.index(seed)] = p
            parts.append(p)
        reference[dev] += weight * np.mean(parts, axis=0)[dev]
    reference[~dev] = np.nan
    np.testing.assert_allclose(accuracy_score(old.y[dev], reference[dev] >= .5), plan["development_accuracy"], atol=1e-14, rtol=0)

    def fit_cached(view, seed, ti, vi, tag):
        stem = f"{view}_seed_{seed}_{tag}"
        cache, modelpath, recordpath = out / f"{stem}.npz", modeldir / f"{stem}.joblib", report / f"{stem}.json"
        if cache.exists() and modelpath.exists() and recordpath.exists():
            saved = np.load(cache)
            np.testing.assert_array_equal(saved["training"], ti)
            np.testing.assert_array_equal(saved["validation"], vi)
            assert sha(modelpath) == json.loads(recordpath.read_text(encoding="utf-8"))["model_sha256"]
            return saved["probability"]
        assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
        builder = UnlabeledViewBuilder(view).fit(old.features.iloc[ti], public)
        X = builder.transform(old.features.iloc[ti])
        encoder = InputEncoder("catboost").fit(X)
        model = estimator(dict(cfg, seed=seed), encoder.cat_columns_, 4)
        model.fit(encoder.transform(X), old.y[ti])
        member = dict(builder=builder, encoder=encoder, model=model)
        p = predict(member, old.features.iloc[vi])
        joblib.dump(member, modelpath, compress=3)
        np.savez_compressed(cache, training=ti, validation=vi, probability=p)
        _write_json(recordpath, dict(view=view, seed=seed, tag=tag, model_sha256=sha(modelpath),
                                    columns=list(X), categorical=encoder.cat_columns_, rows=len(ti), validation_rows=len(vi),
                                    accuracy=float(accuracy_score(old.y[vi], p >= .5))))
        print(f"{stem} ready", flush=True)
        return p

    def evaluate(view, seed):
        p = np.full(len(old.y), np.nan)
        for fold, (ti, vi) in enumerate(old.splits, 1):
            assert not (old.fold_of[ti] == 5).any() and not (old.fold_of[vi] == 5).any()
            p[vi] = fit_cached(view, seed, ti, vi, f"development_fold_{fold}")
        return p

    def mixed(p):
        result = reference.copy()
        result[dev] += .25 * (p[dev] - control_components[dev].mean(axis=1))
        return result

    initial, first = [], {}
    for view in VIEWS:
        first[view] = evaluate(view, 42)
        alternative = mixed(first[view])
        row = dict(view=view, **comparison(old.y[dev], reference[dev], alternative[dev], old.fold_of[dev]))
        initial.append(row)
        print(f"Initial {view}: {row['accuracy']:.6f} gain {row['gain']:+.6f}", flush=True)
    _write_json(report / "single_seed_screen.json", initial)
    selected = sorted(initial, key=lambda r: (-r["accuracy"], r["log_loss"], VIEWS.index(r["view"])))[0]["view"]
    _write_json(report / "selected_view.json", dict(view=selected, no_retuning=True))
    p = np.mean([first[selected], evaluate(selected, 2026), evaluate(selected, 103)], axis=0)
    alternative = mixed(p)
    development = comparison(old.y[dev], reference[dev], alternative[dev], old.fold_of[dev])
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId, Transported=old.y, fold=old.fold_of,
                                reference=reference, alternative=alternative, replacement=p,
                                matched_control=control_components.mean(axis=1))), report / "development_predictions.csv")
    passed = development["gain"] >= .001 and development["improved_folds"] >= 3 and development["log_loss"] <= development["reference_log_loss"] + .002
    _write_json(report / "development.json", dict(view=selected, passed=bool(passed), **development))
    if not passed:
        _write_json(report / "summary.json", dict(status="no_candidate", stage="development", view=selected,
                    development=development, completed_repeats=0, kaggle_score=None,
                    reason="Frozen three-seed replacement development gate failed"))
        print("Development gate failed; retain incumbent without changing the view or weight", flush=True)
        return
    repetitions = []
    for repeat in [1, 2]:
        ref, alt = np.full(len(old.y), np.nan), np.full(len(old.y), np.nan)
        folds = np.zeros(len(old.y), int)
        for fold in range(1, 6):
            base = np.load(PROJECT_ROOT / f"outputs/fixed_weight_confirmation_20261029/repeat_{repeat}_fold_{fold}.npz")
            ti, vi = base["training"], base["validation"]
            q = np.mean([fit_cached(selected, seed, ti, vi, f"repeat_{repeat}_fold_{fold}") for seed in SEEDS], axis=0)
            ref[vi] = base["probability"] @ np.array(plan["weights"])
            alt[vi] = ref[vi] + .25 * (q - base["probability"][:, 1])
            folds[vi] = fold
        row = dict(repeat=repeat, **comparison(old.y, ref, alt, folds))
        repetitions.append(row)
        _write_json(report / "repeat_results.json", repetitions)
        _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId, Transported=old.y, fold=folds,
                                    reference=ref, alternative=alt)), report / f"repeat_{repeat}_predictions.csv")
        if row["gain"] < 0 or row["log_loss"] > row["reference_log_loss"] + .002 or (repeat == 1 and row["improved_folds"] == 0):
            _write_json(report / "summary.json", dict(status="no_candidate", stage="sensitivity", view=selected,
                        development=development, repeats=repetitions, completed_repeats=repeat, kaggle_score=None,
                        reason="Frozen repeated-partition non-regression or futility gate failed"))
            return
    passed = np.mean([r["gain"] for r in repetitions]) >= .001 and sum(r["improved_folds"] for r in repetitions) >= 6
    if not passed:
        _write_json(report / "summary.json", dict(status="no_candidate", stage="sensitivity", view=selected,
                    development=development, repeats=repetitions, completed_repeats=2, kaggle_score=None,
                    reason="Frozen repeated mean gain or fold-count gate failed"))
        return
    _write_json(report / "qualified.json", dict(view=selected, development=development, repeats=repetitions))
    original = joblib.load(PROJECT_ROOT / "models/optimization_20261025/ensemble.joblib")
    members = list(original["members"])
    oof = np.zeros(len(old.y))
    for block, fold in enumerate([5, 1, 2, 3, 4]):
        ti, vi = np.flatnonzero(old.fold_of != fold), np.flatnonzero(old.fold_of == fold)
        for offset, seed in enumerate(SEEDS, 3):
            tag = f"final_fold_{fold}"
            fit_cached(selected, seed, ti, vi, tag)
            path = modeldir / f"{selected}_seed_{seed}_{tag}.joblib"
            members[block*9 + offset] = dict(joblib.load(path), weight=.25/15)
        for member in members[block*9:block*9 + 9]:
            oof[vi] += member["weight"] * 5 * predict(member, old.features.iloc[vi])
    modelpath = modeldir / "ensemble.joblib"
    joblib.dump(dict(members=members, plan=dict(name=NAME, threshold=.5), data_hashes=original["data_hashes"]), modelpath, compress=3)
    probability, submission = predict_ensemble(modelpath)
    direct = sum(m["weight"] * predict(m, old.test_features) for m in members)
    np.testing.assert_allclose(probability, direct, atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(submission, make_submission(old.raw_test.PassengerId, direct, pp.load_submission_template(), .5))
    _write_csv(submission, out / "submission_unlabeled4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId, probability=probability, prediction=probability >= .5)), out / "test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId, Transported=old.y, fold=old.fold_of, probability=oof)), report / "canonical_oof.csv")
    _write_json(report / "summary.json", dict(status="candidate_generated", stage="completed", view=selected,
                development=development, repeats=repetitions, completed_repeats=2, kaggle_score=None,
                models=45, replaced_models=15, canonical_accuracy=float(accuracy_score(old.y, oof >= .5)),
                reload_max_error=float(np.max(np.abs(probability-direct)))))


if __name__ == "__main__":
    main()
