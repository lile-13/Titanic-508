"""Independently reload fitted models and reconstruct the frozen view study."""
import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src import preprocess as pp
from src.competition_features import FeatureBuilder
from src.optimization import SearchRunner, config_id, predict_ensemble
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.train import _write_json, _write_csv

NAME = "unlabeled_views_20261030"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(y, ref, alt, folds):
    before, after = (ref >= .5) == y, (alt >= .5) == y
    gains = [float(after[folds == f].mean() - before[folds == f].mean()) for f in np.unique(folds)]
    return dict(reference_accuracy=float(before.mean()), accuracy=float(after.mean()),
                gain=float(after.mean() - before.mean()), reference_log_loss=float(log_loss(y, ref)),
                log_loss=float(log_loss(y, alt)), corrected=int((~before & after).sum()),
                new_errors=int((before & ~after).sum()), fold_gains=gains, improved_folds=sum(v > 0 for v in gains))


def assert_stats(actual, saved):
    for key, value in actual.items():
        np.testing.assert_allclose(value, saved[key], atol=1e-13, rtol=0, err_msg=key)


def main():
    old = SearchRunner()
    report, out = EXPERIMENTS_DIR / NAME, PROJECT_ROOT / "outputs" / NAME
    modeldir = PROJECT_ROOT / "models" / NAME
    protocol, summary = read(report / "protocol.json"), read(report / "summary.json")
    plan = read(old.report / "frozen_plan.json")
    for path, digest in protocol["sources"].items():
        assert sha(PROJECT_ROOT / path) == digest, path
    for path, digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir() / path) == digest, path
    assert sha(old.report / "frozen_plan.json") == protocol["source_plan_sha256"]
    assert sha(EXPERIMENTS_DIR / "fixed_weight_confirmation_20261029/protocol.json") == protocol["baseline_protocol_sha256"]
    ledger = EXPERIMENTS_DIR / "kaggle_submissions.json"
    ledger_hash = sha(ledger)
    expected = "270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    assert sha(PROJECT_ROOT / "outputs/kaggle_ready_4690/submission_ensemble4690.csv") == expected
    assert sha(old.output / "submission.csv") == expected
    predictions = {}
    rows_checked = 0
    for path in sorted(report.glob("*_seed_*_fold_*.json")):
        record = read(path)
        stem = path.stem
        data = np.load(out / f"{stem}.npz")
        ti, vi = data["training"], data["validation"]
        tag = record["tag"]
        if tag.startswith("development_fold_"):
            fold = int(tag.rsplit("_", 1)[-1])
            expected_ti, expected_vi = old.splits[fold - 1]
            assert not (old.fold_of[ti] == 5).any() and not (old.fold_of[vi] == 5).any()
        elif tag.startswith("repeat_"):
            cache = np.load(PROJECT_ROOT / f"outputs/fixed_weight_confirmation_20261029/{tag}.npz")
            expected_ti, expected_vi = cache["training"], cache["validation"]
        else:
            assert tag.startswith("final_fold_")
            fold = int(tag.rsplit("_", 1)[-1])
            expected_ti, expected_vi = np.flatnonzero(old.fold_of != fold), np.flatnonzero(old.fold_of == fold)
        np.testing.assert_array_equal(ti, expected_ti)
        np.testing.assert_array_equal(vi, expected_vi)
        assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
        modelpath = modeldir / f"{stem}.joblib"
        assert sha(modelpath) == record["model_sha256"]
        member = joblib.load(modelpath)
        assert member["builder"].view == record["view"]
        assert member["model"].get_params()["random_seed"] == record["seed"]
        for key, value in protocol["config"]["params"].items():
            assert member["model"].get_params()[key] == value, key
        X = member["builder"].transform(old.features.iloc[vi])
        control = FeatureBuilder("rich_names", "native").fit(old.features.iloc[ti]).transform(old.features.iloc[vi])
        pd.testing.assert_frame_equal(X[list(control)], control)
        assert pp.TARGET not in X and list(X) == record["columns"]
        assert member["encoder"].cat_columns_ == record["categorical"]
        q = member["model"].predict_proba(member["encoder"].transform(X))[:, 1]
        np.testing.assert_allclose(q, data["probability"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(((q >= .5) == old.y[vi]).mean(), record["accuracy"], atol=1e-14, rtol=0)
        predictions[(record["view"], record["seed"], tag)] = q
        rows_checked += len(vi)
    dev = old.fold_of != 5
    reference = np.full(len(old.y), np.nan)
    reference[dev] = 0.
    control_parts = []
    for j, (weight, component) in enumerate(zip(plan["weights"], plan["members"])):
        parts = []
        for seed in component["seeds"]:
            trial = config_id(dict(component["config"], seed=seed), "confirmation", [1, 2, 3, 4])
            q = np.load(old.output / "search" / f"{trial}.npz")["probability"]
            assert np.isnan(q[~dev]).all()
            parts.append(q)
        avg = np.mean(parts, axis=0)
        reference[dev] += weight * avg[dev]
        if j == 1:
            control_parts = parts
    control = np.mean(control_parts, axis=0)

    def development(view, seed):
        p = np.full(len(old.y), np.nan)
        for fold, (_, vi) in enumerate(old.splits, 1):
            p[vi] = predictions[(view, seed, f"development_fold_{fold}")]
        return p

    screen = []
    for view in protocol["views"]:
        new = development(view, 42)
        alt = reference[dev] + .25 * (new[dev] - control[dev])
        screen.append(dict(view=view, **stats(old.y[dev], reference[dev], alt, old.fold_of[dev])))
    saved_screen = read(report / "single_seed_screen.json")
    for actual, saved in zip(screen, saved_screen):
        assert actual["view"] == saved["view"]
        assert_stats({k: v for k, v in actual.items() if k != "view"}, saved)
    selected = sorted(screen, key=lambda r: (-r["accuracy"], r["log_loss"], protocol["views"].index(r["view"])))[0]["view"]
    assert selected == summary["view"] == read(report / "selected_view.json")["view"]
    replacement = np.mean([development(selected, seed) for seed in protocol["seeds"]], axis=0)
    alternative = reference + .25 * (replacement - control)
    d = stats(old.y[dev], reference[dev], alternative[dev], old.fold_of[dev])
    assert_stats(d, summary["development"])
    assert_stats(d, read(report / "development.json"))
    passed = d["gain"] >= .001 and d["improved_folds"] >= 3 and d["log_loss"] <= d["reference_log_loss"] + .002
    assert passed == read(report / "development.json")["passed"]
    table = pd.read_csv(report / "development_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
    np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
    np.testing.assert_array_equal(table.Transported, old.y)
    np.testing.assert_array_equal(table.fold, old.fold_of)
    for column, value in [("reference", reference), ("alternative", alternative), ("replacement", replacement), ("matched_control", control)]:
        np.testing.assert_allclose(table[column], value, atol=1e-14, rtol=0, equal_nan=True)
        assert np.isnan(table[column].to_numpy()[~dev]).all()
    if not passed:
        assert summary["stage"] == "development" and summary["status"] == "no_candidate"
        assert summary["completed_repeats"] == 0
        assert len(predictions) == 16
    repeat_rows = []
    for repeat in range(1, summary["completed_repeats"] + 1):
        ref, alt = np.zeros(len(old.y)), np.zeros(len(old.y))
        folds, coverage = np.zeros(len(old.y), int), np.zeros(len(old.y), int)
        for fold in range(1, 6):
            tag = f"repeat_{repeat}_fold_{fold}"
            base = np.load(PROJECT_ROOT / f"outputs/fixed_weight_confirmation_20261029/{tag}.npz")
            vi = base["validation"]
            q = np.mean([predictions[(selected, seed, tag)] for seed in protocol["seeds"]], axis=0)
            ref[vi] = base["probability"] @ np.array(plan["weights"])
            alt[vi] = ref[vi] + .25 * (q - base["probability"][:, 1])
            folds[vi], coverage[vi] = fold, coverage[vi] + 1
        np.testing.assert_array_equal(coverage, np.ones(len(old.y), int))
        r = stats(old.y, ref, alt, folds)
        assert_stats(r, summary["repeats"][repeat - 1])
        table = pd.read_csv(report / f"repeat_{repeat}_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
        np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
        for column, values in [("reference", ref), ("alternative", alt), ("fold", folds)]:
            np.testing.assert_allclose(table[column], values, atol=1e-14, rtol=0)
        repeat_rows.append(r)
    qualified = (passed and len(repeat_rows) == 2 and all(r["gain"] >= 0 and r["log_loss"] <= r["reference_log_loss"] + .002 for r in repeat_rows)
                 and np.mean([r["gain"] for r in repeat_rows]) >= .001 and sum(r["improved_folds"] for r in repeat_rows) >= 6)
    assert qualified == (summary["status"] == "candidate_generated")
    if not qualified:
        assert not (modeldir / "ensemble.joblib").exists()
        assert not (out / "submission_unlabeled4690.csv").exists()
    else:
        probability, submission = predict_ensemble(modeldir / "ensemble.joblib")
        saved = pd.read_csv(out / "submission_unlabeled4690.csv", dtype={"PassengerId": str})
        pd.testing.assert_frame_equal(saved, submission)
        saved_prob = pd.read_csv(out / "test_predictions.csv", float_precision="round_trip")
        np.testing.assert_allclose(probability, saved_prob.probability, atol=1e-12, rtol=0)
        assert len(saved) == 4277 and saved.PassengerId.is_unique and saved.Transported.dtype == bool
        ensemble = joblib.load(modeldir / "ensemble.joblib")
        assert len(ensemble["members"]) == 45
        np.testing.assert_allclose(sum(m["weight"] for m in ensemble["members"]), 1., atol=1e-14, rtol=0)
    # Disjoint diagnostic cohorts; labels never determine the feature transformation.
    seen = np.zeros(len(old.y), bool)
    for ti, vi in old.splits:
        seen[vi] = old.features.FirstName.iloc[vi].isin(old.features.FirstName.iloc[ti].dropna()).to_numpy()
    rows = []
    for name, mask in [("全部开发", dev), ("G甲板", dev & old.features.Deck.eq("G").fillna(False).to_numpy(bool)),
                       ("姓名缺失", dev & old.features.FirstName.isna().to_numpy()),
                       ("已知且名字在该训练折出现", dev & seen),
                       ("已知且名字未在该训练折出现", dev & ~seen & old.features.FirstName.notna().to_numpy())]:
        before, after = (reference[mask] >= .5) == old.y[mask], (alternative[mask] >= .5) == old.y[mask]
        rows.append(dict(cohort=name, rows=int(mask.sum()), reference_accuracy=float(before.mean()), accuracy=float(after.mean()),
                         corrected=int((~before & after).sum()), new_errors=int((before & ~after).sum()), net_correct=int(after.sum() - before.sum())))
    _write_csv(pd.DataFrame(rows), report / "subgroup_results.csv")
    changed = dev & ((reference >= .5) != (alternative >= .5))
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId[changed], fold=old.fold_of[changed], Transported=old.y[changed],
                                reference=reference[changed], alternative=alternative[changed])), report / "changed_predictions.csv")
    testlog = PROJECT_ROOT / "outputs/unlabeled_views_20261030_tests.log"
    log = testlog.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Ran (\d+) tests", log)
    assert match and re.search(r"\nOK\s*$", log)
    assert sha(ledger) == ledger_hash
    audit = dict(passed=True, source_and_data_hashes_match=True, models_reloaded=len(predictions),
                 validation_rows_repredicted=rows_checked, tree_parameters_match=True, base_features_identical=True,
                 group_isolation_verified=True, selection_and_metrics_reconstructed=True,
                 excluded_fold_predictions_missing=True, gate_passed=bool(qualified), completed_repeats=len(repeat_rows),
                 tests_passed=int(match.group(1)), test_log_sha256=sha(testlog), incumbent_csv_sha256=expected,
                 platform_ledger_unchanged=True, limitation="Adaptive historical development; no independent holdout or new Kaggle score")
    _write_json(PROJECT_ROOT / "reports/checks/unlabeled_views_20261030.json", audit)
    print(json.dumps(dict(audit=audit, screen=screen, development=d, subgroups=rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
