"""Reconstruct the frozen cohort routing from caches; never retrain tree models."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src import preprocess as pp
from src.g_deck_models import g_mask
from src.g_offset_models import GOffsetCalibrator
from src.g_routing import cohort_masks, route_probability
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.train import _write_csv, _write_json

NAME = "g_cohort_routing_20261029"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, ref, alt, frame, folds):
    cold, child = cohort_masks(frame)
    g = g_mask(frame)
    before, after = (ref >= .5) == y, (alt >= .5) == y
    np.testing.assert_array_equal(ref[~(cold | child)], alt[~(cold | child)])
    gains = [float(after[g & (folds == f)].mean() - before[g & (folds == f)].mean())
             for f in np.unique(folds)]
    return dict(reference_accuracy=float(before.mean()), accuracy=float(after.mean()),
                gain=float(after.mean() - before.mean()), net_correct=int(after.sum() - before.sum()),
                corrected=int((~before & after).sum()), new_errors=int((before & ~after).sum()),
                reference_log_loss=float(log_loss(y, ref)), log_loss=float(log_loss(y, alt)),
                g_rows=int(g.sum()), g_reference_accuracy=float(before[g].mean()), g_accuracy=float(after[g].mean()),
                g_reference_log_loss=float(log_loss(y[g], ref[g])), g_log_loss=float(log_loss(y[g], alt[g])),
                cold_rows=int(cold.sum()), cold_net_correct=int(after[cold].sum() - before[cold].sum()),
                child_rows=int(child.sum()), child_net_correct=int(after[child].sum() - before[child].sum()),
                g_fold_gains=gains, g_improved_folds=sum(v > 0 for v in gains),
                untouched_predictions_identical=True)


def compare(actual, saved):
    assert actual.keys() == saved.keys()
    for key in actual:
        if isinstance(actual[key], dict):
            compare(actual[key], saved[key])
        else:
            np.testing.assert_allclose(actual[key], saved[key], atol=1e-13, rtol=0, err_msg=key)


def main():
    old = SearchRunner()
    report = EXPERIMENTS_DIR / NAME
    out = PROJECT_ROOT / "outputs" / NAME
    protocol = json.loads((report / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    plan = json.loads((old.report / "frozen_plan.json").read_text(encoding="utf-8"))
    for path, digest in protocol["sources"].items():
        assert sha(PROJECT_ROOT / path) == digest, path
    for name, digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir() / name) == digest, name
    for name, digest in protocol["source_choices"].items():
        assert sha(EXPERIMENTS_DIR / name / "frozen_choice.json") == digest
    assert sha(old.report / "frozen_plan.json") == protocol["source_plan_hash"]
    baseline_report = EXPERIMENTS_DIR / "fixed_weight_confirmation_20261029"
    assert sha(baseline_report / "protocol.json") == protocol["baseline_protocol_sha256"]
    public = pd.concat([old.features.drop(columns=pp.TARGET), old.test_features], ignore_index=True)
    assert pp.TARGET not in public
    ledger = EXPERIMENTS_DIR / "kaggle_submissions.json"
    ledger_hash = sha(ledger)
    incumbent_hash = "270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    for path in [old.output / "submission.csv", PROJECT_ROOT / "outputs/kaggle_ready_4690/submission_ensemble4690.csv"]:
        assert sha(path) == incumbent_hash
    reconstructed, subgroup_rows, changes = [], [], []
    teacher_count = 0
    for repeat in range(1, summary["completed_repeats"] + 1):
        full = pd.read_csv(report / f"repeat_{repeat}_oof.csv", dtype={"PassengerId": str}, float_precision="round_trip")
        np.testing.assert_array_equal(full.PassengerId, old.raw.PassengerId)
        np.testing.assert_array_equal(full.Transported, old.y)
        coverage = np.zeros(len(old.y), int)
        seed = [20261029, 20261106][repeat - 1]
        expected_splits = list(GroupKFold(5, shuffle=True, random_state=seed).split(old.features, old.y, old.features.GroupId))
        for fold, (expected_ti, expected_vi) in enumerate(expected_splits, 1):
            tag = f"repeat_{repeat}_fold_{fold}"
            base = np.load(PROJECT_ROOT / f"outputs/fixed_weight_confirmation_20261029/{tag}.npz")
            ti, vi = base["training"], base["validation"]
            np.testing.assert_array_equal(ti, expected_ti)
            np.testing.assert_array_equal(vi, expected_vi)
            assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
            coverage[vi] += 1
            np.testing.assert_array_equal(full.fold.iloc[vi], np.full(len(vi), fold))
            ref = base["probability"] @ np.array(plan["weights"])
            np.testing.assert_allclose(full.reference.iloc[vi], ref, atol=1e-14, rtol=0)
            inner = np.load(out / f"{tag}_inner.npz")
            fit, cal = inner["fit_indices"], inner["cal_indices"]
            a, b = next(GroupShuffleSplit(1, test_size=.25, random_state=690001 + repeat*10 + fold)
                        .split(ti, groups=old.features.GroupId.iloc[ti]))
            np.testing.assert_array_equal(fit, ti[a])
            np.testing.assert_array_equal(cal, ti[b])
            assert not set(fit) & set(cal) and not set(np.r_[fit, cal]) & set(vi)
            np.testing.assert_array_equal(np.sort(np.r_[fit, cal]), ti)
            assert not set(old.features.GroupId.iloc[fit]) & set(old.features.GroupId.iloc[cal])
            teacher = np.zeros(len(cal))
            for j, (weight, component) in enumerate(zip(plan["weights"], plan["members"])):
                for model_seed in component["seeds"]:
                    cache = np.load(out / f"{tag}_teacher_{j}_{model_seed}.npz")
                    np.testing.assert_array_equal(cache["fit_indices"], fit)
                    np.testing.assert_array_equal(cache["cal_indices"], cal)
                    assert np.isfinite(cache["probability"]).all()
                    teacher += weight / len(component["seeds"]) * cache["probability"]
                    teacher_count += 1
            np.testing.assert_allclose(teacher, inner["probability"], atol=1e-14, rtol=0)
            part = pd.read_csv(report / f"{tag}_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
            np.testing.assert_array_equal(part.PassengerId, old.raw.PassengerId.iloc[vi])
            np.testing.assert_allclose(part.reference, ref, atol=1e-14, rtol=0)
            for label, child_key, offset_key in [("roles", "child_profile", "cold_offset"),
                                                 ("control", "control_child_profile", "control_cold_offset")]:
                specialist = []
                for model_seed in protocol["seeds"]:
                    cache = np.load(out / f"{tag}_{label}_{model_seed}.npz")
                    np.testing.assert_array_equal(cache["training"], ti)
                    np.testing.assert_array_equal(cache["validation"], vi)
                    q = cache["probability"]
                    assert np.isfinite(q[g_mask(old.features.iloc[vi])]).all()
                    assert np.isnan(q[~g_mask(old.features.iloc[vi])]).all()
                    specialist.append(q)
                assert protocol[child_key]["roles"] == protocol[offset_key]["roles"]
                calibrator = GOffsetCalibrator(**protocol[offset_key]).fit(old.features.iloc[cal], teacher, old.y[cal], public)
                saved = json.loads((report / f"{tag}_{label}_calibrator.json").read_text(encoding="utf-8"))
                np.testing.assert_allclose(calibrator.coef_, saved["coefficients"], atol=1e-13, rtol=0)
                assert saved["converged"] and saved["g_calibration_rows"] == int(g_mask(old.features.iloc[cal]).sum())
                routed = route_probability(old.features.iloc[vi], ref, calibrator.predict(old.features.iloc[vi], ref),
                                           np.mean(specialist, axis=0), protocol["cold_weight"], protocol["child_weight"])
                column = "alternative" if label == "roles" else "control"
                np.testing.assert_allclose(routed, full[column].iloc[vi], atol=1e-14, rtol=0)
                np.testing.assert_allclose(routed, part[column], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(coverage, np.ones(len(old.y), int))
        ref, alt, ctl, folds = [full[c].to_numpy() for c in ["reference", "alternative", "control", "fold"]]
        row = dict(repeat=repeat, **metrics(old.y, ref, alt, old.features, folds),
                   control=metrics(old.y, ref, ctl, old.features, folds))
        row["added_roles_g_gain"] = row["g_accuracy"] - row["control"]["g_accuracy"]
        compare(row, summary["repeats"][repeat - 1])
        reconstructed.append(row)
        cold, child = cohort_masks(old.features)
        for name, mask in [("全部", np.ones(len(old.y), bool)), ("G冷冻", cold), ("G未冷冻儿童", child),
                           ("其他G", g_mask(old.features) & ~(cold | child)), ("非G", ~g_mask(old.features))]:
            before, after = (ref[mask] >= .5) == old.y[mask], (alt[mask] >= .5) == old.y[mask]
            subgroup_rows.append(dict(repeat=repeat, cohort=name, rows=int(mask.sum()), reference_accuracy=float(before.mean()),
                                      accuracy=float(after.mean()), corrected=int((~before & after).sum()),
                                      new_errors=int((before & ~after).sum()), net_correct=int(after.sum()-before.sum())))
        for i in np.flatnonzero((ref >= .5) != (alt >= .5)):
            changes.append(dict(repeat=repeat, PassengerId=old.raw.PassengerId.iloc[i], fold=int(folds[i]),
                                Transported=int(old.y[i]), reference=float(ref[i]), alternative=float(alt[i]),
                                corrected=bool((alt[i] >= .5) == old.y[i]), cold=bool(cold[i]), child=bool(child[i])))
    saved_rows = json.loads((report / "repeat_results.json").read_text(encoding="utf-8"))
    assert len(reconstructed) == len(saved_rows)
    for actual, saved in zip(reconstructed, saved_rows):
        compare(actual, saved)
    def hard_failure(row):
        return (min(row["net_correct"], row["cold_net_correct"], row["child_net_correct"]) < 0 or
                row["log_loss"] > row["reference_log_loss"] + .002 or
                row["g_log_loss"] > row["g_reference_log_loss"] + .005 or
                (row["repeat"] == 1 and row["g_improved_folds"] == 0))
    passed = (len(reconstructed) == 2 and not any(map(hard_failure, reconstructed)) and
              np.mean([r["gain"] for r in reconstructed]) >= .001 and
              sum(r["g_improved_folds"] for r in reconstructed) >= 6 and
              np.mean([r["added_roles_g_gain"] for r in reconstructed]) >= .003)
    assert (summary["status"] == "candidate_generated") == passed
    if summary.get("early_stop", False):
        assert len(reconstructed) == 1 and hard_failure(reconstructed[0])
        assert not (report / "repeat_2_oof.csv").exists()
    if not passed:
        assert not (out / "submission_grouted4690.csv").exists()
        assert not (PROJECT_ROOT / "models" / NAME / "ensemble.joblib").exists()
    assert sha(ledger) == ledger_hash
    _write_csv(pd.DataFrame(subgroup_rows), report / "subgroup_results.csv")
    _write_csv(pd.DataFrame(changes), report / "changed_predictions.csv")
    audit = dict(passed=True, source_and_data_hashes_match=True, nested_group_isolation_verified=True,
                 teacher_members_checked=teacher_count, specialist_caches_checked=30*len(reconstructed),
                 routing_and_metrics_reconstructed=True, untouched_probabilities_identical=True,
                 gate_passed=bool(passed), completed_repeats=len(reconstructed), early_stop=summary.get("early_stop", False),
                 incumbent_submission_sha256=incumbent_hash, platform_ledger_unchanged=True,
                 limitation="Historical rows and partitions; adaptive sensitivity checks, not independent blind validation")
    _write_json(PROJECT_ROOT / "reports/checks/g_cohort_routing_20261029.json", audit)
    print(json.dumps(dict(audit=audit, results=reconstructed, subgroups=subgroup_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
