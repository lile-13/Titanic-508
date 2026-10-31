"""Reload TabM weights, recompute paired outcomes, and audit nested group isolation."""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from scripts.tabm_study import NAME, PARENT, WEIGHT, FRESH_SEEDS, passes
from scripts.tabicl_study import compare
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabm_models import TabMEncoder, PARAMS
from src.tabicl_models import CHECKPOINT, CHECKPOINT_SHA256
from src.focused_features import FocusedBuilder
from src.train import _write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    old = SearchRunner()
    report, output, models = EXPERIMENTS_DIR/NAME, PROJECT_ROOT/"outputs"/NAME, PROJECT_ROOT/"models"/NAME
    protocol, summary = read(report/"protocol.json"), read(report/"summary.json")
    for p, digest in protocol["source_hashes"].items():
        assert sha(PROJECT_ROOT/p) == digest
    for p, digest in protocol["installed_sources"].items():
        assert sha(Path(__import__(p).__file__)) == digest
    for p, digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p) == digest
    for p, version in protocol["environment"].items():
        assert importlib.metadata.version(p) == version
    parent = EXPERIMENTS_DIR/PARENT
    assert sha(parent/"protocol.json") == protocol["parent_protocol_sha256"]
    for p, digest in protocol["reference_files"].items():
        assert sha(parent/p) == digest
    assert sha(old.report/"frozen_plan.json") == read(parent/"protocol.json")["original_plan_sha256"]
    assert sha(PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT) == CHECKPOINT_SHA256
    plan = read(old.report/"frozen_plan.json")
    public = pd.concat([old.features.drop(columns=pp.TARGET), old.test_features], ignore_index=True)
    all_passed, reloaded, max_error = True, 0, 0.
    reference_trees, reference_contexts, fresh_deltas = 0, 0, []
    tf32_batch_error, ieee_batch_error, precision_label_changes = 0., 0., 0
    results = {}
    for stage, stored in summary["stages"].items():
        rows = []
        for result in stored:
            repeat = result["repeat"]
            table = pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv", dtype={"PassengerId":str}, float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported, old.y)
            if stage == "paired":
                baseline = pd.read_csv(parent/f"fresh_repeat_{repeat}_predictions.csv", float_precision="round_trip")
                np.testing.assert_allclose(table.reference, baseline.alternative, rtol=0, atol=1e-14)
                split = []
                for fold in range(1,6):
                    z = np.load(PROJECT_ROOT/f"outputs/{PARENT}/fresh_repeat_{repeat}_fold_{fold}.npz")
                    split.append((z["training"], z["validation"]))
            else:
                split = list(GroupKFold(5, shuffle=True, random_state=FRESH_SEEDS[repeat-1]).split(old.features, groups=old.features.GroupId))
            for fold, (ti,vi) in enumerate(split,1):
                tag = f"{stage}_repeat_{repeat}_fold_{fold}"
                if stage == "fresh":
                    reference = np.zeros(len(vi))
                    for j, (weight, component) in enumerate(zip(plan["weights"], plan["members"])):
                        for seed in component["seeds"]:
                            name = f"reference_{tag}_component_{j}_seed_{seed}"
                            record = read(report/f"{name}.json")
                            assert record["config"] == dict(component["config"], seed=seed)
                            assert sha(models/f"{name}.joblib") == record["model_sha256"]
                            ref = joblib.load(models/f"{name}.joblib")
                            cache = np.load(output/f"{name}.npz")
                            np.testing.assert_array_equal(cache["training"], ti)
                            np.testing.assert_array_equal(cache["validation"], vi)
                            q = ref["model"].predict_proba(ref["encoder"].transform(ref["builder"].transform(old.features.iloc[vi])))[:,1]
                            np.testing.assert_allclose(q, cache["probability"], rtol=0, atol=1e-14)
                            reference += weight/len(component["seeds"])*q
                            reference_trees += 1
                    name = f"reference_tfm_{tag}"
                    assert sha(models/f"{name}.joblib") == read(report/f"{name}.json")["model_sha256"]
                    ref = joblib.load(models/f"{name}.joblib")
                    cache = np.load(output/f"{name}.npz")
                    np.testing.assert_array_equal(cache["training"], ti)
                    np.testing.assert_array_equal(cache["validation"], vi)
                    np.testing.assert_array_equal(ref.training_y, old.y[ti])
                    expected = FocusedBuilder("compact", True).fit(old.features.iloc[ti], public).transform(old.features.iloc[ti])
                    pd.testing.assert_frame_equal(expected, ref.training_X)
                    q = ref.predict_proba(old.features.iloc[vi])[:,1]
                    np.testing.assert_allclose(q, cache["probability"], rtol=0, atol=2e-6)
                    assert ref._model.get_params()["average_logits"] is False
                    assert sum(len(v) for v in ref._model.ensemble_generator_.class_shuffles_.values()) == 8
                    np.testing.assert_allclose(.5*reference+.5*cache["probability"], table.reference.iloc[vi], rtol=0, atol=1e-14)
                    reference_contexts += 1
                    del ref
                    gc.collect()
                    torch.cuda.empty_cache()
                z = np.load(output/f"{tag}.npz")
                np.testing.assert_array_equal(z["training"], ti)
                np.testing.assert_array_equal(z["validation"], vi)
                assert (table.fold.iloc[vi] == fold).all()
                assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
                mp = models/f"{tag}.joblib"
                record = read(report/f"{tag}.json")
                assert sha(mp) == record["model_sha256"]
                member = joblib.load(mp)
                assert member._model is None and member.training_ids == old.raw.PassengerId.iloc[ti].tolist()
                it, iv = next(GroupShuffleSplit(1, test_size=.18, random_state=PARAMS["inner_seed"]).split(ti, old.y[ti], old.features.GroupId.iloc[ti]))
                assert member.selection["inner_train_ids"] == old.raw.PassengerId.iloc[ti[it]].tolist()
                assert member.selection["inner_validation_ids"] == old.raw.PassengerId.iloc[ti[iv]].tolist()
                assert not set(old.features.GroupId.iloc[ti[it]]) & set(old.features.GroupId.iloc[ti[iv]])
                best, epoch = float("inf"), 1
                for i, value in enumerate(member.selection["inner_losses"],1):
                    if value < best-1e-5:
                        best, epoch = value, i
                assert epoch == record["selected_epoch"] == member.selection["selected_epoch"]
                fitted = TabMEncoder().fit(member.builder.transform(old.features.iloc[ti]))
                assert fitted.categories_ == member.encoder.categories_ and fitted.active_ == member.encoder.active_
                pd.testing.assert_series_equal(fitted.medians_, member.encoder.medians_)
                pd.testing.assert_series_equal(fitted.scales_, member.encoder.scales_)
                p = member.predict_proba(old.features.iloc[vi])[:,1]
                assert member._model.k == 32
                np.testing.assert_allclose(p, z["probability"], atol=2e-6, rtol=0)
                np.testing.assert_allclose(p, table.tabm.iloc[vi], atol=2e-6, rtol=0)
                assert np.array_equal(p>=.5, table.tabm.iloc[vi].to_numpy()>=.5)
                batch_inputs = member.encoder.transform(member.builder.transform(old.features.iloc[vi]))
                single_inputs = member.encoder.transform(member.builder.transform(old.features.iloc[vi[:1]]))
                for batch, single in zip(batch_inputs, single_inputs):
                    np.testing.assert_array_equal(batch[:1], single)
                one = member.predict_proba(old.features.iloc[vi[:1]])[:,1]
                tf32_batch_error = max(tf32_batch_error, float(np.max(np.abs(one-p[:1]))))
                # TF32 kernels can round differently for batch sizes 1 and 512.
                # Isolate numerical rounding from any query-dependent preprocessing.
                torch.set_float32_matmul_precision("highest")
                try:
                    ieee_batch = member.predict_proba(old.features.iloc[vi])[:,1]
                    ieee_single = member.predict_proba(old.features.iloc[vi[:1]])[:,1]
                    np.testing.assert_allclose(ieee_single, ieee_batch[:1], atol=1e-6, rtol=0)
                    ieee_batch_error = max(ieee_batch_error, float(np.max(np.abs(ieee_single-ieee_batch[:1]))))
                    precision_label_changes += int(np.sum((ieee_batch>=.5)!=(p>=.5)))
                finally:
                    torch.set_float32_matmul_precision("high")
                max_error = max(max_error, float(np.max(np.abs(p-z["probability"]))))
                reloaded += 1
                del member
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Audited TabM {tag}", flush=True)
            np.testing.assert_allclose(table.alternative, (1-WEIGHT)*table.reference+WEIGHT*table.tabm, rtol=0, atol=1e-14)
            measured = dict(repeat=repeat, **compare(old.y, table.reference.to_numpy(), table.alternative.to_numpy(), table.fold.to_numpy()),
                            standalone_accuracy=float(((table.tabm.to_numpy()>=.5)==old.y).mean()))
            for key, value in measured.items():
                np.testing.assert_allclose(value, result[key], atol=1e-14, rtol=0)
            rows.append(measured)
            if stage == "fresh":
                fresh_deltas.append(((table.alternative.to_numpy()>=.5)==old.y).astype(float)-((table.reference.to_numpy()>=.5)==old.y))
        all_passed = all_passed and passes(rows)
        results[stage] = rows
    assert all_passed == (summary["status"] == "qualified_for_production")
    if not all_passed:
        assert not (models/"ensemble.joblib").exists()
        assert not list(output.glob("submission*.csv"))
    logpath = PROJECT_ROOT/"outputs/tabm_20261031_tests.log"
    log = logpath.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Ran (\d+) tests", log)
    assert match and re.search(r"\nOK\s*$", log)
    ledger = read(EXPERIMENTS_DIR/"kaggle_submissions.json")
    assert ledger["current_platform_best"]["model_run"] == PARENT and ledger["current_platform_best"]["score"] == .81178
    assert sha(PROJECT_ROOT/ledger["current_platform_best"]["submission"]) == "bd2858e0412f37d22ac38040401bfbe3e459a5ab5690f0e4b98924c78aed8ac0"
    uncertainty = None
    if len(fresh_deltas) == 2:
        delta = np.mean(fresh_deltas, axis=0)
        codes, groups = pd.factorize(old.features.GroupId)
        sums, sizes = np.bincount(codes, weights=delta), np.bincount(codes)
        rng, boot = np.random.default_rng(20481031), []
        for _ in range(20):
            draw = rng.integers(0, len(groups), size=(100, len(groups)))
            boot.extend((sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)).tolist())
        uncertainty = dict(mean_gain=float(delta.mean()), group_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
                           caveat="Fixed-prediction conditional group bootstrap, repeated outcomes averaged per passenger; excludes fitting/selection uncertainty, not a promotion gate")
    audit = dict(passed=True, models_reloaded=reloaded, max_prediction_reload_error=max_error,
                 training_only_preprocessing=True, outer_and_inner_group_isolation=True,
                 stopping_recomputed_from_inner_losses=True, all_32_members_retained=True,
                 reference_is_actual_latest_best=True, source_data_environment_verified=True,
                 one_row_query_consistent=True, gate_passed=bool(all_passed),
                 exact_input_batch_invariance=True, single_query_check_matmul_precision="highest",
                 max_tf32_single_vs_batch_error=tf32_batch_error, max_ieee_single_vs_batch_error=ieee_batch_error,
                 standalone_labels_changed_by_ieee_precision=precision_label_changes,
                 fresh_reference_trees_reloaded=reference_trees, fresh_reference_contexts_reloaded=reference_contexts,
                 conditional_uncertainty=uncertainty,
                 tests_passed=int(match.group(1)), tests_log_sha256=sha(logpath),
                 results=results, incumbent_platform_score=.81178, kaggle_score=None)
    _write_json(PROJECT_ROOT/f"reports/checks/{NAME}.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
