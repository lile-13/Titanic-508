"""Fixed three-checkpoint study against the .81318 algorithm and default V2."""
import gc
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import read, sha, stats
from scripts.tabicl_finetune_study import STAGES
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P, EXPERIMENTS_DIR
from src.tabpfn_diverse_models import CHECKPOINTS, PARAMS, from_context
from src.train import _write_json, _write_csv

NAME = "tabpfn_diversity_20261101"
WEIGHT = .25


def first_pass(result):
    return (result["corrected"] - result["new_errors"] >= 9
            and result["improved_folds"] >= 3
            and result["log_loss"] <= result["reference_log_loss"] + .002
            and result["versus_default"]["gain"] >= 0)


def stage_pass(rows):
    return (len(rows) == 2 and first_pass(rows[0])
            and all(r["gain"] >= 0 and r["versus_default"]["gain"] >= 0
                    and r["log_loss"] <= r["reference_log_loss"] + .002 for r in rows)
            and np.mean([r["gain"] for r in rows]) >= .001
            and sum(r["improved_folds"] for r in rows) >= 6)


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old = SearchRunner()
    report, out, models = EXPERIMENTS_DIR / NAME, P / "outputs" / NAME, P / "models" / NAME
    for d in (report, out, models):
        d.mkdir(parents=True, exist_ok=True)
    sources = ["scripts/tabpfn_diversity_study.py", "src/tabpfn_diverse_models.py", "src/tabpfn_models.py",
               "src/focused_features.py", "src/competition_features.py", "src/preprocess.py",
               "scripts/audit_unlabeled_views.py", "scripts/tabicl_finetune_study.py", "src/optimization.py"]
    protocol = dict(name=NAME, checkpoints=CHECKPOINTS, params=PARAMS, stages=STAGES, weight=WEIGHT, threshold=.5,
        selection="All three non-alias author V2 classifier checkpoints below 14 MB. Selected by artifact size before reading their predictions; equal probability average, no winning checkpoint selection.",
        baseline="0.25 original trees + 0.75 arithmetic TabICL, actual .81318 algorithm",
        candidate="0.75 baseline + 0.25 mean(gn2p4bpt,vutqq28w,znskzxi4), four views each, seed42",
        matched_control="0.75 same baseline + 0.25 default V2, identical training rows, labels, input builder and four-view settings. Cached default component verified against saved context. Three identical default replicas would give the same deterministic average; runtime is not matched.",
        gate="First repeat: net >=9, >=3/5 improved folds, loss deterioration <=.002, nonnegative gain vs default control. Stop otherwise. Each stage: both gains >=0, mean>=.001, >=6/10 improved folds, each loss deterioration<=.002 and gain vs default>=0. Both stages required for production.",
        limitations="Four historical splits share the same 8693 passengers; adaptive local evidence, not independent blind validation. No threshold, weight, checkpoint or seed search. No test labels or persistent-error identities used for fitting.",
        attribution="Built with PriorLabs-TabPFN; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6",
        inventory_sha256=sha(EXPERIMENTS_DIR / "checkpoint_diversity_20261101/tensor_inventory.json"),
        outcomes_sha256=sha(EXPERIMENTS_DIR / "next_directions_20261101/platform_outcomes_20261101.json"),
        sources={f: sha(P / f) for f in sources},
        data_hashes={f: sha(P / "data/raw" / f) for f in ["train.csv", "test.csv", "sample_submission.csv"]},
        references={f"{run}/fresh_repeat_{r}_predictions.csv": sha(EXPERIMENTS_DIR / run / f"fresh_repeat_{r}_predictions.csv") for _, run in STAGES for r in [1, 2]},
        environment={p: importlib.metadata.version(p) for p in ["tabpfn", "torch", "numpy", "pandas", "scikit-learn"]})
    protocol = json.loads(json.dumps(protocol))
    if (report / "protocol.json").exists():
        assert read(report / "protocol.json") == protocol, "Frozen study changed"
    else:
        _write_json(report / "protocol.json", protocol)
    if (report / "summary.json").exists():
        print("Completed diversity study")
        return
    g = old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    stages = {}
    for stage, run in STAGES:
        rows = []
        for repeat in [1, 2]:
            table = pd.read_csv(EXPERIMENTS_DIR / run / f"fresh_repeat_{repeat}_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported, old.y)
            old_reference = table.alternative.to_numpy() if stage == "paired" else table.reference.to_numpy()
            tfm, default = np.full(len(old.y), np.nan), np.full(len(old.y), np.nan)
            components = {k: np.full(len(old.y), np.nan) for k in CHECKPOINTS}
            for fold in range(1, 6):
                stem = f"fresh_repeat_{repeat}_fold_{fold}" if stage == "paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                source = P / "models" / run / f"{stem}.joblib"
                assert sha(source) == read(EXPERIMENTS_DIR / run / f"{stem}.json")["model_sha256"]
                z = np.load(P / "outputs" / run / f"{stem}.npz")
                ti, vi = z["training"], z["validation"]
                assert (table.fold.iloc[vi] == fold).all()
                assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
                original = joblib.load(source)
                np.testing.assert_array_equal(original.training_y, old.y[ti])
                pd.testing.assert_frame_equal(original.training_X, original.builder.transform(old.features.iloc[ti]))
                tfm[vi] = z["probability"]
                tag = f"{stage}_repeat_{repeat}_fold_{fold}"
                control_source = P / "models/tabpfn_v2_20261031" / f"{tag}.joblib"
                assert sha(control_source) == read(EXPERIMENTS_DIR / "tabpfn_v2_20261031" / f"{tag}.json")["model_sha256"]
                control_model = joblib.load(control_source)
                pd.testing.assert_frame_equal(control_model.training_X, original.training_X)
                np.testing.assert_array_equal(control_model.training_y, original.training_y)
                dz = np.load(P / "outputs/tabpfn_v2_20261031" / f"{tag}.npz")
                np.testing.assert_array_equal(dz["training"], ti)
                np.testing.assert_array_equal(dz["validation"], vi)
                default[vi] = dz["probability"]
                for key in CHECKPOINTS:
                    mp, cp, rp = models / f"{tag}_{key}.joblib", out / f"{tag}_{key}.npz", report / f"{tag}_{key}.json"
                    if rp.exists():
                        record = read(rp)
                        assert sha(mp) == record["model_sha256"] and sha(cp) == record["cache_sha256"]
                        saved = np.load(cp)
                        np.testing.assert_array_equal(saved["training"], ti)
                        np.testing.assert_array_equal(saved["validation"], vi)
                        probability = saved["probability"]
                    else:
                        model = from_context(original, key)
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        probability = model.predict_proba(old.features.iloc[vi])[:, 1]
                        assert model._model.n_estimators_ == 4
                        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
                        joblib.dump(model, mp, compress=3)
                        np.savez_compressed(cp, training=ti, validation=vi, probability=probability)
                        _write_json(rp, dict(checkpoint=key, model_sha256=sha(mp), cache_sha256=sha(cp),
                            source_context_sha256=sha(source), default_context_sha256=sha(control_source),
                            default_cache_sha256=sha(P / "outputs/tabpfn_v2_20261031" / f"{tag}.npz"),
                            source_run=run, source_stem=stem, seconds=time.perf_counter()-start,
                            peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    components[key][vi] = probability
                    print(f"Diversity {tag} {key} complete", flush=True)
                del original, control_model
            reference = .5 * old_reference + .5 * tfm
            component = np.mean(np.stack(list(components.values())), axis=0)
            alternative = (1-WEIGHT) * reference + WEIGHT * component
            control = (1-WEIGHT) * reference + WEIGHT * default
            folds = table.fold.to_numpy()
            result = dict(repeat=repeat, **stats(old.y, reference, alternative, folds),
                versus_default=stats(old.y, control, alternative, folds),
                default_vs_reference=stats(old.y, reference, control, folds),
                g=stats(old.y[g], reference[g], alternative[g], folds[g]),
                component_accuracy=float(((component >= .5) == old.y).mean()))
            rows.append(result)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId, Transported=old.y, fold=table.fold,
                reference=reference, default=default, control=control, **components, component=component,
                alternative=alternative)), report / f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report / f"{stage}_results.json", rows)
            print(f"Diversity {stage} repeat {repeat}: net {result['corrected']-result['new_errors']:+d}; vs default {result['versus_default']['corrected']-result['versus_default']['new_errors']:+d}", flush=True)
            if repeat == 1 and not first_pass(result):
                break
        stages[stage] = rows
        if not stage_pass(rows):
            _write_json(report / "summary.json", dict(status="no_candidate", failed_stage=stage, stages=stages, kaggle_score=None))
            return
    _write_json(report / "summary.json", dict(status="qualified_for_production", stages=stages, kaggle_score=None))


if __name__ == "__main__":
    main()
