"""Replay each new context and matched default; reconstruct the fixed decision."""
import gc
import hashlib

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import assert_stats, read, sha, stats
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P, EXPERIMENTS_DIR
from src.train import _write_json

NAME = "tabpfn_diversity_20261101"


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    report = EXPERIMENTS_DIR / NAME
    protocol, summary = read(report / "protocol.json"), read(report / "summary.json")
    for f, h in protocol["sources"].items():
        assert sha(P / f) == h
    for f, h in protocol["data_hashes"].items():
        assert sha(P / "data/raw" / f) == h
    for f, h in protocol["references"].items():
        assert sha(EXPERIMENTS_DIR / f) == h
    assert sha(EXPERIMENTS_DIR / "next_directions_20261101/platform_outcomes_20261101.json") == protocol["outcomes_sha256"]
    inventory_path = EXPERIMENTS_DIR / "checkpoint_diversity_20261101/tensor_inventory.json"
    assert sha(inventory_path) == protocol["inventory_sha256"]
    tensor_hashes = []
    for record in read(inventory_path)["models"]:
        path = P / "models/tabpfn_pretrained" / record["filename"]
        assert sha(path) == record["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.contiguous().numpy().tobytes())
        assert digest.hexdigest() == record["tensor_state_sha256"]
        tensor_hashes.append(digest.hexdigest())
    assert len(tensor_hashes) == 4 and len(set(tensor_hashes)) == 4
    old = SearchRunner()
    g = old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    replayed = 0
    checked_rows = 0
    for stage, rows in summary["stages"].items():
        run = dict(protocol["stages"])[stage]
        for result in rows:
            repeat = result["repeat"]
            table = pd.read_csv(report / f"{stage}_repeat_{repeat}_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
            prior = pd.read_csv(EXPERIMENTS_DIR / run / f"fresh_repeat_{repeat}_predictions.csv", dtype={"PassengerId": str}, float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId, old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported, old.y)
            np.testing.assert_array_equal(table.fold, prior.fold)
            for fold in range(1, 6):
                tag = f"{stage}_repeat_{repeat}_fold_{fold}"
                stem = f"fresh_repeat_{repeat}_fold_{fold}" if stage == "paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                z = np.load(P / "outputs" / run / f"{stem}.npz")
                ti, vi = z["training"], z["validation"]
                assert (table.fold.iloc[vi] == fold).all()
                assert not set(old.features.GroupId.iloc[ti]) & set(old.features.GroupId.iloc[vi])
                old_prob = prior.alternative.to_numpy() if stage == "paired" else prior.reference.to_numpy()
                np.testing.assert_allclose(table.reference.iloc[vi], .5*old_prob[vi]+.5*z["probability"], atol=1e-14, rtol=0)
                original_path = P / "models" / run / f"{stem}.joblib"
                original = joblib.load(original_path)
                for key in protocol["checkpoints"]:
                    record = read(report / f"{tag}_{key}.json")
                    mp, cp = P / "models" / NAME / f"{tag}_{key}.joblib", P / "outputs" / NAME / f"{tag}_{key}.npz"
                    assert sha(mp) == record["model_sha256"] and sha(cp) == record["cache_sha256"]
                    assert sha(original_path) == record["source_context_sha256"]
                    model = joblib.load(mp)
                    assert model._model is None and model.checkpoint_key == key
                    pd.testing.assert_frame_equal(model.training_X, original.training_X)
                    pd.testing.assert_frame_equal(model.training_X, model.builder.transform(old.features.iloc[ti]))
                    np.testing.assert_array_equal(model.training_y, old.y[ti])
                    saved = np.load(cp)
                    np.testing.assert_array_equal(saved["training"], ti)
                    np.testing.assert_array_equal(saved["validation"], vi)
                    actual = model.predict_proba(old.features.iloc[vi])[:, 1]
                    assert model._model.n_estimators_ == 4
                    np.testing.assert_allclose(actual, saved["probability"], atol=1e-7, rtol=0)
                    np.testing.assert_allclose(table[key].iloc[vi], actual, atol=1e-7, rtol=0)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    replayed += 1
                control_path = P / "models/tabpfn_v2_20261031" / f"{tag}.joblib"
                control_cache = P / "outputs/tabpfn_v2_20261031" / f"{tag}.npz"
                assert sha(control_path) == record["default_context_sha256"]
                assert sha(control_cache) == record["default_cache_sha256"]
                control = joblib.load(control_path)
                np.testing.assert_allclose(control.predict_proba(old.features.iloc[vi])[:, 1], table.default.iloc[vi], atol=1e-7, rtol=0)
                del original, control
                gc.collect()
                torch.cuda.empty_cache()
                checked_rows += len(vi)
                print(f"Replayed diversity and default: {tag}", flush=True)
            component = table[list(protocol["checkpoints"])].mean(axis=1).to_numpy()
            ref, alt = table.reference.to_numpy(), table.alternative.to_numpy()
            np.testing.assert_allclose(table.component, component, atol=1e-14, rtol=0)
            np.testing.assert_allclose(alt, .75*ref+.25*component, atol=1e-14, rtol=0)
            np.testing.assert_allclose(table.control, .75*ref+.25*table.default, atol=1e-14, rtol=0)
            assert_stats(stats(old.y, ref, alt, table.fold.to_numpy()), result)
            assert_stats(stats(old.y, table.control.to_numpy(), alt, table.fold.to_numpy()), result["versus_default"])
            assert_stats(stats(old.y[g], ref[g], alt[g], table.fold.to_numpy()[g]), result["g"])
    # Independent implementation of the frozen gate, not the study's gate helper.
    gates = {}
    for stage, rows in summary["stages"].items():
        first = rows[0]
        first_ok = (first["corrected"]-first["new_errors"] >= 9 and first["improved_folds"] >= 3
                    and first["log_loss"]-first["reference_log_loss"] <= .002 and first["versus_default"]["gain"] >= 0)
        if not first_ok:
            assert len(rows) == 1, "Failed screen should stop"
        gates[stage] = bool(first_ok and len(rows) == 2 and np.mean([r["gain"] for r in rows]) >= .001
            and sum(r["improved_folds"] for r in rows) >= 6
            and all(r["gain"] >= 0 and r["versus_default"]["gain"] >= 0
                    and r["log_loss"]-r["reference_log_loss"] <= .002 for r in rows))
    passed = len(gates) == 2 and all(gates.values())
    assert summary["status"] == ("qualified_for_production" if passed else "no_candidate")
    if not passed:
        assert not gates[summary["failed_stage"]]
    _write_json(P / "reports/checks" / f"{NAME}.json", dict(passed=True,
        new_contexts_replayed=replayed, matched_default_contexts_replayed=checked_rows // len(old.y) * 5,
        validation_rows_checked=checked_rows, unique_tensor_states=4, split_and_input_checks_passed=True,
        probability_replay_atol=1e-7, gate_results=gates, qualification=summary["status"],
        protocol_sha256=sha(report / "protocol.json"), summary_sha256=sha(report / "summary.json"),
        audit_source_sha256=sha(P / "scripts/audit_tabpfn_diversity.py")))
    print("Diversity audit passed", flush=True)


if __name__ == "__main__":
    main()
