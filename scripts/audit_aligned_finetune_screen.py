"""Independent checkpoint replay for the corrected-input historical screen."""
import gc
import re

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import stats, assert_stats
from scripts.screen_aligned_finetune import NAME, sha, read
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P, EXPERIMENTS_DIR
from src.tabicl_models import make_model
from src.train import _write_json, _write_csv


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    parent=EXPERIMENTS_DIR/"tabicl_finetune_20261031"
    assert sha(parent/"protocol.json")==protocol["parent_protocol_sha256"]
    assert sha(parent/"summary.json")==protocol["parent_summary_sha256"]
    assert sha(parent/"encoding_diagnostic.json")==protocol["alignment_diagnostic_sha256"]
    for p,h in protocol["sources"].items():
        assert sha(P/p)==h
    table=pd.read_csv(report/"predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    source=EXPERIMENTS_DIR/"tabicl_probability_mean_20261030/fresh_repeat_1_predictions.csv"
    assert sha(source)==protocol["references_sha256"]
    before=pd.read_csv(source,float_precision="round_trip")
    np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
    np.testing.assert_array_equal(table.Transported,old.y)
    np.testing.assert_allclose(table.reference,before.alternative,rtol=0,atol=1e-14)
    max_error=0.
    for fold in range(1,6):
        stem=f"fresh_repeat_1_fold_{fold}"
        context_path=P/f"models/tabicl_probability_mean_20261030/{stem}.joblib"
        checkpoint=P/f"models/tabicl_finetune_20261031/paired_repeat_1_fold_{fold}/best.ckpt"
        record=summary["folds"][fold-1]
        assert sha(context_path)==record["context_sha256"]
        assert sha(context_path)==read(EXPERIMENTS_DIR/f"tabicl_probability_mean_20261030/{stem}.json")["model_sha256"]
        assert sha(checkpoint)==record["checkpoint_sha256"]==protocol["checkpoints"][str(fold)]
        assert sha(checkpoint)==read(parent/f"paired_repeat_1_fold_{fold}.json")["checkpoint_sha256"]
        z=np.load(out/f"fold_{fold}.npz")
        original=np.load(P/f"outputs/tabicl_probability_mean_20261030/{stem}.npz")
        ti,vi=z["training"],z["validation"]
        np.testing.assert_array_equal(ti,original["training"])
        np.testing.assert_array_equal(vi,original["validation"])
        assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
        assert (table.fold.iloc[vi]==fold).all()
        member=joblib.load(context_path)
        np.testing.assert_array_equal(member.training_y,old.y[ti])
        query=member.builder.transform(old.features.iloc[vi])
        for path,column,key in [(None,"control_component","control"),(checkpoint,"tuned_component","tuned")]:
            model=make_model().set_params(average_logits=False)
            if path is not None:
                model.set_params(model_path=str(path))
            model.fit(member.training_X,old.y[ti])
            p=model.predict_proba(query)[:,1]
            np.testing.assert_allclose(p,z[key],rtol=0,atol=2e-6)
            np.testing.assert_allclose(table[column].iloc[vi],z[key],rtol=0,atol=1e-14)
            max_error=max(max_error,float(np.abs(p-z[key]).max()))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        np.testing.assert_allclose(z["control"],original["probability"],rtol=0,atol=2e-6)
        np.testing.assert_array_equal(z["control"]>=.5,original["probability"]>=.5)
        print(f"Audited aligned fine-tuning fold {fold}",flush=True)
    alternative=table.reference+.25*(table.tuned_component-table.original_component)
    np.testing.assert_allclose(alternative,table.alternative,rtol=0,atol=1e-14)
    np.testing.assert_allclose(table.original_component,before.component,rtol=0,atol=1e-14)
    g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    results=stats(old.y,table.reference.to_numpy(),alternative.to_numpy(),table.fold.to_numpy())
    assert_stats(results,summary["results"])
    for name,mask in [("g",g),("non_g",~g)]:
        result=stats(old.y[mask],table.reference.to_numpy()[mask],alternative.to_numpy()[mask],table.fold.to_numpy()[mask])
        assert_stats(result,summary["results"][name])
    gate=(results["gain"]>=.001 and results["improved_folds"]>=3 and summary["results"]["g"]["gain"]>=0
        and summary["results"]["non_g"]["gain"]>=0 and results["log_loss"]<=results["reference_log_loss"]+.002)
    assert summary["screen_passed"]==gate
    assert summary["status"]==("requires_additional_validation" if gate else "no_candidate")
    historical=pd.read_csv(EXPERIMENTS_DIR/"accuracy_headroom_20261031/passenger_diagnostics.csv",dtype={"PassengerId":str})
    np.testing.assert_array_equal(historical.PassengerId,table.PassengerId)
    persistent=historical.wrong_repetitions.eq(4).to_numpy()
    fixed=((table.reference.to_numpy()>=.5)!=old.y)&((alternative.to_numpy()>=.5)==old.y)
    changed=(table.reference>=.5)!=(alternative>=.5)
    cases=table.loc[changed,["PassengerId","Transported","fold","reference","alternative"]].copy()
    cases["historical_wrong_repetitions"]=historical.wrong_repetitions[changed]
    cases["G_deck"]=g[changed.to_numpy()]
    _write_csv(cases,report/"changed_cases_postfit.csv")
    log=(P/"outputs/hard_examples_g_20261031_tests.log").read_text(encoding="utf-8-sig")
    assert re.search(r"Ran 63 tests",log) and re.search(r"\bOK\b",log)
    _write_json(P/f"reports/checks/{NAME}.json",dict(passed=True,screen_passed=gate,
        models_reloaded=10,max_reload_error=max_error,original_controls_reproduced=True,tests=63,
        persistent_1203_corrected=int((fixed&persistent).sum()),
        caveat="Historical screen; no independent new passengers or production qualification."))


if __name__=="__main__":
    main()
