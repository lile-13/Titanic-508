"""Replay components and validate every user-requested exploration CSV."""
import gc
import re

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.prepare_submission_batch import NAME,VARIANTS,mix,sha,read
from scripts.audit_unlabeled_views import stats
from scripts.tabicl_finetune_study import STAGES
from src import preprocess as pp
from src.optimization import SearchRunner,predict_ensemble
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.tabpfn_models import CHECKPOINT
from src.train import _write_json


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME,P/"models"/NAME
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    for f,h in protocol["sources"].items():
        assert sha(P/f)==h
    for f,h in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/f)==h
    assert sha(P/"models/tabpfn_pretrained"/CHECKPOINT)==protocol["checkpoint_sha256"]
    assert sha(EXPERIMENTS_DIR/"tabpfn_v2_20261031/summary.json")==protocol["parent_tabpfn_summary_sha256"]
    assert sha(EXPERIMENTS_DIR/"tabpfn_v2_20261031/protocol.json")==protocol["parent_tabpfn_protocol_sha256"]
    best_file=P/"outputs/kaggle_ready_4690/submission_tabicl_mean4690.csv"
    assert sha(best_file)==protocol["original_best_sha256"]
    best_frame=pd.read_csv(best_file,dtype={"PassengerId":str})
    stored=pd.read_csv(out/"test_components.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(stored.PassengerId,old.raw_test.PassengerId)
    tree,_=predict_ensemble(P/"models/optimization_20261025/ensemble.joblib")
    np.testing.assert_allclose(stored.tree,tree,rtol=0,atol=1e-14)
    pfn,tfm=np.zeros(len(tree)),np.zeros(len(tree))
    max_error=0.
    for fold in range(1,6):
        record=read(report/f"fold_{fold}.json")
        mp=models/f"pfn_fold_{fold}.joblib"
        source=P/f"models/tabicl_probability_mean_20261030/final_fold_{fold}.joblib"
        assert sha(mp)==record["model_sha256"] and sha(source)==record["source_context_sha256"]
        candidate,original=joblib.load(mp),joblib.load(source)
        assert candidate._model is None
        pd.testing.assert_frame_equal(candidate.training_X,original.training_X)
        np.testing.assert_array_equal(candidate.training_y,original.training_y)
        z=np.load(out/f"fold_{fold}.npz")
        for member,key,total in [(candidate,"pfn",pfn),(original,"tfm",tfm)]:
            q=member.predict_proba(old.test_features)[:,1]
            np.testing.assert_allclose(q,z[key],rtol=0,atol=2e-6)
            max_error=max(max_error,float(np.abs(q-z[key]).max()))
            total+=q/5
            member._model=None
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Audited submission components fold {fold}",flush=True)
    best=.5*tree+.5*tfm
    np.testing.assert_allclose(stored.pfn,pfn,rtol=0,atol=2e-6)
    np.testing.assert_allclose(stored.tfm,tfm,rtol=0,atol=2e-6)
    np.testing.assert_allclose(stored.best,best,rtol=0,atol=2e-6)
    np.testing.assert_array_equal(best>=.5,best_frame.Transported)
    template=pp.load_submission_template()
    labels={}
    records=[]
    for record in summary["candidates"]:
        variant=record["variant"]
        path=P/record["file"]
        frame=pd.read_csv(path,dtype={"PassengerId":str})
        assert sha(path)==record["sha256"]
        assert list(frame)==["PassengerId","Transported"] and len(frame)==4277
        assert frame.PassengerId.is_unique and frame.Transported.dtype==bool
        assert not frame.isna().any().any()
        np.testing.assert_array_equal(frame.PassengerId,template.PassengerId)
        q=mix(VARIANTS[variant],best,pfn,tfm)
        np.testing.assert_array_equal(frame.Transported,q>=.5)
        labels[variant]=frame.Transported.to_numpy()
        assert int((labels[variant]!=best_frame.Transported.to_numpy()).sum())==record["changes_vs_current_best"]
        assert record["score"] is None and record["kaggle_score"] is None
        assert record["formal_promotion"] is False and record["exploratory"] is True
        assert record["status"]=="prepared_not_submitted"
        records.append(dict(variant=variant,file=record["file"],sha256=record["sha256"],rows=4277,
                            changes_vs_current_best=record["changes_vs_current_best"]))
    for a in labels:
        assert (labels[a]!=best_frame.Transported.to_numpy()).any()
    for i,a in enumerate(labels):
        for b in list(labels)[i+1:]:
            assert (labels[a]!=labels[b]).any()
            assert int((labels[a]!=labels[b]).sum())==summary["pairwise_label_differences"][f"{a}/{b}"]
    # All model/version variants retain the same known rows in these descriptive CV tables.
    saved=read(report/"historical_local_comparisons.json")["rows"]
    for stage,run in STAGES:
        for repeat in [1,2]:
            t=pd.read_csv(EXPERIMENTS_DIR/f"tabpfn_v2_20261031/{stage}_repeat_{repeat}_predictions.csv",float_precision="round_trip")
            base_tfm=np.full(len(old.y),np.nan)
            for fold in range(1,6):
                stem=f"fresh_repeat_{repeat}_fold_{fold}" if stage=="paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                z=np.load(P/"outputs"/run/f"{stem}.npz")
                base_tfm[z["validation"]]=z["probability"]
            for variant,spec in VARIANTS.items():
                q=mix(spec,t.reference.to_numpy(),t.component.to_numpy(),base_tfm)
                result=stats(old.y,t.reference.to_numpy(),q,t.fold.to_numpy())
                row=next(r for r in saved if r["variant"]==variant and r["stage"]==stage and r["repeat"]==repeat)
                for k,v in result.items():
                    np.testing.assert_allclose(v,row[k],rtol=0,atol=1e-14)
    log=(P/"outputs/foundation_round_20261031_tests.log").read_text(encoding="utf-8-sig")
    assert re.search(r"Ran 64 tests",log) and re.search(r"\bOK\b",log)
    _write_json(P/f"reports/checks/{NAME}.json",dict(passed=True,tests=64,
        models_reloaded=10,max_component_reload_error=max_error,candidates=records,
        all_four_distinct=True,template_order_boolean_labels_and_hashes=True,
        original_best_reproduced=True,formal_promotion=False,submitted_to_kaggle=False,
        caveat="Format, provenance and reproducibility checks passed; this does not establish accuracy improvement. User requested these exploratory submissions despite failed local stability gates."))


if __name__=="__main__":
    main()
