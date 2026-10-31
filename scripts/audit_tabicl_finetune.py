"""Reload both arms; independently reconstruct nested splits and selection."""
import gc
import importlib
import importlib.metadata
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from tabicl._sklearn.preprocessing import TransformToNumerical

from scripts.audit_unlabeled_views import stats, assert_stats
from scripts.tabicl_finetune_study import NAME, STAGES, PARAMS, INNER_SEED, sha, read, passes
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabicl_finetune_probe import predictor
from src.tabicl_models import CHECKPOINT
from src.train import _write_json


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,PROJECT_ROOT/"outputs"/NAME,PROJECT_ROOT/"models"/NAME
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    for p,digest in protocol["sources"].items():
        assert sha(PROJECT_ROOT/p)==digest
    for p,digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==digest
    for p,digest in protocol["installed_sources"].items():
        assert sha(Path(importlib.import_module(p).__file__))==digest
    for p,version in protocol["environment"].items():
        assert importlib.metadata.version(p)==version
    for p,digest in protocol["reference_files"].items():
        assert sha(EXPERIMENTS_DIR/p)==digest
    for run,digest in protocol["reference_audits"].items():
        path=PROJECT_ROOT/f"reports/checks/{run}.json"
        assert sha(path)==digest and read(path)["passed"]
    author=PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT
    assert sha(author)==protocol["checkpoint_sha256"]
    initial=torch.load(author,map_location="cpu",weights_only=False)["state_dict"]
    reloaded,max_error,selected_epochs,baseline_max_difference=0,0.,[],0.
    results={}
    for stage,run in STAGES:
        if stage not in summary["stages"]:
            break
        rows=[]
        for saved in summary["stages"][stage]:
            repeat=saved["repeat"]
            table=pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            reference=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            np.testing.assert_allclose(table.reference,reference.alternative if stage=="paired" else reference.reference,rtol=0,atol=1e-14)
            for fold in range(1,6):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                record=read(report/f"{tag}.json")
                folder=models/tag
                assert sha(folder/"context.joblib")==record["context_sha256"]
                assert sha(folder/"best.ckpt")==record["checkpoint_sha256"]
                source=PROJECT_ROOT/"models"/run/f"{record['source_stem']}.joblib"
                assert sha(source)==record["source_context_sha256"]
                original=joblib.load(source)
                z=np.load(out/f"{tag}.npz")
                ti,vi=z["training"],z["validation"]
                source_z=np.load(PROJECT_ROOT/"outputs"/run/f"{record['source_stem']}.npz")
                np.testing.assert_array_equal(ti,source_z["training"])
                np.testing.assert_array_equal(vi,source_z["validation"])
                np.testing.assert_allclose(table.base_component.iloc[vi],source_z["probability"],rtol=0,atol=1e-14)
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                assert (table.fold.iloc[vi]==fold).all()
                ii,iv=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=INNER_SEED).split(ti,groups=old.features.GroupId.iloc[ti]))
                np.testing.assert_array_equal(z["inner_training"],ti[ii])
                np.testing.assert_array_equal(z["inner_validation"],ti[iv])
                assert not set(old.features.GroupId.iloc[ti[ii]])&set(old.features.GroupId.iloc[ti[iv]])
                context=joblib.load(folder/"context.joblib")
                np.testing.assert_array_equal(context["y"],old.y[ti])
                fresh_encoder=TransformToNumerical().fit(original.training_X.iloc[ii])
                np.testing.assert_allclose(context["encoded_X"],fresh_encoder.transform(original.training_X),rtol=0,atol=0,equal_nan=True)
                query_frame=context["builder"].transform(old.features.iloc[vi])
                pd.testing.assert_frame_equal(query_frame,original.builder.transform(old.features.iloc[vi]))
                query=context["encoder"].transform(query_frame)
                np.testing.assert_allclose(query,fresh_encoder.transform(query_frame),rtol=0,atol=0,equal_nan=True)
                checkpoint=torch.load(folder/"best.ckpt",map_location="cpu",weights_only=False)
                changed=[k for k in initial if not torch.equal(initial[k],checkpoint["state_dict"][k])]
                assert all(k.startswith("icl_predictor.") for k in changed)
                assert all(torch.isfinite(v).all() for v in checkpoint["state_dict"].values())
                selected=0
                for i,h in enumerate(record["history"][1:],1):
                    assert np.isfinite(h["primary"])
                    if h["primary"]>record["history"][selected]["primary"]+PARAMS["min_delta"]:
                        selected=i
                assert selected==record["selected_epoch"]
                assert record["epochs_run"]==len(record["history"])-1<=PARAMS["epochs"]
                assert selected>0 or not changed
                selected_epochs.append(selected)
                for label,path,column in [("fine",folder/"best.ckpt","finetuned"),("control",author,"untuned_control")]:
                    model=predictor(path,context)
                    p=model.predict_proba(query)[:,1]
                    delta=float(np.max(np.abs(p-z[label])))
                    max_error=max(max_error,delta)
                    np.testing.assert_allclose(p,z[label],rtol=0,atol=2e-6)
                    np.testing.assert_allclose(table[column].iloc[vi],z[label],rtol=0,atol=1e-14)
                    np.testing.assert_array_equal(p>=.5,z[label]>=.5)
                    reloaded+=1
                    if label=="fine":
                        inner_context=dict(encoded_X=context["encoded_X"][ii],y=old.y[ti[ii]])
                        inner=predictor(path,inner_context)
                        q=inner.predict_proba(context["encoded_X"][iv])[:,1]
                        np.testing.assert_allclose(log_loss(old.y[ti[iv]],q),record["best_inner_log_loss"],rtol=0,atol=2e-6)
                        del inner
                    else:
                        baseline_max_difference=max(baseline_max_difference,float(np.max(np.abs(p-table.base_component.iloc[vi].to_numpy()))))
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f"Audited {tag}",flush=True)
            alternative=table.reference+.25*(table.finetuned-table.base_component)
            control=table.reference+.25*(table.untuned_control-table.base_component)
            np.testing.assert_allclose(alternative,table.alternative,rtol=0,atol=1e-14)
            np.testing.assert_allclose(control,table.control,rtol=0,atol=1e-14)
            row=dict(repeat=repeat,**stats(old.y,table.reference.to_numpy(),alternative.to_numpy(),table.fold.to_numpy()),
                control_accuracy=float(((control>=.5)==old.y).mean()),
                gain_over_control=float(((alternative>=.5)==old.y).mean()-((control>=.5)==old.y).mean()))
            assert_stats(row,saved)
            rows.append(row)
        results[stage]=rows
    gate= len(results)==2 and all(passes(rows) for rows in results.values())
    assert (summary["status"]=="qualified_for_production")==gate
    if not gate:
        assert not passes(results[summary["failed_stage"]])
        assert not (models/"ensemble.joblib").exists()
        assert not list(out.glob("submission*.csv"))
    test_log=(PROJECT_ROOT/f"outputs/{NAME}_tests.log").read_text(encoding="utf-8-sig")
    assert re.search(r"Ran 61 tests",test_log) and re.search(r"\bOK\b",test_log)
    best=PROJECT_ROOT/"outputs/kaggle_ready_4690/submission_tabicl_mean4690.csv"
    assert sha(best)=="bd2858e0412f37d22ac38040401bfbe3e459a5ab5690f0e4b98924c78aed8ac0"
    _write_json(PROJECT_ROOT/f"reports/checks/{NAME}.json",dict(passed=True,gate_passed=gate,
        prediction_models_reloaded=reloaded,max_reload_error=max_error,selected_epochs=selected_epochs,
        max_control_difference_from_historical_component=baseline_max_difference,
        nested_groups_and_encoder_rebuilt=True,chosen_inner_loss_reproduced=True,
        only_icl_predictor_weights_changed=True,tests=61,stages=results,
        caveat="Only fixed head-only fine tuning was screened. All evaluation passengers are historical. No new Kaggle score."))


if __name__=="__main__":
    main()
