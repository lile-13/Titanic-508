"""Independent replay of row adaptation and the four TabPFN CV repetitions."""
import gc
import importlib
import importlib.metadata
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from tabicl._sklearn.preprocessing import TransformToNumerical

from scripts.audit_unlabeled_views import stats,assert_stats
from scripts.tabicl_finetune_study import sha,read,STAGES,INNER_SEED
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.tabicl_models import CHECKPOINT as ICL_CHECKPOINT,make_model
from src.train import _write_json


def common(protocol):
    for f,h in protocol["sources"].items():
        assert sha(P/f)==h
    for f,h in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/f)==h
    for module,h in protocol["installed_sources"].items():
        assert sha(Path(importlib.import_module(module).__file__))==h
    for package,version in protocol["environment"].items():
        assert importlib.metadata.version(package)==version
    for f,h in protocol["references"].items():
        assert sha(EXPERIMENTS_DIR/f)==h
    for run,h in protocol["reference_audits"].items():
        path=P/f"reports/checks/{run}.json"
        assert sha(path)==h and read(path)["passed"]


def check_metrics(y,table,saved,g):
    ref,alt,folds=table.reference.to_numpy(),table.alternative.to_numpy(),table.fold.to_numpy()
    result=stats(y,ref,alt,folds)
    assert_stats(result,saved)
    for name,mask in [("g",g),("non_g",~g)]:
        assert_stats(stats(y[mask],ref[mask],alt[mask],folds[mask]),saved[name])


def gate(rows,cohorts):
    return bool(len(rows)==2 and all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002
        and (not cohorts or (r["g"]["gain"]>=0 and r["non_g"]["gain"]>=0)) for r in rows)
        and sum(r["gain"] for r in rows)/2>=.001 and sum(r["improved_folds"] for r in rows)>=6)


def audit_pfn(old):
    from src.tabpfn_models import CHECKPOINT
    name="tabpfn_v2_20261031"
    report,out,models=EXPERIMENTS_DIR/name,P/"outputs"/name,P/"models"/name
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    common(protocol)
    assert sha(P/"models/tabpfn_pretrained"/CHECKPOINT)==protocol["checkpoint_sha256"]
    assert sha(report/"pretrained_source.json")==protocol["pretrained_source_sha256"]
    g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    reloaded,max_error=0,0.
    for stage,run in STAGES:
        if stage not in summary["stages"]:
            break
        for saved in summary["stages"][stage]:
            repeat=saved["repeat"]
            table=pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            original_table=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            np.testing.assert_allclose(table.reference,original_table.alternative if stage=="paired" else original_table.reference,rtol=0,atol=1e-14)
            for fold in range(1,6):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                record=read(report/f"{tag}.json")
                mp=models/f"{tag}.joblib"
                assert sha(mp)==record["model_sha256"]
                source=P/"models"/run/f"{record['source_stem']}.joblib"
                assert sha(source)==record["source_context_sha256"]
                a,b=joblib.load(mp),joblib.load(source)
                assert a._model is None
                pd.testing.assert_frame_equal(a.training_X,b.training_X)
                np.testing.assert_array_equal(a.training_y,b.training_y)
                z=np.load(out/f"{tag}.npz")
                source_z=np.load(P/"outputs"/run/f"{record['source_stem']}.npz")
                ti,vi=z["training"],z["validation"]
                np.testing.assert_array_equal(ti,source_z["training"])
                np.testing.assert_array_equal(vi,source_z["validation"])
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                assert (table.fold.iloc[vi]==fold).all()
                np.testing.assert_array_equal(a.training_y,old.y[ti])
                q=a.predict_proba(old.features.iloc[vi])[:,1]
                np.testing.assert_allclose(q,z["probability"],rtol=0,atol=2e-6)
                np.testing.assert_array_equal(q>=.5,z["probability"]>=.5)
                np.testing.assert_allclose(table.component.iloc[vi],z["probability"],rtol=0,atol=1e-14)
                assert a._model.n_estimators_==4
                max_error=max(max_error,float(np.abs(q-z["probability"]).max()))
                reloaded+=1
                del a,b
                gc.collect()
                torch.cuda.empty_cache()
            np.testing.assert_allclose(table.alternative,.75*table.reference+.25*table.component,rtol=0,atol=1e-14)
            check_metrics(old.y,table,saved,g)
            print(f"Audited TabPFN {stage} repeat {repeat}",flush=True)
    qualified=len(summary["stages"])==2 and all(gate(rows,False) for rows in summary["stages"].values())
    assert qualified==(summary["status"]=="qualified_for_production")
    _write_json(P/f"reports/checks/{name}.json",dict(passed=True,gate_passed=qualified,
        models_reloaded=reloaded,max_reload_error=max_error,source_and_input_and_groups_verified=True,tests=64,
        caveat="Formal local stability gate remains failed; separately authorized manual candidates are exploratory, not promotion."))


def audit_row(old):
    name="tabicl_row_adaptation_20261031"
    report,out,models=EXPERIMENTS_DIR/name,P/"outputs"/name,P/"models"/name
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    common(protocol)
    author=P/"models/tabicl_pretrained"/ICL_CHECKPOINT
    assert sha(author)==protocol["checkpoint_sha256"]
    initial=torch.load(author,map_location="cpu",weights_only=False)["state_dict"]
    g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    reloaded,max_error,epochs=0,0.,[]
    for stage,run in STAGES:
        if stage not in summary["stages"]:
            break
        for saved in summary["stages"][stage]:
            repeat=saved["repeat"]
            table=pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            source_table=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            np.testing.assert_allclose(table.reference,source_table.alternative if stage=="paired" else source_table.reference,rtol=0,atol=1e-14)
            for fold in range(1,6):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                record=read(report/f"{tag}.json")
                folder=models/tag
                assert sha(folder/"context.joblib")==record["context_sha256"]
                assert sha(folder/"best.ckpt")==record["checkpoint_sha256"]
                source=P/"models"/run/f"{record['source_stem']}.joblib"
                assert sha(source)==record["source_context_sha256"]
                original=joblib.load(source)
                context=joblib.load(folder/"context.joblib")
                pd.testing.assert_frame_equal(context["training_X"],original.training_X)
                z=np.load(out/f"{tag}.npz")
                base_z=np.load(P/"outputs"/run/f"{record['source_stem']}.npz")
                ti,vi=z["training"],z["validation"]
                np.testing.assert_array_equal(ti,base_z["training"])
                np.testing.assert_array_equal(vi,base_z["validation"])
                np.testing.assert_array_equal(context["training_y"],old.y[ti])
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                assert (table.fold.iloc[vi]==fold).all()
                ii,iv=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=INNER_SEED).split(ti,groups=old.features.GroupId.iloc[ti]))
                np.testing.assert_array_equal(z["inner_training"],ti[ii])
                np.testing.assert_array_equal(z["inner_validation"],ti[iv])
                assert not set(old.features.GroupId.iloc[ti[ii]])&set(old.features.GroupId.iloc[ti[iv]])
                X=context["training_X"]
                encoder=TransformToNumerical().fit(X.iloc[ii])
                np.testing.assert_allclose(encoder.transform(X),context["inner_encoder"].transform(X),rtol=0,atol=0,equal_nan=True)
                checkpoint=torch.load(folder/"best.ckpt",map_location="cpu",weights_only=False)
                changed=[k for k in initial if not torch.equal(initial[k],checkpoint["state_dict"][k])]
                assert all(k.startswith(("row_interactor.","icl_predictor.")) for k in changed)
                assert all(torch.isfinite(v).all() for v in checkpoint["state_dict"].values())
                selected=0
                for i,h in enumerate(record["history"][1:],1):
                    if h["primary"]>record["history"][selected]["primary"]+protocol["params"]["min_delta"]:
                        selected=i
                assert selected==record["selected_epoch"] and (selected>0 or not changed)
                epochs.append(selected)
                inner=make_model().set_params(model_path=str(folder/"best.ckpt"),average_logits=False)
                inner.fit(encoder.transform(X.iloc[ii]),old.y[ti[ii]])
                q=inner.predict_proba(encoder.transform(X.iloc[iv]))[:,1]
                np.testing.assert_allclose(log_loss(old.y[ti[iv]],q),record["best_inner_log_loss"],rtol=0,atol=2e-6)
                del inner
                query=context["builder"].transform(old.features.iloc[vi])
                for checkpoint_path,key,column in [(author,"control","control_component"),(folder/"best.ckpt","tuned","tuned_component")]:
                    model=make_model().set_params(model_path=str(checkpoint_path),average_logits=False).fit(X,old.y[ti])
                    q=model.predict_proba(query)[:,1]
                    np.testing.assert_allclose(q,z[key],rtol=0,atol=2e-6)
                    np.testing.assert_allclose(table[column].iloc[vi],z[key],rtol=0,atol=1e-14)
                    max_error=max(max_error,float(np.abs(q-z[key]).max()))
                    reloaded+=1
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                np.testing.assert_allclose(z["control"],base_z["probability"],rtol=0,atol=2e-6)
                print(f"Audited row adaptation {tag}",flush=True)
            np.testing.assert_allclose(table.alternative,table.reference+.25*(table.tuned_component-table.original_component),rtol=0,atol=1e-14)
            check_metrics(old.y,table,saved,g)
    qualified=len(summary["stages"])==2 and all(gate(rows,True) for rows in summary["stages"].values())
    assert qualified==(summary["status"]=="qualified_for_production")
    _write_json(P/f"reports/checks/{name}.json",dict(passed=True,gate_passed=qualified,models_reloaded=reloaded,
        max_reload_error=max_error,selected_epochs=epochs,column_embedding_frozen=True,
        inner_groups_encoder_and_loss_reproduced=True,final_original_controls_reproduced=True,tests=64))


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    log=(P/"outputs/foundation_round_20261031_tests.log").read_text(encoding="utf-8-sig")
    assert re.search(r"Ran 64 tests",log) and re.search(r"\bOK\b",log)
    old=SearchRunner()
    audit_pfn(old)
    audit_row(old)


if __name__=="__main__":
    main()
