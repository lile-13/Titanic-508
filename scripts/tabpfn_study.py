"""A fixed public TabPFN-V2 addition to the actual Kaggle-best algorithm."""
import gc
import importlib
import importlib.metadata
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import stats
from scripts.tabicl_finetune_study import sha, read, STAGES
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P, EXPERIMENTS_DIR
from src.tabpfn_models import CHECKPOINT,CHECKPOINT_SHA256,PARAMS,from_context
from src.train import _write_json,_write_csv

NAME="tabpfn_v2_20261031"
WEIGHT=.25


def passes(rows):
    return (len(rows)==2 and all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows)
        and np.mean([r["gain"] for r in rows])>=.001 and sum(r["improved_folds"] for r in rows)>=6)


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME,P/"models"/NAME
    for d in [report,out,models]:
        d.mkdir(parents=True,exist_ok=True)
    checkpoint=P/"models/tabpfn_pretrained"/CHECKPOINT
    assert sha(checkpoint)==CHECKPOINT_SHA256
    assert read(report/"pretrained_source.json")["sha256"]==CHECKPOINT_SHA256
    files=["scripts/tabpfn_study.py","src/tabpfn_models.py","src/focused_features.py","src/competition_features.py","src/preprocess.py","scripts/audit_unlabeled_views.py"]
    modules=["tabpfn.classifier","tabpfn.model_loading","tabpfn.preprocessing.configs"]
    protocol=dict(name=NAME,params=PARAMS,model_version="Explicit V2; no default/latest checkpoint or gated version",
        weight=WEIGHT,threshold=.5,stages=STAGES,checkpoint_sha256=CHECKPOINT_SHA256,
        input="Reuse the actual incumbent TabICL context's builder, raw compact18 training_X and training_y. Exactly the same outer train/validation rows; only the new component architecture and its own preprocessing change.",
        combination=".75 current Kaggle-best algorithm + .25 TabPFN V2; four internal views, arithmetic probability aggregation, fixed seed42; no input/model/weight/threshold search or fine tuning",
        promotion="Both repeat gains nonnegative, mean gain>=.001, >=6/10 strictly improved folds, each logloss deterioration<=.002. First negative repeat, bad loss or zero improved folds stops; both existing historical stages required before production.",
        platform_target="Kaggle accuracy >.82; local CV cannot certify the platform outcome",
        caveat="These are previously inspected passengers and partitions, so evidence is adaptive. G metrics and persistent error cohorts are descriptive, not fitting/routing rules. No competition data sent to external inference services.",
        attribution="Prior Labs, TabPFN V2; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6",
        sources={f:sha(P/f) for f in files},installed_sources={m:sha(Path(importlib.import_module(m).__file__)) for m in modules},
        data_hashes=read(old.report/"protocol.json")["data_hashes"],pretrained_source_sha256=sha(report/"pretrained_source.json"),
        references={f"{run}/fresh_repeat_{r}_predictions.csv":sha(EXPERIMENTS_DIR/run/f"fresh_repeat_{r}_predictions.csv") for _,run in STAGES for r in [1,2]},
        reference_audits={run:sha(P/f"reports/checks/{run}.json") for _,run in STAGES},
        environment={p:importlib.metadata.version(p) for p in ["tabpfn","torch","skrub","numpy","pandas","scikit-learn"]})
    import json
    protocol=json.loads(json.dumps(protocol))
    if (report/"protocol.json").exists():
        assert read(report/"protocol.json")==protocol,"Frozen TabPFN study changed"
    else:
        _write_json(report/"protocol.json",protocol)
    for f,h in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/f)==h
    if (report/"summary.json").exists():
        print("Completed TabPFN study")
        return
    g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    stages={}
    for stage,run in STAGES:
        rows=[]
        for repeat in [1,2]:
            table=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            reference=table.alternative.to_numpy() if stage=="paired" else table.reference.to_numpy()
            probability=np.full(len(old.y),np.nan)
            for fold in range(1,6):
                stem=f"fresh_repeat_{repeat}_fold_{fold}" if stage=="paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                source=P/"models"/run/f"{stem}.joblib"
                assert sha(source)==read(EXPERIMENTS_DIR/run/f"{stem}.json")["model_sha256"]
                z=np.load(P/"outputs"/run/f"{stem}.npz")
                ti,vi=z["training"],z["validation"]
                assert (table.fold.iloc[vi]==fold).all()
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                mp,cache,rp=models/f"{tag}.joblib",out/f"{tag}.npz",report/f"{tag}.json"
                if rp.exists():
                    assert sha(mp)==read(rp)["model_sha256"]
                    saved=np.load(cache)
                    np.testing.assert_array_equal(saved["training"],ti)
                    np.testing.assert_array_equal(saved["validation"],vi)
                    probability[vi]=saved["probability"]
                    continue
                original=joblib.load(source)
                np.testing.assert_array_equal(original.training_y,old.y[ti])
                model=from_context(original)
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                probability[vi]=model.predict_proba(old.features.iloc[vi])[:,1]
                assert model._model.n_estimators_==4 and np.isfinite(probability[vi]).all()
                joblib.dump(model,mp,compress=3)
                np.savez_compressed(cache,training=ti,validation=vi,probability=probability[vi])
                _write_json(rp,dict(model_sha256=sha(mp),source_context_sha256=sha(source),source_stem=stem,
                    training_rows=len(ti),validation_rows=len(vi),columns=list(model.training_X),views=4,
                    seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                del original,model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"TabPFN {tag} completed",flush=True)
            alternative=(1-WEIGHT)*reference+WEIGHT*probability
            result=dict(repeat=repeat,**stats(old.y,reference,alternative,table.fold.to_numpy()),
                component_accuracy=float(((probability>=.5)==old.y).mean()),
                g=stats(old.y[g],reference[g],alternative[g],table.fold.to_numpy()[g]),
                non_g=stats(old.y[~g],reference[~g],alternative[~g],table.fold.to_numpy()[~g]))
            rows.append(result)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=table.fold,
                reference=reference,component=probability,alternative=alternative)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report/f"{stage}_results.json",rows)
            print(f"TabPFN {stage} repeat {repeat}: net {result['corrected']-result['new_errors']:+d}, {result['improved_folds']}/5 improved",flush=True)
            if result["gain"]<0 or result["log_loss"]>result["reference_log_loss"]+.002 or result["improved_folds"]==0:
                break
        stages[stage]=rows
        if not passes(rows):
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None))
            return
    _write_json(report/"summary.json",dict(status="qualified_for_production",stages=stages,kaggle_score=None))


if __name__=="__main__":
    main()
