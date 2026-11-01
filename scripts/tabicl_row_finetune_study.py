"""Row-representation adaptation with incumbent-aligned final preprocessing."""
import gc
import importlib
import importlib.metadata
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from scripts.audit_unlabeled_views import stats
from scripts.tabicl_finetune_study import sha, read, STAGES, INNER_SEED
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P, EXPERIMENTS_DIR
from src.tabicl_finetune_probe import PARAMS, ObservedFinetune
from src.tabicl_models import CHECKPOINT, CHECKPOINT_SHA256, make_model
from src.train import _write_json, _write_csv

NAME="tabicl_row_adaptation_20261031"
ROW_PARAMS={**PARAMS,"freeze_row":False}


def passes(rows):
    return (len(rows)==2 and all(r["gain"]>=0 and r["g"]["gain"]>=0 and r["non_g"]["gain"]>=0
        and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows)
        and np.mean([r["gain"] for r in rows])>=.001
        and sum(r["improved_folds"] for r in rows)>=6)


def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME,P/"models"/NAME
    for path in [report,out,models]:
        path.mkdir(parents=True,exist_ok=True)
    author=P/"models/tabicl_pretrained"/CHECKPOINT
    assert sha(author)==CHECKPOINT_SHA256
    source_files=["scripts/tabicl_row_finetune_study.py","scripts/tabicl_finetune_study.py","src/tabicl_finetune_probe.py",
        "src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/preprocess.py","scripts/audit_unlabeled_views.py"]
    modules=["tabicl._finetune.base","tabicl._finetune.classifier","tabicl._finetune.data","tabicl.train._optim"]
    protocol=dict(name=NAME,params=ROW_PARAMS,inner_seed=INNER_SEED,stages=STAGES,
        actual_platform_target="Accuracy strictly above .82 on Kaggle; local scores are separate and cannot certify this target",
        adaptation="Freeze only column embedding; row interaction and ICL predictor receive gradients. Fixed LR1e-5, max30 epochs, patience5, inner18% grouped split, loss selection including original epoch0.",
        final_encoding="After inner group-isolated weight selection, fit the ordinary TabICL classifier on raw compact18 outer-training inputs. Reproduce the incumbent preprocessing and compare original weights on exactly the same inputs. Inner validation statistics are never fit before weight selection.",
        combination=".5 original trees + .25 original arithmetic8 TabICL + .25 row-adapted arithmetic8 TabICL. Fixed threshold .5; no coefficient or threshold search.",
        gate="Each historical stage needs both repeat gains>=0, mean gain>=.001, >=6/10 improved folds, G and non-G nonnegative each repeat, each loss deterioration<=.002. First negative overall/G/non-G repeat, bad loss or zero improved folds stops. Both stages required before production.",
        caveat="All evaluation passengers and partitions were previously viewed. Hypothesis follows failed head-only adaptation and input-alignment diagnosis. This is adaptive evidence, not an independent blind holdout; no global error-ID-based fitting or routing.",
        sources={f:sha(P/f) for f in source_files},
        installed_sources={m:sha(Path(importlib.import_module(m).__file__)) for m in modules},
        data_hashes=read(old.report/"protocol.json")["data_hashes"],checkpoint_sha256=CHECKPOINT_SHA256,
        references={f"{run}/fresh_repeat_{r}_predictions.csv":sha(EXPERIMENTS_DIR/run/f"fresh_repeat_{r}_predictions.csv") for _,run in STAGES for r in [1,2]},
        reference_audits={run:sha(P/f"reports/checks/{run}.json") for _,run in STAGES},
        environment={p:importlib.metadata.version(p) for p in ["tabicl","torch","transformers","numpy","pandas","scikit-learn"]})
    import json
    protocol=json.loads(json.dumps(protocol))
    if (report/"protocol.json").exists():
        assert read(report/"protocol.json")==protocol,"Frozen row-adaptation protocol changed"
    else:
        _write_json(report/"protocol.json",protocol)
    for f,h in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/f)==h
    if (report/"summary.json").exists():
        print("Completed row-adaptation study")
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
            tuned,control,base=[np.full(len(old.y),np.nan) for _ in range(3)]
            for fold in range(1,6):
                stem=f"fresh_repeat_{repeat}_fold_{fold}" if stage=="paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                source=P/"models"/run/f"{stem}.joblib"
                assert sha(source)==read(EXPERIMENTS_DIR/run/f"{stem}.json")["model_sha256"]
                original=joblib.load(source)
                z=np.load(P/"outputs"/run/f"{stem}.npz")
                ti,vi=z["training"],z["validation"]
                assert (table.fold.iloc[vi]==fold).all()
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                np.testing.assert_array_equal(original.training_y,old.y[ti])
                base[vi]=z["probability"]
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                folder=models/tag
                rp=report/f"{tag}.json"
                if rp.exists():
                    record=read(rp)
                    assert sha(folder/"context.joblib")==record["context_sha256"]
                    assert sha(folder/"best.ckpt")==record["checkpoint_sha256"]
                    saved=np.load(out/f"{tag}.npz")
                    np.testing.assert_array_equal(saved["training"],ti)
                    np.testing.assert_array_equal(saved["validation"],vi)
                    tuned[vi],control[vi]=saved["tuned"],saved["control"]
                    continue
                folder.mkdir(parents=True,exist_ok=True)
                X=original.training_X.copy(deep=True)
                ii,iv=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=INNER_SEED).split(X,groups=old.features.GroupId.iloc[ti]))
                assert not set(old.features.GroupId.iloc[ti[ii]])&set(old.features.GroupId.iloc[ti[iv]])
                torch.manual_seed(42)
                np.random.seed(42)
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                trained=ObservedFinetune(**ROW_PARAMS,model_path=str(author))
                trained.fit(X.iloc[ii],old.y[ti[ii]],X_val=X.iloc[iv],y_val=old.y[ti[iv]],output_dir=folder)
                history=trained.validation_history_
                selected=0
                for i,h in enumerate(history[1:],1):
                    if h["primary"]>history[selected]["primary"]+ROW_PARAMS["min_delta"]:
                        selected=i
                context=dict(builder=original.builder,training_X=X,training_y=old.y[ti].copy(),inner_encoder=trained._X_encoder_)
                joblib.dump(context,folder/"context.joblib",compress=3)
                del trained
                gc.collect()
                torch.cuda.empty_cache()
                query=original.builder.transform(old.features.iloc[vi])
                for checkpoint,dest in [(author,control),(folder/"best.ckpt",tuned)]:
                    model=make_model().set_params(average_logits=False,model_path=str(checkpoint))
                    model.fit(X,old.y[ti])
                    dest[vi]=model.predict_proba(query)[:,1]
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                np.testing.assert_allclose(control[vi],base[vi],rtol=0,atol=2e-6)
                np.savez_compressed(out/f"{tag}.npz",training=ti,validation=vi,inner_training=ti[ii],inner_validation=ti[iv],tuned=tuned[vi],control=control[vi])
                _write_json(rp,dict(source_context_sha256=sha(source),source_stem=stem,
                    context_sha256=sha(folder/"context.joblib"),checkpoint_sha256=sha(folder/"best.ckpt"),
                    history=history,selected_epoch=selected,epochs_run=len(history)-1,
                    best_inner_log_loss=-history[selected]["primary"],seconds=time.perf_counter()-start,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                print(f"Row adaptation {tag}: epoch {selected}/{len(history)-1}",flush=True)
            alternative=reference+.25*(tuned-base)
            result=dict(repeat=repeat,**stats(old.y,reference,alternative,table.fold.to_numpy()),
                g=stats(old.y[g],reference[g],alternative[g],table.fold.to_numpy()[g]),
                non_g=stats(old.y[~g],reference[~g],alternative[~g],table.fold.to_numpy()[~g]))
            rows.append(result)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=table.fold,reference=reference,
                original_component=base,control_component=control,tuned_component=tuned,alternative=alternative)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report/f"{stage}_results.json",rows)
            print(f"Row adaptation {stage} repeat {repeat}: net {result['corrected']-result['new_errors']:+d}, G net {result['g']['corrected']-result['g']['new_errors']:+d}",flush=True)
            if result["gain"]<0 or result["g"]["gain"]<0 or result["non_g"]["gain"]<0 or result["log_loss"]>result["reference_log_loss"]+.002 or result["improved_folds"]==0:
                break
        stages[stage]=rows
        if not passes(rows):
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None))
            return
    _write_json(report/"summary.json",dict(status="qualified_for_production",stages=stages,kaggle_score=None))


if __name__=="__main__":
    main()
