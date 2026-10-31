"""Frozen predictor-only fine-tuning screen with nested group validation."""
import gc
import hashlib
import importlib
import importlib.metadata
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from scripts.audit_unlabeled_views import stats
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.tabicl_finetune_probe import PARAMS, ObservedFinetune, predictor
from src.tabicl_models import CHECKPOINT, CHECKPOINT_SHA256
from src.train import _write_json, _write_csv

NAME = "tabicl_finetune_20261031"
STAGES = [("paired", "tabicl_probability_mean_20261030"), ("additional", "tabm_20261031")]
INNER_SEED = 312026


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def passes(rows):
    return (len(rows)==2 and all(r["gain"]>=0 and r["gain_over_control"]>=0
            and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows)
        and np.mean([r["gain"] for r in rows])>=.001
        and np.mean([r["gain_over_control"] for r in rows])>=.0005
        and sum(r["improved_folds"] for r in rows)>=6)


def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old = SearchRunner()
    report, out, models = EXPERIMENTS_DIR/NAME, PROJECT_ROOT/"outputs"/NAME, PROJECT_ROOT/"models"/NAME
    for directory in [report,out,models]:
        directory.mkdir(parents=True,exist_ok=True)
    author = PROJECT_ROOT/"models/tabicl_pretrained"/CHECKPOINT
    assert sha(author)==CHECKPOINT_SHA256
    modules = ["tabicl._finetune.base", "tabicl._finetune.classifier", "tabicl._finetune.data", "tabicl.train._optim"]
    sources = ["scripts/tabicl_finetune_study.py", "src/tabicl_finetune_probe.py", "src/tabicl_models.py",
               "src/focused_features.py", "src/competition_features.py", "src/preprocess.py", "scripts/audit_unlabeled_views.py"]
    protocol = dict(name=NAME, params=PARAMS, checkpoint_sha256=CHECKPOINT_SHA256, inner_seed=INNER_SEED,
        inner_validation_group_fraction=.18, threshold=.5,
        combination="0.5 original trees + 0.25 original TabICL + 0.25 fine-tuned TabICL; matched control replaces the last term with original weights using identical inner-fit categorical encoder and full outer training context",
        selection="Only the ICL predictor parameters are updated. Column/row encoders frozen. Inner GroupShuffleSplit chooses checkpoint by log loss, including epoch zero. Then all outer training labels become inference context; inner validation rows are not used for gradient updates. Outer validation labels never choose epochs.",
        meta_batches="Author row-based context/query sampling occurs exclusively inside inner training groups. This differs from group-disjoint outer evaluation; no validation labels enter meta-batches.",
        gates="Each stage: two nonnegative gains versus incumbent and matched control; mean incumbent gain>=.001, mean matched gain>=.0005; >=6/10 improved folds; each log loss deterioration<=.002. Stop after a negative repeat or zero improved folds. Historical and additional historical stages must both pass before considering production.",
        caveat="Adaptive study on previously viewed labels and partitions, not new independent evidence. No weight, threshold, learning-rate, input or freezing search. A failed frozen head-only configuration does not rule out all fine-tuning.",
        stages=STAGES,
        reference_files={f"{run}/fresh_repeat_{r}_predictions.csv":sha(EXPERIMENTS_DIR/run/f"fresh_repeat_{r}_predictions.csv") for _,run in STAGES for r in [1,2]},
        reference_audits={run:sha(PROJECT_ROOT/f"reports/checks/{run}.json") for _,run in STAGES},
        sources={p:sha(PROJECT_ROOT/p) for p in sources},
        installed_sources={m:sha(Path(importlib.import_module(m).__file__)) for m in modules},
        data_hashes=read(old.report/"protocol.json")["data_hashes"],
        environment={p:importlib.metadata.version(p) for p in ["tabicl","torch","transformers","numpy","pandas","scikit-learn"]})
    # Canonicalize tuples before checking an existing JSON protocol.
    protocol = json.loads(json.dumps(protocol))
    if (report/"protocol.json").exists():
        assert read(report/"protocol.json")==protocol, "Frozen study changed"
    else:
        _write_json(report/"protocol.json",protocol)
    for p,digest in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/p)==digest
    if (report/"summary.json").exists():
        print("Study completed; audit rather than retrain")
        return
    stages = {}
    for stage,run in STAGES:
        rows=[]
        for repeat in [1,2]:
            table = pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(table.Transported,old.y)
            reference = table.alternative.to_numpy() if stage=="paired" else table.reference.to_numpy()
            fine,control,base_component = [np.full(len(old.y),np.nan) for _ in range(3)]
            for fold in range(1,6):
                source_stem=f"fresh_repeat_{repeat}_fold_{fold}" if stage=="paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                source = PROJECT_ROOT/"models"/run/f"{source_stem}.joblib"
                source_record=read(EXPERIMENTS_DIR/run/f"{source_stem}.json")
                assert sha(source)==source_record["model_sha256"]
                z=np.load(PROJECT_ROOT/"outputs"/run/f"{source_stem}.npz")
                ti,vi=z["training"],z["validation"]
                assert (table.fold.iloc[vi]==fold).all()
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                base_component[vi]=z["probability"]
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                fold_dir=models/tag
                record_path=report/f"{tag}.json"
                if record_path.exists():
                    record=read(record_path)
                    assert sha(fold_dir/"context.joblib")==record["context_sha256"]
                    assert sha(fold_dir/"best.ckpt")==record["checkpoint_sha256"]
                    cached=np.load(out/f"{tag}.npz")
                    np.testing.assert_array_equal(cached["training"],ti)
                    np.testing.assert_array_equal(cached["validation"],vi)
                    fine[vi],control[vi]=cached["fine"],cached["control"]
                    continue
                fold_dir.mkdir(parents=True,exist_ok=True)
                original=joblib.load(source)
                X=original.training_X.copy(deep=True)
                np.testing.assert_array_equal(original.training_y,old.y[ti])
                ii,iv=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=INNER_SEED).split(X,groups=old.features.GroupId.iloc[ti]))
                assert not set(old.features.GroupId.iloc[ti[ii]])&set(old.features.GroupId.iloc[ti[iv]])
                torch.manual_seed(42)
                np.random.seed(42)
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                trained=ObservedFinetune(**PARAMS,model_path=str(author))
                trained.fit(X.iloc[ii],old.y[ti[ii]],X_val=X.iloc[iv],y_val=old.y[ti[iv]],output_dir=fold_dir)
                context=dict(builder=original.builder,encoder=trained._X_encoder_,
                    encoded_X=np.asarray(trained._X_encoder_.transform(X),dtype=float),y=old.y[ti].copy())
                query=context["encoder"].transform(context["builder"].transform(old.features.iloc[vi]))
                tuned_model=predictor(fold_dir/"best.ckpt",context)
                fine[vi]=tuned_model.predict_proba(query)[:,1]
                control_model=predictor(author,context)
                control[vi]=control_model.predict_proba(query)[:,1]
                assert np.isfinite(fine[vi]).all() and np.isfinite(control[vi]).all()
                history=trained.validation_history_
                best_epoch=0
                for i,h in enumerate(history[1:],1):
                    if h["primary"]>history[best_epoch]["primary"]+PARAMS["min_delta"]:
                        best_epoch=i
                joblib.dump(context,fold_dir/"context.joblib",compress=3)
                np.savez_compressed(out/f"{tag}.npz",training=ti,validation=vi,inner_training=ti[ii],
                    inner_validation=ti[iv],fine=fine[vi],control=control[vi])
                _write_json(record_path,dict(source_context_sha256=sha(source),source_stem=source_stem,
                    context_sha256=sha(fold_dir/"context.joblib"),checkpoint_sha256=sha(fold_dir/"best.ckpt"),
                    history=history,selected_epoch=best_epoch,epochs_run=len(history)-1,
                    best_inner_log_loss=-history[best_epoch]["primary"],seconds=time.perf_counter()-start,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                print(f"Finetune {tag}: selected epoch {best_epoch}/{len(history)-1}",flush=True)
                del original,trained,tuned_model,control_model,context
                gc.collect()
                torch.cuda.empty_cache()
            # The original current-best algorithm is 0.5 trees + 0.5 old TFM.
            alternative=reference+.25*(fine-base_component)
            matched=reference+.25*(control-base_component)
            row=dict(repeat=repeat,**stats(old.y,reference,alternative,table.fold.to_numpy()),
                control_accuracy=float(((matched>=.5)==old.y).mean()),
                gain_over_control=float(((alternative>=.5)==old.y).mean()-((matched>=.5)==old.y).mean()))
            rows.append(row)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=table.fold,
                reference=reference,base_component=base_component,finetuned=fine,untuned_control=control,
                control=matched,alternative=alternative)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report/f"{stage}_results.json",rows)
            print(f"Finetune {stage} repeat {repeat}: net {row['corrected']-row['new_errors']:+d}, {row['improved_folds']}/5 improved",flush=True)
            if row["gain"]<0 or row["gain_over_control"]<0 or row["log_loss"]>row["reference_log_loss"]+.002 or row["improved_folds"]==0:
                break
        stages[stage]=rows
        if not passes(rows):
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None))
            return
    _write_json(report/"summary.json",dict(status="qualified_for_production",stages=stages,kaggle_score=None))


if __name__=="__main__":
    main()
