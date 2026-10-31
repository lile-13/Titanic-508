"""Fixed input-alignment screen using the five already trained checkpoints."""
import gc
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import stats
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P, EXPERIMENTS_DIR
from src.tabicl_models import make_model
from src.train import _write_json, _write_csv

NAME="finetune_aligned_screen_20261031"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME
    report.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    parent=EXPERIMENTS_DIR/"tabicl_finetune_20261031"
    protocol=dict(name=NAME,change="Reuse exactly the five selected head-only checkpoints. At final inference, fit the ordinary TabICL input encoder on the full outer training context, as in the incumbent. Inner checkpoint selection remains untouched.",
        mixture="current best + .25*(aligned fine-tuned component - original component)",threshold=.5,
        screen_gate="First historical repeat: gain>=.001, >=3/5 improved folds, G and non-G nonnegative; loss deterioration<=.002. Passing would qualify only for further validation, never immediate production.",
        selection_history="Alignment hypothesis follows the failed original screen and input-only diagnostic; all historical labels have been viewed. No new independent holdout or new gradient training.",
        parent_protocol_sha256=sha(parent/"protocol.json"),parent_summary_sha256=sha(parent/"summary.json"),
        alignment_diagnostic_sha256=sha(parent/"encoding_diagnostic.json"),
        references_sha256=sha(EXPERIMENTS_DIR/"tabicl_probability_mean_20261030/fresh_repeat_1_predictions.csv"),
        sources={p:sha(P/p) for p in ["scripts/screen_aligned_finetune.py","src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/preprocess.py"]},
        checkpoints={str(f):sha(P/f"models/tabicl_finetune_20261031/paired_repeat_1_fold_{f}/best.ckpt") for f in range(1,6)})
    if (report/"protocol.json").exists():
        assert read(report/"protocol.json")==protocol
    else:
        _write_json(report/"protocol.json",protocol)
    if (report/"summary.json").exists():
        print("Completed alignment screen")
        return
    table=pd.read_csv(EXPERIMENTS_DIR/"tabicl_probability_mean_20261030/fresh_repeat_1_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
    np.testing.assert_array_equal(table.Transported,old.y)
    tuned,control,base=[np.full(len(old.y),np.nan) for _ in range(3)]
    folds=[]
    for fold in range(1,6):
        stem=f"fresh_repeat_1_fold_{fold}"
        context_path=P/f"models/tabicl_probability_mean_20261030/{stem}.joblib"
        context=joblib.load(context_path)
        z=np.load(P/f"outputs/tabicl_probability_mean_20261030/{stem}.npz")
        ti,vi=z["training"],z["validation"]
        assert (table.fold.iloc[vi]==fold).all()
        assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
        np.testing.assert_array_equal(context.training_y,old.y[ti])
        X=context.training_X
        query=context.builder.transform(old.features.iloc[vi])
        path=P/f"models/tabicl_finetune_20261031/paired_repeat_1_fold_{fold}/best.ckpt"
        for checkpoint,dest in [(None,control),(path,tuned)]:
            model=make_model().set_params(average_logits=False)
            if checkpoint is not None:
                model.set_params(model_path=str(checkpoint))
            model.fit(X,old.y[ti])
            p=model.predict_proba(query)[:,1]
            np.testing.assert_allclose(p,model.predict_proba(query)[:,1],rtol=0,atol=2e-6)
            dest[vi]=p
            del model
            gc.collect()
            torch.cuda.empty_cache()
        base[vi]=z["probability"]
        np.testing.assert_allclose(control[vi],base[vi],rtol=0,atol=2e-6)
        folds.append(dict(fold=fold,context_sha256=sha(context_path),checkpoint_sha256=sha(path),
            max_control_error=float(np.abs(control[vi]-base[vi]).max())))
        np.savez_compressed(out/f"fold_{fold}.npz",training=ti,validation=vi,tuned=tuned[vi],control=control[vi])
        print(f"Aligned fine-tuning inference fold {fold} completed",flush=True)
    alternative=table.alternative.to_numpy()+.25*(tuned-base)
    g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    reference=table.alternative.to_numpy()
    results=stats(old.y,reference,alternative,table.fold.to_numpy())
    results["g"]=stats(old.y[g],reference[g],alternative[g],table.fold.to_numpy()[g])
    results["non_g"]=stats(old.y[~g],reference[~g],alternative[~g],table.fold.to_numpy()[~g])
    passed=(results["gain"]>=.001 and results["improved_folds"]>=3
        and results["g"]["gain"]>=0 and results["non_g"]["gain"]>=0
        and results["log_loss"]<=results["reference_log_loss"]+.002)
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=table.fold,
        reference=reference,original_component=base,control_component=control,tuned_component=tuned,alternative=alternative)),report/"predictions.csv")
    _write_json(report/"summary.json",dict(status="requires_additional_validation" if passed else "no_candidate",
        screen_passed=bool(passed),results=results,folds=folds,new_gradient_fits=0,kaggle_score=None,
        caveat="Post-diagnosis historical screen only; no new submission or production model."))
    print(results,flush=True)


if __name__=="__main__":
    main()
