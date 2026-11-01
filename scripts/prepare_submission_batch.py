"""User-requested exploratory CSVs, separate from formal model promotion."""
import copy
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import stats
from scripts.tabicl_finetune_study import sha,read,STAGES
from src import preprocess as pp
from src.optimization import SearchRunner,predict_ensemble
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.tabpfn_models import from_context,CHECKPOINT,CHECKPOINT_SHA256
from src.train import _write_csv,_write_json,make_submission

NAME="manual_submission_batch_20261031"
VARIANTS={
    "pfn25":dict(best=.75,pfn=.25,tfm=0.,description="75% current best + 25% TabPFN V2",order=1),
    "pfn50":dict(best=.5,pfn=.5,tfm=0.,description="50% current best + 50% TabPFN V2",order=2),
    "pfn100":dict(best=0.,pfn=1.,tfm=0.,description="TabPFN V2 only, five grouped contexts",order=4),
    "tabicl75":dict(best=.5,pfn=0.,tfm=.5,description="25% original trees + 75% arithmetic TabICL",order=3),
}


def mix(spec,best,pfn,tfm):
    return spec["best"]*best+spec["pfn"]*pfn+spec["tfm"]*tfm


def local_comparisons(old,report):
    rows=[]
    g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    for stage,run in STAGES:
        for repeat in [1,2]:
            t=pd.read_csv(EXPERIMENTS_DIR/f"tabpfn_v2_20261031/{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(t.PassengerId,old.raw.PassengerId)
            tfm=np.full(len(old.y),np.nan)
            for fold in range(1,6):
                stem=f"fresh_repeat_{repeat}_fold_{fold}" if stage=="paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                z=np.load(P/"outputs"/run/f"{stem}.npz")
                assert (t.fold.iloc[z["validation"]]==fold).all()
                tfm[z["validation"]]=z["probability"]
            for variant,spec in VARIANTS.items():
                q=mix(spec,t.reference.to_numpy(),t.component.to_numpy(),tfm)
                row=dict(variant=variant,stage=stage,repeat=repeat,**stats(old.y,t.reference.to_numpy(),q,t.fold.to_numpy()),
                    g_accuracy=float(((q[g]>=.5)==old.y[g]).mean()))
                rows.append(row)
    _write_json(report/"historical_local_comparisons.json",dict(rows=rows,
        caveat="Exploratory versions fixed before computing these extra weights' scores, after viewing earlier failures. All four repetitions use the same passengers. Not independent, not a Kaggle prediction and not formal promotion."))
    _write_csv(pd.DataFrame([{k:v for k,v in r.items() if k!="fold_gains"} for r in rows]),report/"historical_local_comparisons.csv")
    return rows


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME,P/"models"/NAME
    ready=P/"outputs/kaggle_ready_4690/next_batch_20261031"
    for d in [report,out,models,ready]:
        d.mkdir(parents=True,exist_ok=True)
    best_path=P/"outputs/kaggle_ready_4690/submission_tabicl_mean4690.csv"
    assert sha(best_path)=="bd2858e0412f37d22ac38040401bfbe3e459a5ab5690f0e4b98924c78aed8ac0"
    checkpoint=P/"models/tabpfn_pretrained"/CHECKPOINT
    assert sha(checkpoint)==CHECKPOINT_SHA256
    protocol=dict(name=NAME,variants=VARIANTS,threshold=.5,
        authorization="User explicitly requested multiple CSVs for manual Kaggle comparison, including experimental directions. This separate batch does not override any failed local study gate or promote its models.",
        parent_tabpfn_summary_sha256=sha(EXPERIMENTS_DIR/"tabpfn_v2_20261031/summary.json"),
        parent_tabpfn_protocol_sha256=sha(EXPERIMENTS_DIR/"tabpfn_v2_20261031/protocol.json"),
        original_best_sha256=sha(best_path),checkpoint_sha256=CHECKPOINT_SHA256,
        data_hashes=read(old.report/"protocol.json")["data_hashes"],
        sources={p:sha(P/p) for p in ["scripts/prepare_submission_batch.py","src/tabpfn_models.py","src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/preprocess.py"]},
        caveat="Four historical-comparison variants for user-directed public evaluation, not four new independent models. No submission is sent and no Kaggle score inferred. Refine directions using local evidence as well as reported platform feedback.")
    if (report/"protocol.json").exists():
        assert read(report/"protocol.json")==protocol,"Frozen manual batch changed"
    else:
        _write_json(report/"protocol.json",protocol)
    for f,h in protocol["data_hashes"].items():
        assert sha(pp.find_data_dir()/f)==h
    comparisons=local_comparisons(old,report)
    tree,_=predict_ensemble(P/"models/optimization_20261025/ensemble.joblib")
    pfn,tfm=np.zeros(len(old.raw_test)),np.zeros(len(old.raw_test))
    context_records=[]
    for fold in range(1,6):
        source=P/f"models/tabicl_probability_mean_20261030/final_fold_{fold}.joblib"
        assert sha(source)==read(EXPERIMENTS_DIR/f"tabicl_probability_mean_20261030/final_fold_{fold}.json")["model_sha256"]
        mp,cache,rp=models/f"pfn_fold_{fold}.joblib",out/f"fold_{fold}.npz",report/f"fold_{fold}.json"
        if rp.exists():
            record=read(rp)
            assert sha(mp)==record["model_sha256"] and sha(source)==record["source_context_sha256"]
            z=np.load(cache)
            a,b=z["pfn"],z["tfm"]
        else:
            start=time.perf_counter()
            original=joblib.load(source)
            model=from_context(original)
            a=model.predict_proba(old.test_features)[:,1]
            b=original.predict_proba(old.test_features)[:,1]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            joblib.dump(model,mp,compress=3)
            np.savez_compressed(cache,pfn=a,tfm=b)
            record=dict(fold=fold,source_context_sha256=sha(source),model_sha256=sha(mp),seconds=time.perf_counter()-start,
                rows=len(a),original_training_rows=len(original.training_y))
            _write_json(rp,record)
            del original,model
            gc.collect()
            torch.cuda.empty_cache()
        pfn+=a/5
        tfm+=b/5
        context_records.append(record)
        print(f"Submission batch test inference fold {fold} completed",flush=True)
    best=.5*tree+.5*tfm
    saved_best=pd.read_csv(P/"outputs/tabicl_probability_mean_20261030/test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(saved_best.PassengerId,old.raw_test.PassengerId)
    np.testing.assert_allclose(best,saved_best.probability,rtol=0,atol=2e-6)
    incumbent=pd.read_csv(best_path,dtype={"PassengerId":str})
    np.testing.assert_array_equal(best>=.5,incumbent.Transported)
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,tree=tree,tfm=tfm,best=best,pfn=pfn)),out/"test_components.csv")
    labels={}
    candidates=[]
    template=pp.load_submission_template()
    for variant,spec in VARIANTS.items():
        probability=mix(spec,best,pfn,tfm)
        submission=make_submission(old.raw_test.PassengerId,probability,template,.5)
        path=ready/f"submission_{variant}_4690.csv"
        if path.exists():
            prior=pd.read_csv(path,dtype={"PassengerId":str})
            pd.testing.assert_frame_equal(prior,submission)
        else:
            _write_csv(submission,path)
        labels[variant]=probability>=.5
        counts=[r for r in comparisons if r["variant"]==variant]
        candidates.append(dict(id=f"{NAME}_{variant}",model_run=f"{NAME}_{variant}",variant=variant,
            status="prepared_not_submitted",prepared_on="2026-10-31",file=path.relative_to(P).as_posix(),
            sha256=sha(path),rows=len(submission),score=None,kaggle_score=None,
            description=spec["description"]+", threshold 0.5",suggested_order=spec["order"],
            changes_vs_current_best=int((labels[variant]!=incumbent.Transported.to_numpy()).sum()),
            predicted_true=int(labels[variant].sum()),
            mean_historical_repeat_accuracy=float(np.mean([r["accuracy"] for r in counts])),
            historical_repeat_gains=[r["gain"] for r in counts],
            exploratory=True,formal_promotion=False,source_protocol_sha256=sha(report/"protocol.json"),
            audit=f"reports/checks/{NAME}.json"))
    for i,a in enumerate(labels):
        assert (labels[a]!=incumbent.Transported.to_numpy()).any(),f"{a} duplicates current best"
        for b in list(labels)[i+1:]:
            assert (labels[a]!=labels[b]).any(),f"Duplicate candidates {a}/{b}"
    _write_csv(pd.DataFrame([{**c,"historical_repeat_gains":str(c["historical_repeat_gains"])} for c in candidates]),report/"candidate_comparison.csv")
    summary=dict(status="exploratory_batch_prepared",candidates=candidates,contexts=context_records,
        pairwise_label_differences={f"{a}/{b}":int((labels[a]!=labels[b]).sum()) for i,a in enumerate(labels) for b in list(labels)[i+1:]},
        current_platform_best=.81178,submitted_to_kaggle=False,
        caveat="Candidates intentionally produced at user's request despite failed local stability gates; none replaces the observed platform best. Historical scores are descriptive and may not track Kaggle ordering.")
    _write_json(report/"summary.json",summary)
    _write_json(ready/"manifest.json",summary)
    # Registration occurs only after independent audit; no scores are written here.
    guide=["建议按顺序逐份手动提交，之后把文件名和分数发给我。", "当前已确认最佳为submission_tabicl_mean4690.csv：0.81178。", "以下均为探索候选，本地分数不是Kaggle分数。",""]
    for c in sorted(candidates,key=lambda x:x["suggested_order"]):
        guide.append(f"{c['suggested_order']}. {c['file'].split('/')[-1]} | {c['description']} | 与当前最佳不同{c['changes_vs_current_best']}人 | Kaggle分数：待填")
    (ready/"提交顺序和成绩记录.txt").write_text("\n".join(guide)+"\n",encoding="utf-8-sig")
    print(json.dumps(candidates,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
