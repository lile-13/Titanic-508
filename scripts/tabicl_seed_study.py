"""A fixed 42/2026/103 seed average, compared against the actual .81318 winner."""
import gc

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import stats
from scripts.tabicl_finetune_study import sha,read,STAGES
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.tabicl_seed_models import from_context
from src.train import _write_json,_write_csv

NAME="tabicl_seeds_20261101"
SEEDS=[42,2026,103]


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME,P/"models"/NAME
    for d in [report,out,models]:d.mkdir(parents=True,exist_ok=True)
    outcomes=EXPERIMENTS_DIR/"manual_submission_batch_20261031/platform_outcomes_20261101.json"
    protocol=dict(name=NAME,seeds=SEEDS,views_per_seed=8,average="arithmetic probabilities within and across seeds",
        baseline="Actual platform-best algorithm: .25 original trees + .75 seed42 arithmetic TabICL",
        candidate=".25 same original trees + .75 mean(seed42,seed2026,seed103). Same checkpoint, raw inputs and training contexts. All 24 views retained, no view selection.",
        gate="Both repeat gains>=0, mean gain>=.001, >=6/10 improved folds, loss deterioration<=.002. First negative repeat, bad loss or zero improved folds stops. Both historical stages required for formal production.",
        history="Different fixed eight-view shuffles, not a rerun of the failed 32-view generator. Seeds match the long-established tree seed set; no seed search. Previously inspected passengers and partitions, not independent blind evidence.",
        platform_outcomes_sha256=sha(outcomes),
        sources={f:sha(P/f) for f in ["scripts/tabicl_seed_study.py","src/tabicl_seed_models.py","src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/preprocess.py"]})
    if (report/"protocol.json").exists():assert read(report/"protocol.json")==protocol
    else:_write_json(report/"protocol.json",protocol)
    if (report/"summary.json").exists():print("Completed seed study");return
    stages={}
    g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    for stage,run in STAGES:
        rows=[]
        for repeat in [1,2]:
            table=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            old_reference=table.alternative.to_numpy() if stage=="paired" else table.reference.to_numpy()
            probabilities={s:np.full(len(old.y),np.nan) for s in SEEDS}
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
                probabilities[42][vi]=z["probability"]
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                for seed in [2026,103]:
                    mp,cache,rp=models/f"{tag}_{seed}.joblib",out/f"{tag}_{seed}.npz",report/f"{tag}_{seed}.json"
                    if rp.exists():
                        assert sha(mp)==read(rp)["model_sha256"]
                        p=np.load(cache)["probability"]
                    else:
                        model=from_context(original,seed)
                        p=model.predict_proba(old.features.iloc[vi])[:,1]
                        assert model._model.get_params()["n_estimators"]==8
                        joblib.dump(model,mp,compress=3)
                        np.savez_compressed(cache,training=ti,validation=vi,probability=p)
                        _write_json(rp,dict(seed=seed,model_sha256=sha(mp),source_run=run,source_stem=stem,
                            source_context_sha256=sha(source)))
                        del model
                        gc.collect();torch.cuda.empty_cache()
                    probabilities[seed][vi]=p
                print(f"Seed contexts {tag} completed",flush=True)
            tree=2*old_reference-probabilities[42]
            reference=.25*tree+.75*probabilities[42]
            bag=np.mean(np.stack([probabilities[s] for s in SEEDS]),axis=0)
            alternative=.25*tree+.75*bag
            result=dict(repeat=repeat,**stats(old.y,reference,alternative,table.fold.to_numpy()),
                g=stats(old.y[g],reference[g],alternative[g],table.fold.to_numpy()[g]))
            rows.append(result)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=table.fold,
                reference=reference,tree=tree,**{f"seed{s}":probabilities[s] for s in SEEDS},alternative=alternative)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report/f"{stage}_results.json",rows)
            print(f"Seed bag {stage} repeat{repeat}: net {result['corrected']-result['new_errors']:+d}",flush=True)
            if result["gain"]<0 or result["log_loss"]>result["reference_log_loss"]+.002 or result["improved_folds"]==0:break
        stages[stage]=rows
        passed=len(rows)==2 and all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows) and np.mean([r["gain"] for r in rows])>=.001 and sum(r["improved_folds"] for r in rows)>=6
        if not passed:
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None));return
    _write_json(report/"summary.json",dict(status="qualified_for_production",stages=stages,kaggle_score=None))


if __name__=="__main__":main()
