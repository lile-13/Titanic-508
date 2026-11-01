"""Cross-fit teacher probabilities inside each outer training fold before name effects."""
import gc

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from scripts.audit_unlabeled_views import stats
from scripts.tabicl_finetune_study import sha,read,STAGES
from src import preprocess as pp
from src.name_residuals import NameResidual
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.tabicl_aggregation import fit_probability_mean
from src.train import _write_json,_write_csv

NAME="name_residuals_20261101"
INNER_SEED=111026


def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME,P/"models"/NAME
    for d in [report,out,models]:d.mkdir(parents=True,exist_ok=True)
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public
    protocol=dict(name=NAME,inner_seed=INNER_SEED,inner_group_folds=3,penalty=4.,strength=.5,threshold=.5,
        hypothesis="Compact TabICL does not use first/surnames. Add shared, strongly shrunk name effects on its residual log-odds instead of ordinal names or raw name fragments. No inferred gender labels.",
        training="Three group-isolated inner teachers predict all outer training rows. Fit a global intercept plus FirstName/Surname regularized effects to those probabilities and training labels. Matched control fits only the intercept, same penalty and strength. Unknown names contribute zero name effect.",
        candidate="Keep original tree25%, replace TabICL75% by fixed-strength named residual correction. Baseline is the actual .81318 platform algorithm. No global 1203 error list is read during fitting.",
        gate="Two repeat gains nonnegative vs incumbent and matched intercept control, mean incumbent gain>=.001, mean matched gain>=.0005, >=6/10 improved folds, loss deterioration<=.002. First negative repeat, negative matched gain, bad loss or zero improved folds stops. Both historical stages required for production.",
        caveat="New hypothesis after public feedback, using previously viewed passengers and partitions. Valid within-algorithm group isolation, but not a new independent blind sample. Prior raw-name/character feature failures remain failed.",
        platform_outcomes_sha256=sha(EXPERIMENTS_DIR/"manual_submission_batch_20261031/platform_outcomes_20261101.json"),
        sources={f:sha(P/f) for f in ["scripts/name_residual_study.py","src/name_residuals.py","src/tabicl_aggregation.py","src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/preprocess.py"]})
    if (report/"protocol.json").exists():assert read(report/"protocol.json")==protocol
    else:_write_json(report/"protocol.json",protocol)
    if (report/"summary.json").exists():print("Completed name residual study");return
    stages={};g=old.features.Deck.eq("G").fillna(False).to_numpy(bool)
    for stage,run in STAGES:
        rows=[]
        for repeat in [1,2]:
            table=pd.read_csv(EXPERIMENTS_DIR/run/f"fresh_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId)
            old_reference=table.alternative.to_numpy() if stage=="paired" else table.reference.to_numpy()
            base,named,control=[np.full(len(old.y),np.nan) for _ in range(3)]
            for fold in range(1,6):
                stem=f"fresh_repeat_{repeat}_fold_{fold}" if stage=="paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                source=P/"models"/run/f"{stem}.joblib"
                assert sha(source)==read(EXPERIMENTS_DIR/run/f"{stem}.json")["model_sha256"]
                z=np.load(P/"outputs"/run/f"{stem}.npz")
                ti,vi=z["training"],z["validation"]
                assert (table.fold.iloc[vi]==fold).all()
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                base[vi]=z["probability"]
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                teacher=np.full(len(ti),np.nan);inner_of=np.zeros(len(ti),int)
                for inner,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=INNER_SEED).split(ti,groups=old.features.GroupId.iloc[ti]),1):
                    name=f"{tag}_teacher_{inner}"
                    mp,cache,rp=models/f"{name}.joblib",out/f"{name}.npz",report/f"{name}.json"
                    if rp.exists():
                        assert sha(mp)==read(rp)["model_sha256"]
                        p=np.load(cache)["probability"]
                    else:
                        model=fit_probability_mean(old.features.iloc[ti[it]],old.y[ti[it]],public)
                        p=model.predict_proba(old.features.iloc[ti[iv]])[:,1]
                        joblib.dump(model,mp,compress=3)
                        np.savez_compressed(cache,training=ti[it],validation=ti[iv],probability=p)
                        _write_json(rp,dict(model_sha256=sha(mp)))
                        del model;gc.collect();torch.cuda.empty_cache()
                    teacher[iv]=p;inner_of[iv]=inner
                assert np.isfinite(teacher).all()
                contexts={}
                for enabled in [False,True]:
                    model=NameResidual(include_names=enabled).fit(old.features.iloc[ti],old.y[ti],teacher)
                    p=model.predict(old.features.iloc[vi],base[vi])
                    (named if enabled else control)[vi]=p
                    contexts["named" if enabled else "control"]=model
                mp=models/f"{tag}_corrections.joblib"
                joblib.dump(contexts,mp,compress=3)
                np.savez_compressed(out/f"{tag}_corrections.npz",training=ti,validation=vi,teacher=teacher,inner_fold=inner_of,named=named[vi],control=control[vi])
                _write_json(report/f"{tag}_corrections.json",dict(model_sha256=sha(mp),source_run=run,source_stem=stem,
                    source_context_sha256=sha(source),named_optimization=contexts["named"].optimization_,control_optimization=contexts["control"].optimization_))
                print(f"Name residual {tag} completed",flush=True)
            tree=2*old_reference-base
            reference=.25*tree+.75*base
            alternative=.25*tree+.75*named
            matched=.25*tree+.75*control
            result=dict(repeat=repeat,**stats(old.y,reference,alternative,table.fold.to_numpy()),
                control_accuracy=float(((matched>=.5)==old.y).mean()),
                gain_over_control=float(((alternative>=.5)==old.y).mean()-((matched>=.5)==old.y).mean()),
                g=stats(old.y[g],reference[g],alternative[g],table.fold.to_numpy()[g]))
            rows.append(result)
            _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,Transported=old.y,fold=table.fold,reference=reference,base_tfm=base,named_tfm=named,control_tfm=control,alternative=alternative,control=matched)),report/f"{stage}_repeat_{repeat}_predictions.csv")
            _write_json(report/f"{stage}_results.json",rows)
            print(f"Name correction {stage} repeat{repeat}: net {result['corrected']-result['new_errors']:+d}",flush=True)
            if result["gain"]<0 or result["gain_over_control"]<0 or result["log_loss"]>result["reference_log_loss"]+.002 or result["improved_folds"]==0:break
        stages[stage]=rows
        passed=len(rows)==2 and all(r["gain"]>=0 and r["gain_over_control"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows) and np.mean([r["gain"] for r in rows])>=.001 and np.mean([r["gain_over_control"] for r in rows])>=.0005 and sum(r["improved_folds"] for r in rows)>=6
        if not passed:
            _write_json(report/"summary.json",dict(status="no_candidate",failed_stage=stage,stages=stages,kaggle_score=None));return
    _write_json(report/"summary.json",dict(status="qualified_for_production",stages=stages,kaggle_score=None))


if __name__=="__main__":main()
