"""Reconstruct input controls, grouped sampling, and the per-view failure evidence."""
import gc
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from scripts.audit_unlabeled_views import stats,assert_stats
from scripts.diagnose_tabicl_failure import cohorts,summarize
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.tabicl_inputs import TabICLInputBuilder,VIEWS
from src.tabicl_models import PARAMS
from src.train import _write_json


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=SearchRunner();run="tabicl_input_audit_20261030";report=EXPERIMENTS_DIR/run;out=PROJECT_ROOT/"outputs"/run
    protocol=read(report/"protocol.json");summary=read(report/"summary.json")
    assert summary["status"]=="no_candidate" and summary["completed_repeats"]==0
    for path,digest in protocol["sources"].items():assert sha(PROJECT_ROOT/path)==digest,path
    for path,digest in protocol["data_hashes"].items():assert sha(pp.find_data_dir()/path)==digest
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    dev=old.fold_of!=5;predictions={};count=0;max_error=0.
    for view in VIEWS:
        for family in ["tfm","cat"]:
            if view=="compact" and family=="tfm":
                p=np.full(len(old.y),np.nan)
                for fold,(_,vi) in enumerate(old.splits,1):p[vi]=np.load(PROJECT_ROOT/f"outputs/tabicl_20261030/development_fold_{fold}.npz")["probability"]
                predictions[(family,view)]=p;continue
            p=np.full(len(old.y),np.nan)
            for fold,(ti,vi) in enumerate(old.splits,1):
                stem=f"{family}_{view}_development_fold_{fold}"
                z=np.load(out/f"{stem}.npz");np.testing.assert_array_equal(z["training"],ti);np.testing.assert_array_equal(z["validation"],vi)
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                modelpath=PROJECT_ROOT/"models"/run/f"{stem}.joblib";meta=read(report/f"{stem}.json")
                assert sha(modelpath)==meta["model_sha256"]
                m=joblib.load(modelpath);b=TabICLInputBuilder(view).fit(old.features.iloc[ti],public)
                expected=b.transform(old.features.iloc[ti]);X=b.transform(old.features.iloc[vi]);assert pp.TARGET not in X
                if family=="tfm":
                    assert m._model is None;pd.testing.assert_frame_equal(m.training_X,expected);np.testing.assert_array_equal(m.training_y,old.y[ti])
                    q=m.predict_proba(old.features.iloc[vi])[:,1]
                    for key,value in PARAMS.items():assert m._model.get_params()[key]==value
                else:
                    pd.testing.assert_frame_equal(m["builder"].transform(old.features.iloc[vi]),X)
                    for key,value in protocol["cat_params"].items():assert m["model"].get_params()[key]==value
                    assert m["model"].get_params()["random_seed"]==42
                    q=m["model"].predict_proba(m["encoder"].transform(X))[:,1]
                np.testing.assert_allclose(q,z["probability"],atol=2e-6 if family=="tfm" else 1e-14,rtol=0)
                np.testing.assert_array_equal(q>=.5,z["probability"]>=.5)
                max_error=max(max_error,float(np.max(np.abs(q-z["probability"]))));p[vi]=z["probability"];count+=1
                del m;gc.collect();torch.cuda.empty_cache()
                print(f"Audited input {stem}",flush=True)
            predictions[(family,view)]=p
    rows=read(report/"development_comparison.json");compact=None
    for row in rows:
        view,family=row["view"],row["family"]
        table=pd.read_csv(report/f"development_{family}_{view}.csv",dtype={"PassengerId":str},float_precision="round_trip")
        np.testing.assert_array_equal(table.PassengerId,old.raw.PassengerId);np.testing.assert_array_equal(table.Transported,old.y)
        p=predictions[(family,view)];ref=table.reference.to_numpy();mixed=.5*ref+.5*p
        np.testing.assert_allclose(p,table.component,atol=1e-14,rtol=0,equal_nan=True)
        np.testing.assert_allclose(mixed,table.alternative,atol=1e-14,rtol=0,equal_nan=True)
        actual=stats(old.y[dev],ref[dev],mixed[dev],old.fold_of[dev]);assert_stats(actual,row)
        if view=="compact" and family=="tfm":compact=actual["accuracy"]
    eligible=[r for r in rows if r["family"]=="tfm" and r["view"]!="compact" and r["gain"]>=.001 and r["improved_folds"]>=3
              and r["log_loss"]<=r["reference_log_loss"]+.002 and r["accuracy"]>=compact+.0005]
    assert not eligible
    diag=EXPERIMENTS_DIR/"tabicl_failure_diagnosis_20261030";dp=read(diag/"protocol.json")
    for path,digest in dp["sources"].items():assert sha(PROJECT_ROOT/path)==digest
    base=np.load(PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029/repeat_2_fold_3.npz");ti,vi=base["training"],base["validation"]
    source=pd.read_csv(EXPERIMENTS_DIR/"tabicl_paired_20261030/repeat_2_predictions.csv",float_precision="round_trip")
    groups=np.unique(old.features.GroupId.iloc[ti]);curve=pd.read_csv(diag/"fixed_validation_learning_curve.csv",float_precision="round_trip")
    diagnostic_models=0
    for seed in dp["seeds"]:
        order=np.random.default_rng(seed).permutation(groups)
        for fraction in dp["fractions"]:
            selected=set(order[:int(np.floor(len(groups)*fraction))]);fit=ti[old.features.GroupId.iloc[ti].isin(selected).to_numpy()]
            stem=f"seed_{seed}_fraction_{fraction}";modelpath=PROJECT_ROOT/"models"/diag.name/f"{stem}.joblib"
            z=np.load(PROJECT_ROOT/"outputs"/diag.name/f"{stem}.npz")
            np.testing.assert_array_equal(z["training"],fit);np.testing.assert_array_equal(z["validation"],vi)
            assert sha(modelpath)==read(diag/f"{stem}.json")["model_sha256"]
            assert not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[vi])
            m=joblib.load(modelpath);np.testing.assert_array_equal(m.training_y,old.y[fit])
            pd.testing.assert_frame_equal(m.training_X,m.builder.transform(old.features.iloc[fit]))
            q=m.predict_proba(old.features.iloc[vi])[:,1];np.testing.assert_allclose(q,z["probability"],atol=2e-6,rtol=0)
            mixed=.5*source.reference.to_numpy()[vi]+.5*z["probability"]
            for name,mask in cohorts(old.features.iloc[vi]).items():
                expected=summarize(old.y[vi],source.reference.to_numpy()[vi],mixed,mask)
                row=curve[(curve.seed==seed)&(curve.fraction==fraction)&(curve.cohort==name)].iloc[0]
                for key,value in expected.items():np.testing.assert_allclose(row[key],value,atol=1e-13,rtol=0)
            diagnostic_models+=1;del m;gc.collect();torch.cuda.empty_cache();print(f"Audited diagnostic {stem}",flush=True)
    summaries=read(diag/"per_view_summary.json");views=pd.read_csv(diag/"per_view_cold_G.csv",float_precision="round_trip")
    cold=cohorts(old.features.iloc[vi])["G_cold"]
    for summary in summaries:
        z=np.load(PROJECT_ROOT/"outputs"/diag.name/f"views_{summary['context']}.npz")
        np.testing.assert_array_equal(z["validation"],vi)
        p=softmax(z["logits"].mean(axis=0)/.9,axis=-1)[:,1]
        probabilities=softmax(z["logits"]/.9,axis=-1)[:,:,1]
        np.testing.assert_allclose(z["logit_mean_probability"],p,atol=1e-6,rtol=0)
        np.testing.assert_allclose(z["probabilities"],probabilities,atol=1e-6,rtol=0)
        assert probabilities.shape[0]==8
        for index,q in enumerate(probabilities):
            row=views[(views.context==summary["context"])&(views.view==index)].iloc[0]
            np.testing.assert_allclose(row.cold_G_mean_probability,q[cold].mean(),atol=1e-6,rtol=0)
        np.testing.assert_allclose(summary["cold_G_mean_probability"],p[cold].mean(),atol=1e-6,rtol=0)
        assert summary["historical_prediction_reproduced"]
    _write_json(PROJECT_ROOT/"reports/checks/tabicl_input_followup_20261030.json",dict(passed=True,input_models_reloaded=count,
                learning_curve_models_reloaded=diagnostic_models,source_and_data_hashes_match=True,group_isolation_verified=True,
                input_gate_recomputed=True,per_view_aggregation_reconstructed=True,all_eight_views_retained=True,
                maximum_input_reload_error=max_error,selected_fold_diagnosis_not_independent=True,
                test_suite="outputs/tabicl_input_followup_20261030_tests.log",no_input_candidate=True))
    print("Input and failure diagnosis audit passed",flush=True)


if __name__=="__main__":main()
