"""Reconstruct new studies, observed-winner bundle and all four pending CSVs."""
import gc
import re

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from scripts.audit_unlabeled_views import stats,assert_stats,sha,read
from src import preprocess as pp
from src.name_residuals import NameResidual
from src.optimization import SearchRunner,predict_ensemble
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.tabicl_seed_models import from_context
from src.train import _write_json


def check_sources(protocol):
    for f,h in protocol["sources"].items():assert sha(P/f)==h


def audit_seeds(old):
    name="tabicl_seeds_20261101"
    report,out,models=EXPERIMENTS_DIR/name,P/"outputs"/name,P/"models"/name
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    check_sources(protocol)
    count,max_error=0,0.
    for stage,rows in summary["stages"].items():
        for row in rows:
            repeat=row["repeat"]
            t=pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(t.PassengerId,old.raw.PassengerId)
            np.testing.assert_array_equal(t.Transported,old.y)
            for fold in range(1,6):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                for seed in [2026,103]:
                    record=read(report/f"{tag}_{seed}.json")
                    mp=models/f"{tag}_{seed}.joblib"
                    assert sha(mp)==record["model_sha256"]
                    source=P/"models"/record["source_run"]/f"{record['source_stem']}.joblib"
                    assert sha(source)==record["source_context_sha256"]
                    a,b=joblib.load(mp),joblib.load(source)
                    pd.testing.assert_frame_equal(a.training_X,b.training_X)
                    np.testing.assert_array_equal(a.training_y,b.training_y)
                    assert a.seed==seed
                    z=np.load(out/f"{tag}_{seed}.npz")
                    ti,vi=z["training"],z["validation"]
                    assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                    assert (t.fold.iloc[vi]==fold).all()
                    np.testing.assert_array_equal(a.training_y,old.y[ti])
                    p=a.predict_proba(old.features.iloc[vi])[:,1]
                    np.testing.assert_allclose(p,z["probability"],rtol=0,atol=2e-6)
                    np.testing.assert_allclose(t[f"seed{seed}"].iloc[vi],z["probability"],rtol=0,atol=1e-14)
                    assert a._model.get_params()["n_estimators"]==8
                    if seed==2026:
                        original=from_context(b,42)
                        q=original.predict_proba(old.features.iloc[vi])[:,1]
                        np.testing.assert_allclose(q,t.seed42.iloc[vi],rtol=0,atol=2e-6)
                        del original
                    max_error=max(max_error,float(np.abs(p-z["probability"]).max()))
                    count+=1
                    del a,b;gc.collect();torch.cuda.empty_cache()
            np.testing.assert_allclose(t.reference,.25*t.tree+.75*t.seed42,rtol=0,atol=1e-14)
            np.testing.assert_allclose(t.alternative,.25*t.tree+.25*(t.seed42+t.seed2026+t.seed103),rtol=0,atol=1e-14)
            assert_stats(stats(old.y,t.reference.to_numpy(),t.alternative.to_numpy(),t.fold.to_numpy()),row)
    rows=summary["stages"][summary.get("failed_stage","additional")]
    gate=len(rows)==2 and all(r["gain"]>=0 and r["log_loss"]<=r["reference_log_loss"]+.002 for r in rows) and np.mean([r["gain"] for r in rows])>=.001 and sum(r["improved_folds"] for r in rows)>=6
    assert not gate and summary["status"]=="no_candidate"
    _write_json(P/f"reports/checks/{name}.json",dict(passed=True,gate_passed=False,new_seed_models_reloaded=count,
        original_seed42_controls_reproduced=True,max_reload_error=max_error,tests=66))


def audit_names(old):
    name="name_residuals_20261101"
    report,out,models=EXPERIMENTS_DIR/name,P/"outputs"/name,P/"models"/name
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    check_sources(protocol)
    count,corrections,max_error=0,0,0.
    for stage,rows in summary["stages"].items():
        for row in rows:
            repeat=row["repeat"]
            t=pd.read_csv(report/f"{stage}_repeat_{repeat}_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            np.testing.assert_array_equal(t.PassengerId,old.raw.PassengerId)
            for fold in range(1,6):
                tag=f"{stage}_repeat_{repeat}_fold_{fold}"
                record=read(report/f"{tag}_corrections.json")
                assert sha(models/f"{tag}_corrections.joblib")==record["model_sha256"]
                source=P/"models"/record["source_run"]/f"{record['source_stem']}.joblib"
                assert sha(source)==record["source_context_sha256"]
                z=np.load(out/f"{tag}_corrections.npz")
                ti,vi=z["training"],z["validation"]
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                assert (t.fold.iloc[vi]==fold).all()
                teacher=np.full(len(ti),np.nan)
                for inner,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=protocol["inner_seed"]).split(ti,groups=old.features.GroupId.iloc[ti]),1):
                    stem=f"{tag}_teacher_{inner}"
                    mp=models/f"{stem}.joblib"
                    assert sha(mp)==read(report/f"{stem}.json")["model_sha256"]
                    data=np.load(out/f"{stem}.npz")
                    np.testing.assert_array_equal(data["training"],ti[it]);np.testing.assert_array_equal(data["validation"],ti[iv])
                    assert not set(old.features.GroupId.iloc[ti[it]])&set(old.features.GroupId.iloc[ti[iv]])
                    model=joblib.load(mp)
                    np.testing.assert_array_equal(model.training_y,old.y[ti[it]])
                    p=model.predict_proba(old.features.iloc[ti[iv]])[:,1]
                    np.testing.assert_allclose(p,data["probability"],rtol=0,atol=2e-6)
                    teacher[iv]=data["probability"]
                    max_error=max(max_error,float(np.abs(p-data["probability"]).max()))
                    count+=1
                    del model;gc.collect();torch.cuda.empty_cache()
                np.testing.assert_array_equal(teacher,z["teacher"])
                contexts=joblib.load(models/f"{tag}_corrections.joblib")
                for enabled,key in [(False,"control"),(True,"named")]:
                    fitted=contexts[key]
                    fresh=NameResidual(include_names=enabled).fit(old.features.iloc[ti],old.y[ti],teacher)
                    np.testing.assert_allclose(fresh.parameters_,fitted.parameters_,rtol=0,atol=1e-12)
                    p=fitted.predict(old.features.iloc[vi],t.base_tfm.iloc[vi])
                    np.testing.assert_allclose(p,z[key],rtol=0,atol=1e-14)
                    np.testing.assert_allclose(t[f"{key}_tfm"].iloc[vi],p,rtol=0,atol=1e-14)
                    corrections+=1
            tree=(t.reference-.75*t.base_tfm)/.25
            np.testing.assert_allclose(t.alternative,.25*tree+.75*t.named_tfm,rtol=0,atol=1e-14)
            np.testing.assert_allclose(t.control,.25*tree+.75*t.control_tfm,rtol=0,atol=1e-14)
            assert_stats(stats(old.y,t.reference.to_numpy(),t.alternative.to_numpy(),t.fold.to_numpy()),row)
            gain=float(((t.alternative>=.5)==old.y).mean()-((t.control>=.5)==old.y).mean())
            np.testing.assert_allclose(gain,row["gain_over_control"],rtol=0,atol=1e-14)
    assert summary["status"]=="no_candidate"
    _write_json(P/f"reports/checks/{name}.json",dict(passed=True,gate_passed=False,teacher_contexts_reloaded=count,
        correction_models_reloaded=corrections,parameters_refitted=True,max_teacher_reload_error=max_error,tests=66))


def audit_candidates(old):
    name="next_directions_20261101"
    report=EXPERIMENTS_DIR/name
    protocol,summary=read(report/"protocol.json"),read(report/"summary.json")
    check_sources(protocol)
    assert sha(P/"outputs/manual_submission_batch_20261031/test_components.csv")==protocol["test_components_sha256"]
    model=P/summary["platform_best_model_bundle"]
    assert sha(model)==summary["platform_best_model_sha256"]
    assert sha(P/"models/tabicl_probability_mean_20261030/ensemble.joblib")==summary["source_bundle_sha256"]
    probability,reproduced=predict_ensemble(model)
    best=protocol["observed_platform_best"]
    actual=pd.read_csv(P/best["submission"],dtype={"PassengerId":str})
    pd.testing.assert_frame_equal(actual,reproduced)
    comp=pd.read_csv(P/"outputs/manual_submission_batch_20261031/test_components.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_allclose(probability,.25*comp.tree+.75*comp.tfm,rtol=0,atol=2e-6)
    from scripts.prepare_next_directions import VARIANTS
    template=pp.load_submission_template()
    all_candidates=[];labels=[]
    for c in summary["candidates"]:
        path=P/c["file"]
        assert sha(path)==c["sha256"]
        frame=pd.read_csv(path,dtype={"PassengerId":str})
        spec=VARIANTS[c["variant"]]
        q=spec["tree"]*comp.tree+spec["tfm"]*comp.tfm+spec["pfn"]*comp.pfn
        assert list(frame)==["PassengerId","Transported"] and len(frame)==4277
        assert frame.PassengerId.is_unique and frame.Transported.dtype==bool and not frame.isna().any().any()
        np.testing.assert_array_equal(frame.PassengerId,template.PassengerId)
        np.testing.assert_array_equal(frame.Transported,q>=.5)
        assert int((frame.Transported!=actual.Transported).sum())==c["changes_vs_current_best"]
        labels.append(frame.Transported.to_numpy());all_candidates.append(c)
    _write_json(P/f"reports/checks/{name}.json",dict(passed=True,tests=66,candidates=all_candidates,
        current_platform_best_bundle_reproduced=True,original_bundle_unchanged=True,formal_promotion=False))
    full_name="full_context_strategy_20261101"
    full_report=EXPERIMENTS_DIR/full_name
    full_protocol=read(full_report/"protocol.json");check_sources(full_protocol)
    c=read(full_report/"summary.json")["candidate"]
    full_path=P/f"models/{full_name}/full_context.joblib"
    assert sha(full_path)==read(full_report/"model.json")["model_sha256"]
    member=joblib.load(full_path)
    assert len(member.training_y)==8693
    np.testing.assert_array_equal(member.training_y,old.y)
    p=member.predict_proba(old.test_features)[:,1]
    cached=np.load(P/f"outputs/{full_name}/full_context_test.npz")["probability"]
    np.testing.assert_allclose(p,cached,rtol=0,atol=2e-6)
    q=.25*comp.tree+.375*comp.tfm+.375*p
    frame=pd.read_csv(P/c["file"],dtype={"PassengerId":str})
    assert sha(P/c["file"])==c["sha256"]
    np.testing.assert_array_equal(frame.PassengerId,template.PassengerId)
    np.testing.assert_array_equal(frame.Transported,q>=.5)
    assert len(frame)==4277 and frame.PassengerId.is_unique and frame.Transported.dtype==bool
    assert int((frame.Transported!=actual.Transported).sum())==c["changes_vs_current_best"]
    labels.append(frame.Transported.to_numpy())
    for i,a in enumerate(labels):
        assert (a!=actual.Transported.to_numpy()).any()
        for b in labels[i+1:]:assert (a!=b).any()
    _write_json(P/f"reports/checks/{full_name}.json",dict(passed=True,tests=66,candidate=c,
        full_training_rows=8693,max_reload_error=float(np.abs(p-cached).max()),all_four_pending_unique=True,
        local_cv_accuracy=None,formal_promotion=False))
    del member;gc.collect();torch.cuda.empty_cache()
    print("Four pending CSVs and observed winner bundle audited",flush=True)


def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    log=(P/"outputs/night_round_20261101_tests.log").read_text(encoding="utf-8-sig")
    assert re.search(r"Ran 66 tests",log) and re.search(r"\bOK\b",log)
    old=SearchRunner()
    audit_seeds(old);print("Seed study audited",flush=True)
    audit_names(old);print("Name study audited",flush=True)
    audit_candidates(old)


if __name__=="__main__":main()
