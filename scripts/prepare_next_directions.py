"""Materialize the observed winner and three fixed, user-requested next CSVs."""
import copy

import joblib
import numpy as np
import pandas as pd

from scripts.audit_unlabeled_views import sha,read
from src import preprocess as pp
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.train import make_submission,_write_csv,_write_json

NAME="next_directions_20261101"
BEST_RUN="manual_submission_batch_20261031_tabicl75"
VARIANTS={
    "tabicl875":dict(tree=.125,tfm=.875,pfn=0.,description="12.5% original trees + 87.5% arithmetic TabICL",order=1),
    "tabicl100":dict(tree=0.,tfm=1.,pfn=0.,description="Arithmetic TabICL only, five grouped contexts",order=3),
    "triple":dict(tree=.125,tfm=.75,pfn=.125,description="12.5% original trees + 75% TabICL + 12.5% TabPFN",order=2),
}


def main():
    report=EXPERIMENTS_DIR/NAME
    ready=P/"outputs/kaggle_ready_4690/next_batch_20261101"
    report.mkdir(parents=True,exist_ok=True)
    ready.mkdir(parents=True,exist_ok=True)
    outcomes=EXPERIMENTS_DIR/"manual_submission_batch_20261031/platform_outcomes_20261101.json"
    best=read(EXPERIMENTS_DIR/"kaggle_submissions.json")["current_platform_best"]
    assert best["model_run"]==BEST_RUN and best["score"]==.81318
    old_report=EXPERIMENTS_DIR/"manual_submission_batch_20261031"
    source=P/"outputs/manual_submission_batch_20261031/test_components.csv"
    assert read(P/"reports/checks/manual_submission_batch_20261031.json")["passed"]
    protocol=dict(name=NAME,variants=VARIANTS,threshold=.5,
        observed_platform_best=best,platform_outcomes_sha256=sha(outcomes),test_components_sha256=sha(source),
        rationale="User feedback favors 75% TabICL over 50%, while standalone PFN is weaker. Check two larger TabICL weights and a three-way mix that preserves its 75% share and replaces half of remaining tree weight with PFN.",
        caveat="Adaptive directions selected after public leaderboard feedback; no new independent validation and no score guarantee. CSVs produced at the user's explicit request for manual platform trials. No per-passenger label rules or threshold search.",
        sources={f:sha(P/f) for f in ["scripts/prepare_next_directions.py","scripts/record_manual_batch_result.py"]})
    if (report/"protocol.json").exists():
        assert read(report/"protocol.json")==protocol,"Frozen next-direction protocol changed"
    else:
        _write_json(report/"protocol.json",protocol)
    components=pd.read_csv(source,dtype={"PassengerId":str},float_precision="round_trip")
    template=pp.load_submission_template()
    np.testing.assert_array_equal(components.PassengerId,template.PassengerId)
    old_best=pd.read_csv(P/best["submission"],dtype={"PassengerId":str})
    np.testing.assert_array_equal(.25*components.tree+.75*components.tfm>=.5,old_best.Transported)
    # A separate portable bundle for the actually observed .81318 winner.
    original_path=P/"models/tabicl_probability_mean_20261030/ensemble.joblib"
    original=joblib.load(original_path)
    assert len(original["members"])==50
    assert abs(sum(m["weight"] for m in original["members"][:45])-.5)<1e-12
    assert abs(sum(m["weight"] for m in original["members"][45:])-.5)<1e-12
    members=[dict(m,weight=m["weight"]*(.5 if i<45 else 1.5)) for i,m in enumerate(original["members"])]
    bundle=dict(members=members,plan=dict(name=BEST_RUN,threshold=.5,selection_basis="Observed platform .81318; historical exploratory weight choice"),data_hashes=original["data_hashes"])
    folder=P/"models"/BEST_RUN
    folder.mkdir(parents=True,exist_ok=True)
    model_path=folder/"ensemble.joblib"
    if not model_path.exists():
        joblib.dump(bundle,model_path,compress=3)
    loaded=joblib.load(model_path)
    np.testing.assert_allclose([m["weight"] for m in loaded["members"]],[m["weight"] for m in members],atol=0,rtol=0)
    candidates=[]
    labels={}
    for variant,spec in VARIANTS.items():
        probability=spec["tree"]*components.tree+spec["tfm"]*components.tfm+spec["pfn"]*components.pfn
        frame=make_submission(components.PassengerId,probability,template,.5)
        path=ready/f"submission_{variant}_4690.csv"
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path,dtype={"PassengerId":str}),frame)
        else:
            _write_csv(frame,path)
        labels[variant]=frame.Transported.to_numpy()
        candidates.append(dict(id=f"{NAME}_{variant}",model_run=f"{NAME}_{variant}",variant=variant,
            status="prepared_not_submitted",prepared_on="2026-11-01",file=path.relative_to(P).as_posix(),sha256=sha(path),
            rows=4277,score=None,kaggle_score=None,description=spec["description"]+", threshold0.5",
            suggested_order=spec["order"],changes_vs_current_best=int((frame.Transported!=old_best.Transported).sum()),
            exploratory=True,formal_promotion=False,source_protocol_sha256=sha(report/"protocol.json"),
            audit=f"reports/checks/{NAME}.json"))
    for i,a in enumerate(labels):
        assert (labels[a]!=old_best.Transported.to_numpy()).any()
        for b in list(labels)[i+1:]:
            assert (labels[a]!=labels[b]).any()
    # Descriptive, same-row CV only; no score is used to retune these frozen choices.
    from scripts.tabicl_finetune_study import STAGES
    from src.optimization import SearchRunner
    from scripts.audit_unlabeled_views import stats
    old=SearchRunner()
    metrics=[]
    for stage,run in STAGES:
        for repeat in [1,2]:
            t=pd.read_csv(EXPERIMENTS_DIR/f"tabpfn_v2_20261031/{stage}_repeat_{repeat}_predictions.csv",float_precision="round_trip")
            tfm=np.full(len(old.y),np.nan)
            for fold in range(1,6):
                stem=f"fresh_repeat_{repeat}_fold_{fold}" if stage=="paired" else f"reference_tfm_fresh_repeat_{repeat}_fold_{fold}"
                z=np.load(P/"outputs"/run/f"{stem}.npz")
                tfm[z["validation"]]=z["probability"]
            tree=2*t.reference.to_numpy()-tfm
            reference=.25*tree+.75*tfm
            for variant,spec in VARIANTS.items():
                q=spec["tree"]*tree+spec["tfm"]*tfm+spec["pfn"]*t.component.to_numpy()
                metrics.append(dict(variant=variant,stage=stage,repeat=repeat,**stats(old.y,reference,q,t.fold.to_numpy())))
    summary=dict(status="exploratory_batch_prepared",candidates=candidates,historical_comparisons=metrics,
        platform_best_model_bundle=model_path.relative_to(P).as_posix(),platform_best_model_sha256=sha(model_path),
        source_bundle_sha256=sha(original_path),source_bundle_unchanged=True,submitted_to_kaggle=False)
    _write_json(report/"summary.json",summary)
    _write_json(ready/"manifest.json",summary)
    print([(c["variant"],c["changes_vs_current_best"]) for c in candidates])


if __name__=="__main__":
    main()
