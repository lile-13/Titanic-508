"""Record four user-supplied platform scores without rewriting prior studies."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from scripts.audit_unlabeled_views import sha,read
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.train import _write_json

BATCH="manual_submission_batch_20261031"
SCORES={"pfn25":.81271,"pfn50":.81271,"pfn100":.80921,"tabicl75":.81318}
DAY="2026-11-01"


def main(screenshot):
    evidence=P/"reports/figures/kaggle_manual_batch_20261101.png"
    if evidence.exists():
        assert sha(evidence)==sha(screenshot),"Do not replace different evidence"
    else:
        shutil.copyfile(screenshot,evidence)
    candidates=read(EXPERIMENTS_DIR/BATCH/"summary.json")["candidates"]
    ledger_path=EXPERIMENTS_DIR/"kaggle_submissions.json"
    ledger=read(ledger_path)
    if ledger["current_platform_best"]["score"]>.81318:
        raise ValueError("A newer platform best is recorded; do not reset it with this historical result")
    original_entries=copy.deepcopy(ledger["submissions"][:5])
    records=[]
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=P,text=True).strip()
    for variant,score in SCORES.items():
        candidate=next(c for c in candidates if c["variant"]==variant)
        path=P/candidate["file"]
        assert sha(path)==candidate["sha256"]
        frame=pd.read_csv(path,dtype={"PassengerId":str})
        assert len(frame)==4277 and frame.Transported.dtype==bool and frame.PassengerId.is_unique
        prior=next((r for r in ledger["submissions"] if r["model_run"]==candidate["model_run"]),None)
        entry=dict(id=f"{variant}_20261101_{score:.5f}".replace(".",""),model_run=candidate["model_run"],
            status="Complete",observed_on=DAY,submitted_at=None,uploaded_filename=path.name,
            account_as_displayed="Litianle4690",score=score,displayed_score=f"{score:.5f}",rank=None,
            evidence=evidence.relative_to(P).as_posix(),evidence_sha256=sha(evidence),
            local_expected_source_sha256=candidate["sha256"],uploaded_bytes_verified=False,
            model_association="From filename, provided screenshot and preceding candidate workflow. Screenshot cannot verify uploaded file bytes.",
            repository_head_when_recorded=prior["repository_head_when_recorded"] if prior else head,
            prepared_candidate=copy.deepcopy(candidate),difference_from_previous_best=round(score-.81178,5))
        if prior:
            assert prior==entry,"Existing observed result differs"
        else:
            ledger["submissions"].append(entry)
        records.append(entry)
    runs={r["model_run"] for r in records}
    ledger["prepared_candidates"]=[r for r in ledger.get("prepared_candidates",[]) if r["model_run"] not in runs]
    winner=next(c for c in candidates if c["variant"]=="tabicl75")
    ledger["current_platform_best"]=dict(model_run=winner["model_run"],score=.81318,submission=winner["file"],
        accuracy_difference=.01005,percentage_point_difference=1.005,
        previous_best_model_run="tabicl_probability_mean_20261030",previous_best_score=.81178,
        improvement_over_previous_best=.0014,percentage_point_improvement_over_previous_best=.14,
        basis="Highest displayed score among nine screenshot-confirmed submissions. Observational platform choice after exploratory weight comparisons; not new independent validation or a significance claim.",
        tied_model_runs=[winner["model_run"]],target_score=.82,gap_to_target=.00682)
    assert ledger["submissions"][:5]==original_entries and len(ledger["submissions"])==9
    _write_json(ledger_path,ledger)
    manifest_path=P/"outputs/kaggle_ready_4690/manifest.json"
    manifest=read(manifest_path)
    manifest["additional_unsubmitted_comparisons"]=[r for r in manifest.get("additional_unsubmitted_comparisons",[]) if r["model_run"] not in runs]
    prior=[r for r in manifest.get("additional_submitted_comparisons",[]) if r["model_run"] not in runs]
    for entry in records:
        c=next(c for c in candidates if c["model_run"]==entry["model_run"])
        prior.append(dict(model_run=c["model_run"],file=c["file"],sha256=c["sha256"],rows=4277,
            kaggle_score=entry["score"],status="Complete",observed_on=DAY,evidence=entry["evidence"]))
    manifest.update(additional_submitted_comparisons=prior,current_platform_best=winner["model_run"],
        current_platform_best_score=.81318,current_platform_best_file=winner["file"],last_updated_on=DAY)
    _write_json(manifest_path,manifest)
    # The preparation summary/protocol remain historical snapshots; separate outcomes are added.
    outcomes=dict(observed_on=DAY,source="user screenshot",records=records,current_platform_best=ledger["current_platform_best"],
        caveat="No exact submission timestamps or ranking inferred. Uploaded bytes not verified. Original failed stability studies and preparation snapshots unchanged.")
    _write_json(EXPERIMENTS_DIR/BATCH/"platform_outcomes_20261101.json",outcomes)
    ready_manifest=P/"outputs/kaggle_ready_4690/next_batch_20261031/manifest.json"
    ready=read(ready_manifest)
    for c in ready["candidates"]:
        c.update(status="Complete",score=SCORES[c["variant"]],kaggle_score=SCORES[c["variant"]],observed_on=DAY)
    ready.update(status="all_four_submitted",current_platform_best=.81318,evidence=evidence.relative_to(P).as_posix())
    _write_json(ready_manifest,ready)
    _write_json(P/"reports/checks/kaggle_manual_batch_result_20261101.json",dict(passed=True,
        observed_scores={r["uploaded_filename"]:r["score"] for r in records},
        evidence_sha256=sha(evidence),observed_submissions=9,original_five_preserved=True,
        current_platform_best=.81318,improvement_over_previous_best=.0014,gap_to_target=.00682,
        uploaded_bytes_verified=False,training_and_preparation_snapshots_unchanged=True))
    print("Recorded four Complete results; current platform best .81318")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--screenshot",type=Path,required=True)
    main(parser.parse_args().screenshot)
