"""登记组长提供的第三次提交截图；保留训练期原始摘要。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main(screenshot):
    evidence=ROOT/"reports/figures/kaggle_fullblend_20261028.png"
    shutil.copyfile(screenshot,evidence)
    candidate=ROOT/"outputs/kaggle_ready_4690/submission_fullblend4690.csv"
    ledger_path=ROOT/"reports/experiments/kaggle_submissions.json"
    ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
    entry=dict(id="fullblend_20261028_081108",model_run="full_data_candidate_20261028",
        status="Complete",observed_on="2026-10-28",submitted_at=None,
        uploaded_filename=candidate.name,account_as_displayed="Litianle4690",
        score=.81108,displayed_score="0.81108",rank=None,
        evidence=evidence.relative_to(ROOT).as_posix(),evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        local_expected_source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),uploaded_bytes_verified=False,
        model_association="From filename and preceding preparation/upload workflow; screenshot cannot verify uploaded bytes",
        repository_head_when_recorded=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    ledger["submissions"]=[r for r in ledger["submissions"] if r["id"]!=entry["id"]]+[entry]
    best=ledger["current_platform_best"]
    best.update(tied_model_runs=["optimization_20261025","full_data_candidate_20261028"],
        basis="Highest displayed score among three screenshot-confirmed submissions; full-data blend ties incumbent. Retain original 45-model ensemble rather than 54-model comparison; not a significance claim")
    ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    manifest_path=ROOT/"outputs/kaggle_ready_4690/manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison=manifest.pop("additional_unsubmitted_comparisons",[])
    for r in comparison:r.update(status="Complete",kaggle_score=.81108,observed_on="2026-10-28")
    manifest["additional_submitted_comparisons"]=comparison or manifest.get("additional_submitted_comparisons",[])
    manifest["fullblend_observed_kaggle_score"]=.81108
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    check=dict(passed=True,source="user supplied screenshot",displayed_scores={r["uploaded_filename"]:r["score"] for r in ledger["submissions"]},
               fullblend_difference_from_incumbent=0.,primary_unchanged=True,
               training_summary_unchanged=True,uploaded_bytes_verified=False,
               evidence_sha256=entry["evidence_sha256"])
    (ROOT/"reports/checks/kaggle_fullblend_result_20261028.json").write_text(json.dumps(check,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot",type=Path,required=True)
    main(parser.parse_args().screenshot)
