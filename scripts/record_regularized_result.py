"""登记第四次提交的用户截图，保留训练期原始摘要及原最佳方案。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN="paired_regularization_20261029"
EXPECTED="8715f16bb14eb6d5e4cdec84cd55b39ac1977208be71a1220833274b8a4d8ce9"


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main(screenshot):
    candidate=ROOT/"outputs/kaggle_ready_4690/submission_regularized4690.csv"
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==EXPECTED
    evidence=ROOT/"reports/figures/kaggle_regularized_20261029.png"
    shutil.copyfile(screenshot,evidence)
    ledger_path=ROOT/"reports/experiments/kaggle_submissions.json"
    ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
    previous=next((r for r in ledger["submissions"] if r["model_run"]==RUN),{})
    entry=dict(id="regularized_20261029_081061",model_run=RUN,
        status="Complete",observed_on="2026-10-29",submitted_at=None,
        uploaded_filename=candidate.name,account_as_displayed="Litianle4690",
        score=.81061,displayed_score="0.81061",rank=None,
        evidence=evidence.relative_to(ROOT).as_posix(),
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        local_expected_source_sha256=EXPECTED,uploaded_bytes_verified=False,
        model_association="From filename and preceding preparation/upload workflow; screenshot cannot verify uploaded bytes",
        repository_head_when_recorded=previous.get("repository_head_when_recorded") or
            subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    ledger["submissions"]=[r for r in ledger["submissions"] if r["model_run"]!=RUN]+[entry]
    ledger["prepared_candidates"]=[r for r in ledger.get("prepared_candidates",[]) if r["model_run"]!=RUN]
    assert ledger["current_platform_best"]["model_run"]=="optimization_20261025"
    assert ledger["current_platform_best"]["score"]==.81108
    ledger["current_platform_best"]["basis"]=(
        "Highest displayed score among four screenshot-confirmed submissions; full-data blend ties incumbent and regularized blend scores lower. Retain original 45-model ensemble; not a significance claim")
    write_json(ledger_path,ledger)
    manifest_path=ROOT/"outputs/kaggle_ready_4690/manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["additional_unsubmitted_comparisons"]=[r for r in manifest.get("additional_unsubmitted_comparisons",[]) if r["model_run"]!=RUN]
    comparisons=[r for r in manifest.get("additional_submitted_comparisons",[]) if r["model_run"]!=RUN]
    comparisons.append(dict(model_run=RUN,file=candidate.relative_to(ROOT).as_posix(),
        sha256=EXPECTED,rows=4277,kaggle_score=.81061,status="Complete",observed_on="2026-10-29"))
    manifest.update(additional_submitted_comparisons=comparisons,last_updated_on="2026-10-29")
    write_json(manifest_path,manifest)
    write_json(ROOT/"reports/checks/kaggle_regularized_result_20261029.json",
        dict(passed=True,source="user supplied screenshot",
            displayed_scores={r["uploaded_filename"]:r["score"] for r in ledger["submissions"]},
            regularized_difference_from_incumbent=-.00047,primary_unchanged=True,
            training_summary_unchanged=True,uploaded_bytes_verified=False,
            evidence_sha256=entry["evidence_sha256"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot",type=Path,required=True)
    main(parser.parse_args().screenshot)
