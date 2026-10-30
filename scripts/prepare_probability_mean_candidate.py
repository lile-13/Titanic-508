"""Prepare the audited candidate for a manual Kaggle submission."""
import copy
import hashlib
import json
import shutil

import pandas as pd

from src import preprocess as pp
from src.paths import PROJECT_ROOT


RUN = "tabicl_probability_mean_20261030"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    report = PROJECT_ROOT / "reports/experiments" / RUN
    audit_path = PROJECT_ROOT / "reports/checks" / f"{RUN}.json"
    audit = read(audit_path)
    summary = read(report / "summary.json")
    assert audit["passed"] and audit["gate_passed"]
    assert summary["status"] == "candidate_generated" and summary["kaggle_score"] is None
    source = PROJECT_ROOT / "outputs" / RUN / "submission_tabicl_probability_mean4690.csv"
    digest = sha(source)
    assert digest == audit["candidate"]["sha256"]
    frame = pd.read_csv(source, dtype={"PassengerId": str})
    template = pd.read_csv(pp.find_data_dir() / "sample_submission.csv", dtype={"PassengerId": str})
    assert list(frame) == ["PassengerId", "Transported"]
    assert len(frame) == 4277 and frame.PassengerId.is_unique
    assert frame.PassengerId.equals(template.PassengerId)
    assert frame.Transported.dtype == bool and not frame.isna().any().any()
    ready = PROJECT_ROOT / "outputs/kaggle_ready_4690"
    ready.mkdir(parents=True, exist_ok=True)
    destination = ready / "submission_tabicl_mean4690.csv"
    if destination.exists() and sha(destination) != digest:
        raise ValueError("Destination already contains different bytes")
    ledger_path = PROJECT_ROOT / "reports/experiments/kaggle_submissions.json"
    ledger = read(ledger_path)
    submitted, best = copy.deepcopy(ledger["submissions"]), copy.deepcopy(ledger["current_platform_best"])
    if any(row["model_run"] == RUN for row in submitted):
        raise ValueError("Candidate already recorded as submitted; preserve its actual result")
    assert best["model_run"] == "optimization_20261025" and best["score"] == 0.81108
    assert sha(PROJECT_ROOT / best["submission"]) == audit["incumbent_submission_sha256"]
    manifest_path = ready / "manifest.json"
    manifest = read(manifest_path)
    record = dict(
        model_run=RUN, status="prepared_not_submitted", prepared_on="2026-10-30",
        file=destination.relative_to(PROJECT_ROOT).as_posix(), sha256=digest, rows=4277,
        score=None, kaggle_score=None, source_protocol_sha256=sha(report / "protocol.json"),
        audit=audit_path.relative_to(PROJECT_ROOT).as_posix(), audit_sha256=sha(audit_path),
        description="TabICLv2 probability mean (8 views), 50% frozen tree ensemble, grouped validation, threshold 0.5",
    )
    shutil.copyfile(source, destination)
    assert sha(destination) == digest
    ledger["prepared_candidates"] = [r for r in ledger.get("prepared_candidates", []) if r["model_run"] != RUN] + [record]
    assert ledger["submissions"] == submitted and ledger["current_platform_best"] == best
    manifest["additional_unsubmitted_comparisons"] = [
        r for r in manifest.get("additional_unsubmitted_comparisons", []) if r["model_run"] != RUN
    ] + [record]
    manifest["last_updated_on"] = "2026-10-30"
    write(ledger_path, ledger)
    write(manifest_path, manifest)
    write(report / "prepared_candidate.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
