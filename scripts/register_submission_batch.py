"""Register audited manual candidates while preserving all observed scores."""
import copy

from scripts.audit_unlabeled_views import sha,read
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.train import _write_json

NAME="manual_submission_batch_20261031"


def main():
    report=EXPERIMENTS_DIR/NAME
    summary=read(report/"summary.json")
    audit_path=P/f"reports/checks/{NAME}.json"
    audit=read(audit_path)
    assert audit["passed"] and len(audit["candidates"])==4
    ledger_path=EXPERIMENTS_DIR/"kaggle_submissions.json"
    manifest_path=P/"outputs/kaggle_ready_4690/manifest.json"
    ledger,manifest=read(ledger_path),read(manifest_path)
    submitted,best=copy.deepcopy(ledger["submissions"]),copy.deepcopy(ledger["current_platform_best"])
    assert best["score"]==.81178 and best["model_run"]=="tabicl_probability_mean_20261030"
    additions=[]
    for row in summary["candidates"]:
        assert not any(r["model_run"]==row["model_run"] for r in submitted),"Already submitted; do not reset its status"
        assert sha(P/row["file"])==row["sha256"]
        verified=next(r for r in audit["candidates"] if r["variant"]==row["variant"])
        assert verified["sha256"]==row["sha256"]
        additions.append(dict(row,audit_sha256=sha(audit_path),source_report=f"reports/experiments/{NAME}/summary.json"))
    runs={r["model_run"] for r in additions}
    ledger["prepared_candidates"]=[r for r in ledger.get("prepared_candidates",[]) if r["model_run"] not in runs]+additions
    manifest["additional_unsubmitted_comparisons"]=[r for r in manifest.get("additional_unsubmitted_comparisons",[]) if r["model_run"] not in runs]+additions
    manifest["last_updated_on"]="2026-10-31"
    assert ledger["submissions"]==submitted and ledger["current_platform_best"]==best
    _write_json(ledger_path,ledger)
    _write_json(manifest_path,manifest)
    _write_json(report/"registration.json",dict(prepared_candidates=additions,actual_submissions_unchanged=True,
        current_platform_best_unchanged=True,current_platform_best_score=best["score"],submitted_to_kaggle=False))
    print("Four exploratory candidates registered; five observed submissions and best .81178 preserved")


if __name__=="__main__":
    main()
