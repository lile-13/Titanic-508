"""Publish audited local CSV pointers, preserving all nine observed results."""
import copy

from scripts.audit_unlabeled_views import sha,read
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.train import _write_json


def main():
    report=EXPERIMENTS_DIR/"next_directions_20261101"
    summary=read(report/"summary.json")
    full=read(EXPERIMENTS_DIR/"full_context_strategy_20261101/summary.json")["candidate"]
    records=copy.deepcopy(summary["candidates"])+[copy.deepcopy(full)]
    order={"next_directions_20261101_tabicl875":1,"full_context_strategy_20261101":2,
           "next_directions_20261101_triple":3,"next_directions_20261101_tabicl100":4}
    for row in records:
        audit=P/row["audit"]
        assert read(audit)["passed"]
        assert sha(P/row["file"])==row["sha256"]
        row["audit_sha256"]=sha(audit)
        row["suggested_order"]=order[row["model_run"]]
    records.sort(key=lambda r:r["suggested_order"])
    ledger_path=EXPERIMENTS_DIR/"kaggle_submissions.json"
    ledger=read(ledger_path)
    before,best=copy.deepcopy(ledger["submissions"]),copy.deepcopy(ledger["current_platform_best"])
    assert len(before)==9 and best["score"]==.81318
    ids={r["model_run"] for r in records}
    assert not any(r["model_run"] in ids for r in before),"Do not reset submitted candidates"
    ledger["prepared_candidates"]=[r for r in ledger.get("prepared_candidates",[]) if r["model_run"] not in ids]+records
    assert ledger["submissions"]==before and ledger["current_platform_best"]==best
    _write_json(ledger_path,ledger)
    path=P/"outputs/kaggle_ready_4690/manifest.json"
    manifest=read(path)
    manifest["additional_unsubmitted_comparisons"]=[r for r in manifest.get("additional_unsubmitted_comparisons",[]) if r["model_run"] not in ids]+records
    manifest.update(last_updated_on="2026-11-01",current_platform_best_model_bundle=summary["platform_best_model_bundle"])
    _write_json(path,manifest)
    ready=P/"outputs/kaggle_ready_4690/next_batch_20261101"
    _write_json(ready/"manifest.json",dict(status="four_exploratory_candidates_ready",candidates=records,
        current_platform_best=best,submitted_to_kaggle=False))
    guide=["当前实际最佳：submission_tabicl75_4690.csv = 0.81318。", "以下4份尚未提交，属于探索候选；仅上传.csv文件。",""]
    for row in records:
        guide.append(f"{row['suggested_order']}. {row['file'].split('/')[-1]} | {row['description']} | 与当前最佳不同{row['changes_vs_current_best']}人 | Kaggle分数：待填")
    guide.extend(["","提交后请反馈文件名和平台分数。全部为4277行、True/False标签；本地验证不能代替实际榜分。"])
    (ready/"提交顺序和成绩记录.txt").write_text("\n".join(guide)+"\n",encoding="utf-8-sig")
    _write_json(report/"registration.json",dict(candidates=records,observed_submissions_unchanged=True,
        actual_best_score=.81318,submitted_to_kaggle=False))
    print("Four audited CSVs ready; nine observed results and best .81318 preserved")


if __name__=="__main__":main()
