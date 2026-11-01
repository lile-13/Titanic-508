"""Register the four screenshot-confirmed results after the 0.81318 winner."""
import copy
import hashlib
import json
import subprocess

import pandas as pd

from src.paths import PROJECT_ROOT as P
from src.train import _write_json

DAY = "2026-11-01"
BATCH = "next_directions_20261101"
SCORES = {
    "submission_tabicl875_4690.csv": 0.81295,
    "submission_fullcontext_4690.csv": 0.81271,
    "submission_triple_4690.csv": 0.81295,
    "submission_tabicl100_4690.csv": 0.81155,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = P / "reports/experiments" / BATCH
    evidence = P / "reports/figures/kaggle_next_batch_feedback_20261101.png"
    ledger_path = P / "reports/experiments/kaggle_submissions.json"
    manifest_path = P / "outputs/kaggle_ready_4690/manifest.json"
    ready_dir = P / "outputs/kaggle_ready_4690/next_batch_20261101"
    candidates = read(report / "registration.json")["candidates"]
    ledger, manifest = read(ledger_path), read(manifest_path)
    ready = read(ready_dir / "manifest.json")
    full = read(ready_dir / "full_context_manifest.json")
    original = copy.deepcopy(ledger["submissions"][:9])
    best = copy.deepcopy(ledger["current_platform_best"])
    assert len(original) == 9 and best["score"] == .81318
    assert len(ledger["submissions"]) in (9, 13)
    assert len(candidates) == 4
    expected_ids = pd.read_csv(P / "data/raw/test.csv", dtype={"PassengerId": str}).PassengerId
    preserved = list(report.glob("*.json")) + list((P / "reports/experiments/full_context_strategy_20261101").glob("*.json"))
    preserved += [P / "reports/checks/next_directions_20261101.json", P / "reports/checks/full_context_strategy_20261101.json", P / "scripts/record_manual_batch_result.py"]
    frozen = {p.relative_to(P).as_posix(): sha(p) for p in preserved if not p.name.startswith("platform_outcomes_")}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=P, text=True).strip()
    records = []
    for c in candidates:
        path = P / c["file"]
        score = SCORES[path.name]
        assert sha(path) == c["sha256"]
        frame = pd.read_csv(path, dtype={"PassengerId": str})
        assert list(frame) == ["PassengerId", "Transported"]
        assert len(frame) == 4277 and not frame.isna().any().any()
        assert frame.Transported.dtype == bool and frame.PassengerId.is_unique
        assert frame.PassengerId.equals(expected_ids)
        prior = next((r for r in ledger["submissions"] if r["model_run"] == c["model_run"]), None)
        entry = dict(id=f"{c['model_run']}_{score:.5f}".replace(".", ""),
            model_run=c["model_run"], status="Complete", observed_on=DAY,
            submitted_at=None, uploaded_filename=path.name, account_as_displayed="Litianle4690",
            score=score, displayed_score=f"{score:.5f}", rank=None,
            evidence=evidence.relative_to(P).as_posix(), evidence_sha256=sha(evidence),
            local_expected_source_sha256=c["sha256"], uploaded_bytes_verified=False,
            model_association="Filename and user screenshot match the prepared workflow; uploaded server bytes remain unverified.",
            repository_head_when_recorded=prior["repository_head_when_recorded"] if prior else head,
            prepared_candidate=copy.deepcopy(c), difference_from_previous_best=round(score-.81318, 5))
        if prior:
            assert prior == entry, "Conflicting existing platform result"
        else:
            ledger["submissions"].append(entry)
        records.append(entry)
    runs = {r["model_run"] for r in records}
    assert len(runs) == 4 and {r["uploaded_filename"] for r in records} == set(SCORES)
    assert ledger["submissions"][:9] == original and len(ledger["submissions"]) == 13
    ledger["prepared_candidates"] = [r for r in ledger.get("prepared_candidates", []) if r["model_run"] not in runs]
    ledger["current_platform_best"]["basis"] = best["basis"].replace("nine screenshot-confirmed", "thirteen screenshot-confirmed")
    assert {k: v for k, v in ledger["current_platform_best"].items() if k != "basis"} == {k: v for k, v in best.items() if k != "basis"}
    manifest["additional_unsubmitted_comparisons"] = [r for r in manifest.get("additional_unsubmitted_comparisons", []) if r["model_run"] not in runs]
    submitted = [r for r in manifest.get("additional_submitted_comparisons", []) if r["model_run"] not in runs]
    for r in records:
        c = r["prepared_candidate"]
        submitted.append(dict(model_run=r["model_run"], file=c["file"], sha256=c["sha256"], rows=4277,
            kaggle_score=r["score"], status="Complete", observed_on=DAY, evidence=r["evidence"], uploaded_bytes_verified=False))
    manifest.update(additional_submitted_comparisons=submitted, last_updated_on=DAY)
    for c in ready["candidates"]:
        score = SCORES[(P / c["file"]).name]
        c.update(status="Complete", score=score, kaggle_score=score, observed_on=DAY, uploaded_bytes_verified=False)
    ready.update(status="all_four_submitted", submitted_to_kaggle=True,
        current_platform_best=ledger["current_platform_best"], evidence=evidence.relative_to(P).as_posix())
    full.update(status="Complete", score=.81271, kaggle_score=.81271, observed_on=DAY, uploaded_bytes_verified=False)
    outcomes = dict(observed_on=DAY, source="user screenshot", records=records,
        current_platform_best=ledger["current_platform_best"],
        caveat="Submission timestamps and ranks unknown; server bytes unverified. Historical preparation reports remain unchanged.")
    # Write only after every candidate and existing record has passed validation.
    for path, payload in [(ledger_path, ledger), (manifest_path, manifest),
                          (ready_dir / "manifest.json", ready), (ready_dir / "full_context_manifest.json", full),
                          (report / "platform_outcomes_20261101.json", outcomes)]:
        _write_json(path, payload)
    lines = ["当前实际最佳：submission_tabicl75_4690.csv = 0.81318。", "本批4份均已完成提交，无需重复上传。", ""]
    lines += [f"{name} | Complete | Kaggle分数：{score:.5f}" for name, score in SCORES.items()]
    lines += ["", "截图确认日期：2026-11-01；准确提交时刻和排名未知；服务器上传字节未经验证。"]
    (ready_dir / "提交顺序和成绩记录.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    assert all(sha(P / path) == digest for path, digest in frozen.items())
    _write_json(P / "reports/checks/kaggle_next_batch_result_20261101.json", dict(
        passed=True, observed_submissions=13, original_nine_preserved=True,
        observed_scores=SCORES, evidence_sha256=sha(evidence), current_platform_best=.81318,
        uploaded_bytes_verified=False, candidate_csv_hashes={r["uploaded_filename"]: r["local_expected_source_sha256"] for r in records},
        csv_rows=4277, csv_id_order_bool_and_missing_checks_passed=True,
        preserved_preparation_sha256=frozen, outcomes_sha256=sha(report / "platform_outcomes_20261101.json")))
    print("Recorded 13 Complete submissions; best remains 0.81318; preparation snapshots unchanged.")


if __name__ == "__main__":
    main()
