"""核验隔离目录重训与历史集成逐项一致，并记录实际验证范围。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import PROJECT_ROOT
from src.train import _write_json


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=PROJECT_ROOT
    snapshot=root/"outputs/reproduction_snapshot_20261028"
    name="isolated_refit_v3_20261028"
    old=root/"outputs/optimization_20261025"
    new=snapshot/"outputs"/name
    report=snapshot/"reports/experiments"/name
    a=pd.read_csv(old/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    b=pd.read_csv(new/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    assert digest(old/"submission.csv")==digest(new/"submission.csv")
    old_oof=root/"reports/experiments/optimization_20261025/oof_predictions.csv"
    new_oof=report/"oof_predictions.csv"
    pd.testing.assert_frame_equal(pd.read_csv(old_oof,dtype={"PassengerId":str,"GroupId":str}),
                                 pd.read_csv(new_oof,dtype={"PassengerId":str,"GroupId":str}),check_exact=True)
    manifest_path=root/"outputs/reproduction_snapshot_manifest_20261028.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    amendments=[]
    for relative,expected in manifest["files"].items():
        actual=digest(snapshot/relative)
        if actual!=expected:
            assert relative=="scripts/refit_frozen.py",relative
            assert actual==digest(root/relative)
            amendments.append(dict(path=relative,initial_hash=expected,executed_hash=actual,
                                   reason="Create models parent directory before historical finalize in a clean checkout"))
    check=json.loads((report/"reproduction_check.json").read_text(encoding="utf-8"))
    assert check["passed"] and check["model_reload_matches"]
    summary=dict(passed=True,date="2026-10-28",same_machine=True,same_python_environment=True,
        independent_person_or_machine=False,source_head=manifest["source_head"],
        included_uncommitted_work=True,initial_source_files=len(manifest["files"]),
        initial_manifest_sha256=digest(manifest_path),source_amendments=amendments,
        snapshot=snapshot.relative_to(root).as_posix(),run_name=name,
        training_started_without_model_weights_or_prediction_cache=True,
        submission_byte_identical=True,submission_sha256=digest(new/"submission.csv"),
        oof_identical=True,test_probabilities_exactly_equal=True,
        probability_max_error=float(np.max(np.abs(a.probability-b.probability))),
        independent_model_reload=check,
        caveat="Same-environment isolated-directory reproduction; not a fresh blind evaluation or a claim of teammate reproduction")
    _write_json(root/"reports/checks/isolated_reproduction_20261028.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
