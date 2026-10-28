"""在新目录重训已冻结集成，不重做参数搜索：python -m scripts.refit_frozen。"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from src.optimization import SearchRunner,predict_ensemble
from src.paths import EXPERIMENTS_DIR,PROJECT_ROOT
from src.train import _write_json


def text_hash_matches(data, expected):
    """仅允许Git的LF/CRLF换行差异；不忽略代码或CSV内容的其他变化。"""
    lf=data.replace(b"\r\n",b"\n")
    return expected in {hashlib.sha256(v).hexdigest() for v in [data,lf,lf.replace(b"\n",b"\r\n")]}


def refit(source_name,run_name,threads=4):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",run_name):
        raise ValueError("run-name仅允许字母、数字、短横线和下划线")
    source=(EXPERIMENTS_DIR/source_name).resolve()
    if not source.is_relative_to(EXPERIMENTS_DIR.resolve()):
        raise ValueError("source-name必须位于实验目录内")
    plan_bytes=(source/"frozen_plan.json").read_bytes()
    plan=json.loads(plan_bytes)
    protocol=json.loads((source/"protocol.json").read_text(encoding="utf-8"))
    dependencies=json.loads((source/"dependency_hashes.json").read_text(encoding="utf-8"))
    from src import preprocess as pp
    raw_dir=pp.find_data_dir()
    for relative,digest in {**protocol["sources"],**dependencies}.items():
        path=PROJECT_ROOT/relative.replace("\\","/")
        if not text_hash_matches(path.read_bytes(),digest):
            raise ValueError(f"冻结方案源码或折划分已变化：{relative}；请另做实验，不混充历史复现")
    for name,digest in protocol["data_hashes"].items():
        if not text_hash_matches((raw_dir/name).read_bytes(),digest):
            raise ValueError(f"原始数据与冻结实验不一致：{name}")
    for path in [EXPERIMENTS_DIR/run_name,PROJECT_ROOT/"models"/run_name,PROJECT_ROOT/"outputs"/run_name]:
        if path.exists():raise FileExistsError("复现目录已存在，请换一个新的--run-name")
    runner=SearchRunner(run_name,threads)
    (runner.report/"frozen_plan.json").write_bytes(plan_bytes)
    _write_json(runner.report/"dependency_hashes.json",
                {p:hashlib.sha256((PROJECT_ROOT/p.replace('\\','/')).read_bytes()).hexdigest() for p in dependencies})
    _write_json(runner.report/"reproduction_source.json",
                dict(source_experiment=source_name,source_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                     locked_at=plan["locked_at"],purpose="Refit an unchanged historical choice; this is not a new blind confirmation or model selection"))
    # 空白源码包没有models父目录；旧finalize只创建本次模型子目录。
    (PROJECT_ROOT/"models").mkdir(exist_ok=True)
    result=runner.finalize()
    if result["status"]!="promoted_candidate":
        raise RuntimeError("复现未达到原候选生成条件；请核对环境和数据，不调整冻结参数")
    probability,submission=predict_ensemble(PROJECT_ROOT/"models"/run_name/"ensemble.joblib")
    import numpy as np
    import pandas as pd
    saved=pd.read_csv(runner.output/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    expected=pd.read_csv(runner.output/"submission.csv",dtype={"PassengerId":str})
    np.testing.assert_allclose(probability,saved.probability,atol=1e-12,rtol=0)
    pd.testing.assert_frame_equal(submission,expected)
    _write_json(runner.report/"reproduction_check.json",dict(passed=True,rows=len(submission),model_reload_matches=True,
                  max_probability_error=float(np.max(np.abs(probability-saved.probability))),
                  source_plan_unchanged=True,interpretation="Historical refit, no new Kaggle score"))
    print(f"Frozen refit completed: {runner.output/'submission.csv'}",flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-name",default="optimization_20261025")
    parser.add_argument("--run-name",default="frozen_refit_20261028")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    refit(args.source_name,args.run_name,args.threads)
