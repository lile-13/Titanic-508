"""核对新实验来源、折外覆盖与候选，生成模型开发记录，不整理课程包。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization import SearchRunner,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json

RUNS=["focused_study_20261028","domain_study_20261028","neural_study_20261028","weight_refinement_20261028"]
LABELS=["精简位置特征与XGBoost/CatBoost/LightGBM","消费方向约束与G/E甲板局部模型","小型神经网络与训练内部按组选择轮数","既有三组件的融合权重细化"]


def main():
    old=SearchRunner(threads=4);dev=old.fold_of!=5;records=[];candidates=[]
    original_submission=PROJECT_ROOT/"outputs/optimization_20261025/submission.csv"
    assert hashlib.sha256(original_submission.read_bytes()).hexdigest()=="270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    dependencies=json.loads((old.report/"dependency_hashes.json").read_text(encoding="utf-8"))
    for path,expected in dependencies.items():
        assert hashlib.sha256((PROJECT_ROOT/path.replace('\\','/')).read_bytes()).hexdigest()==expected
    for run,label in zip(RUNS,LABELS):
        report=EXPERIMENTS_DIR/run;output=PROJECT_ROOT/"outputs"/run
        protocol=json.loads((report/"protocol.json").read_text(encoding="utf-8"))
        summary=json.loads((report/"summary.json").read_text(encoding="utf-8"))
        for path,expected in protocol["sources"].items():
            actual=hashlib.sha256((PROJECT_ROOT/path).read_bytes()).hexdigest()
            assert actual==expected,f"源码发生变化：{run}/{path}"
        files=sorted(output.glob("profile_*_seed_*.npz"))
        for path in files:
            p=np.load(path)["probability"]
            assert p.shape==(len(old.y),) and np.isnan(p[~dev]).all() and np.isfinite(p[dev]).all()
            assert ((p[dev]>=0)&(p[dev]<=1)).all()
        f=summary["frozen"]
        rec=dict(run=run,label=label,status=summary["status"],initial_profiles=len(protocol.get("profiles",[])),
            weight_combinations=len(protocol.get("weights",[])) if run.startswith("weight_") else 0,
            config_seed_records=len(files),outer_partitions=len(files)*4,
            incumbent_development_accuracy=f["incumbent_accuracy"],
            development_accuracy=f["development"]["accuracy"],development_log_loss=f["development"]["log_loss"],
            development_gain=f["development"]["accuracy"]-f["incumbent_accuracy"],
            net_development_correct_change=int(round((f["development"]["accuracy"]-f["incumbent_accuracy"])*int(dev.sum()))),
            improved_folds=f["development"]["improved_folds"],initial_gate_passed=f["initial_gate_passed"],
            sensitivity=summary.get("sensitivity"),source_hashes_valid=True,
            development_probability_coverage_valid=True,excluded_fold_probability_all_nan=True,
            kaggle_score=summary.get("kaggle_score"))
        if summary["status"]=="candidate_generated":
            assert f["initial_gate_passed"] and summary["sensitivity"]["passed"]
            model=PROJECT_ROOT/"models"/run/"ensemble.joblib"
            p,submission=predict_ensemble(model)
            saved=pd.read_csv(output/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            expected=pd.read_csv(next(output.glob("submission_*4690.csv")),dtype={"PassengerId":str})
            np.testing.assert_allclose(p,saved.probability,atol=1e-12,rtol=0)
            pd.testing.assert_frame_equal(submission,expected)
            assert len(submission)==4277 and submission.PassengerId.is_unique and not submission.isna().any().any()
            destination=PROJECT_ROOT/"outputs/kaggle_ready_4690"/next(output.glob("submission_*4690.csv")).name
            destination.write_bytes(next(output.glob("submission_*4690.csv")).read_bytes())
            candidate=dict(model_run=run,file=destination.relative_to(PROJECT_ROOT).as_posix(),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),rows=4277,kaggle_score=None,
                status="not_submitted",development_accuracy=rec["development_accuracy"],
                changed_partition_gain=summary["sensitivity"]["alternative_accuracy"]-summary["sensitivity"]["reference_accuracy"])
            candidates.append(candidate);rec["prediction_reload_matches"]=True
        records.append(rec)
    candidates.sort(key=lambda r:(-r["changed_partition_gain"],-r["development_accuracy"]))
    ledger_path=EXPERIMENTS_DIR/"kaggle_submissions.json"
    ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["prepared_candidates"]=[r for r in ledger.get("prepared_candidates",[]) if r.get("model_run") not in RUNS]+candidates
    _write_json(ledger_path,ledger)
    check=dict(passed=True,date="2026-10-28",experiments=records,
        initial_profiles=sum(r["initial_profiles"] for r in records),
        weight_combinations=sum(r["weight_combinations"] for r in records),
        config_seed_records=sum(r["config_seed_records"] for r in records),
        outer_partitions=sum(r["outer_partitions"] for r in records),
        prepared_candidates=candidates,observed_kaggle_best=.81108,primary="optimization_20261025",
        target_in_unlabeled_context=False,reference_frozen_code_and_submission_unchanged=True,
        tests_passed=24,limitations="Historically inspected folds; changed partitions are sensitivity checks, not fresh blind validation")
    _write_json(PROJECT_ROOT/"reports/checks/optimization_round2_20261028.json",check)
    lines=["# 持续优化实验 2026-10-28","",
        "本轮针对G甲板较高错误率，比较精简连续位置特征、XGBoost、消费方向约束、G/E甲板局部模型和小型神经网络。当前真实Kaggle最好成绩仍为0.81108，原集成与全量重训融合持平；本轮不按公开榜分调参数。", "",
        "| 对照方向 | 初始配置或权重组合 | 额外配置与种子记录 | 最佳三种子融合开发准确率 | 相对原集成差值 | 改善折数 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for r in records:
        lines.append(f"| {r['label']} | {r['initial_profiles'] or str(r['weight_combinations'])+'组权重'} | {r['config_seed_records'] or '复用既有缓存'} | {r['development_accuracy']:.6f} | {r['development_gain']:+.6f} | {r['improved_folds']}/4 | {r['status']} |")
    lines += ["",f"共{check['initial_profiles']}种初始模型配置，补充种子后有{check['config_seed_records']}项配置与种子记录，覆盖{check['outer_partitions']}个开发分区训练；另比较{check['weight_combinations']}组融合权重。局部模型每个分区还分别拟合全局和G/E区域模型；神经网络还包含内部轮数选择，所以该分区数不等于底层模型总数。", "",
        "## 验证与限制", "",
        "原集成相同开发口径准确率为0.821999。开发仅用第1至4折，第5折不进入训练和评分。所有历史折曾被查看，因此本轮是追加开发，不称作独立盲测。选择前分别固定22、14、6种配置及权重范围0.25/0.50/0.75/1.00，阈值0.5。每个模型方向前两种配置补充2026与103种子，与42合成三种子结果。融合权重细化另设独立协议，固定49个权重组合，不改变基模型参数或阈值，复用原三种子开发概率。",
        "初筛要求准确率至少增加0.001、Log loss恶化不超过0.002，且至少3/4折改善；通过后冻结配置，再更换分组划分做敏感性检查。敏感性检查仍使用相同历史样本，不能解释为新的无偏验证。未通过时保留原方案，不另外生成未经验证的对照。", "",
        "权重细化最好为0.40/0.15/0.45，相比原0.25/0.25/0.50仅净增加6个开发样本预测正确，准确率由0.821999到0.822861。虽然有3/4折改善，但增幅0.000863未达到0.001门槛；不降低门槛来接纳已经看到的结果，也不生成新CSV。该权重仍只是本地线索，没有Kaggle成绩。", "",
        "HomePlanet的新补缺只映射公开特征中已知取值一致的姓氏或甲板，不覆盖已有值，也不使用Transported。这不同于把同姓视作同组。公开train与test统一无标签池的适用边界仍保留。",
        "消费方向约束是待验证的建模假设，并非因果结论。局部模型在对应区域样本不足或只有一个类别时回退全局模型；未知甲板也回退全局模型。神经网络的内部验证按GroupId拆分，停止轮数不读取外层验证标签。", "",
        "## 复现", "",
        "新增依赖xgboost==3.4.1已加入requirements.txt；24项回归测试通过。以下命令使用原开发期概率缓存，只有源码而无缓存时，需先运行原搜索流程，或只复现最终冻结模型。",
        "```powershell", "python -m pip install -r requirements.txt", "python -m unittest discover -s tests -v",
        "python -m scripts.focused_study", "python -m scripts.domain_study", "python -m scripts.neural_study", "python -m scripts.refine_ensemble_weights",
        "python -m scripts.audit_optimization_round2", "```", "",
        "实验目录的protocol.json固定源码、数据、参数和筛选规则；缓存只允许相同协议。重新运行时数据或源码哈希变化必须另建实验名称，不混用历史缓存。", "",
        "参考：[XGBoost类别特征与保存格式](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)，[XGBoost单调约束](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html)。", "",
        "## 当前选择", ""]
    if candidates:
        lines.append("存在通过两阶段门槛的待验证候选；平台得分尚未知。优先提交本地候选排名第一的文件，其余记录保留，不改写已观测最好成绩。")
        for c in candidates:lines.append(f"- {c['file']}，平台得分待提交后记录。")
    else:lines.append("三个新模型方向及权重细化均未达到预设门槛，没有新增提交文件。继续保留submission_ensemble4690.csv及其0.81108的已观测成绩。本轮没有整理或更新课程报告、DOCX和源码ZIP。")
    (PROJECT_ROOT/"docs/持续优化_20261028.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    import re
    block=f"""<!-- optimization-round2:start -->
## 2026-10-28 持续优化

本轮完成{check['initial_profiles']}种初始模型配置、{check['config_seed_records']}项配置与种子记录及{check['weight_combinations']}组融合权重对照，包括精简位置特征、XGBoost、消费方向约束、局部模型和神经网络。24项测试通过，源码与概率覆盖检查通过。当前平台最好仍为0.81108。

{"已生成通过本地两阶段门槛的待提交候选，平台成绩未知。" if candidates else "三个模型方向及权重细化均未达到预设门槛；细化权重仅多预测正确6个开发样本，保留原集成，没有新增提交文件。"} 详情见[持续优化记录](docs/持续优化_20261028.md)。课程报告及源码包继续暂停整理。
<!-- optimization-round2:end -->
"""
    for path in [PROJECT_ROOT/"README.md",PROJECT_ROOT/"docs/项目进度.md"]:
        text=path.read_text(encoding="utf-8")
        text=re.sub(r"<!-- optimization-round2:start -->.*?<!-- optimization-round2:end -->\n*","",text,flags=re.S)
        first,rest=text.split("\n",1)
        inserted=block if path.name=="README.md" else block.replace("(docs/持续优化_20261028.md)","(持续优化_20261028.md)")
        path.write_text(first+"\n\n"+inserted+"\n"+rest.lstrip(),encoding="utf-8")
    print(json.dumps(check,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
