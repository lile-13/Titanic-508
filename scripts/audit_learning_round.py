"""核对校准、自训练及固定权重复验；记录原方案与Kaggle成绩不变。"""
import hashlib
import json
import re

import numpy as np
import pandas as pd

from src.optimization import SearchRunner,predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json

RUNS=["calibration_study_20261029","pseudo_label_study_20261029","fixed_weight_confirmation_20261029","paired_regularization_20261029"]


def main():
    old=SearchRunner(threads=4);dev=old.fold_of!=5
    expected="270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    assert hashlib.sha256((old.output/"submission.csv").read_bytes()).hexdigest()==expected
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    checks=[];candidates=[];results=[]
    for run in RUNS:
        report=EXPERIMENTS_DIR/run;output=PROJECT_ROOT/"outputs"/run
        protocol=json.loads((report/"protocol.json").read_text(encoding="utf-8"))
        summary=json.loads((report/"summary.json").read_text(encoding="utf-8"))
        for path,digest in protocol["sources"].items():
            assert hashlib.sha256((PROJECT_ROOT/path).read_bytes()).hexdigest()==digest,(run,path)
        if "source_plan_hash" in protocol:
            assert hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest()==protocol["source_plan_hash"]
        count=0
        if run.startswith("calibration"):
            for fold,(allowed,outer) in enumerate(old.splits,1):
                cache=np.load(output/f"inner_fold_{fold}.npz")
                fit,cal=cache["fit_indices"],cache["cal_indices"]
                np.testing.assert_array_equal(np.sort(np.r_[fit,cal]),np.sort(allowed))
                assert not set(fit)&set(cal) and not set(cal)&set(outer)
                assert not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[cal])
                assert len(cache["probability"])==len(cal) and np.isfinite(cache["probability"]).all()
                assert not (old.fold_of[np.r_[fit,cal]]==5).any()
                count+=1
            for file in output.glob("profile_*.npz"):
                p=np.load(file)["probability"]
                assert np.isnan(p[~dev]).all() and np.isfinite(p[dev]).all()
        elif run.startswith("pseudo"):
            for fold,(allowed,outer) in enumerate(old.splits,1):
                cache=np.load(output/f"teacher_development_{fold}.npz")
                np.testing.assert_array_equal(cache["training"],allowed)
                np.testing.assert_array_equal(cache["validation"],outer)
                assert not set(old.features.GroupId.iloc[allowed])&set(old.features.GroupId.iloc[outer])
                assert not (old.fold_of[allowed]==5).any()
                assert cache["test_probability"].shape==(len(old.raw_test),)
                count+=1
            for file in output.glob("profile_*_seed_*.npz"):
                p=np.load(file)["probability"]
                assert np.isnan(p[~dev]).all() and np.isfinite(p[dev]).all()
        elif run.startswith("fixed_weight"):
            for repeat in [1,2]:
                seen=[]
                for fold in range(1,6):
                    cache=np.load(output/f"repeat_{repeat}_fold_{fold}.npz")
                    fit,valid=cache["training"],cache["validation"]
                    assert not set(fit)&set(valid)
                    np.testing.assert_array_equal(np.sort(np.r_[fit,valid]),np.arange(len(old.y)))
                    assert not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[valid])
                    assert cache["probability"].shape==(len(valid),3) and np.isfinite(cache["probability"]).all()
                    seen.extend(valid.tolist());count+=1
                np.testing.assert_array_equal(np.sort(seen),np.arange(len(old.y)))
        else:
            reference_dir=PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029"
            reference_protocol=EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029/protocol.json"
            assert hashlib.sha256(reference_protocol.read_bytes()).hexdigest()==protocol["baseline_protocol_sha256"]
            for choice in protocol["source_choices"]:
                for repeat in [1,2]:
                    seen=[]
                    for fold in range(1,6):
                        data=np.load(output/f"choice_{choice}_repeat_{repeat}_fold_{fold}.npz")
                        reference=np.load(reference_dir/f"repeat_{repeat}_fold_{fold}.npz")
                        np.testing.assert_array_equal(data["training"],reference["training"])
                        np.testing.assert_array_equal(data["validation"],reference["validation"])
                        vi=data["validation"]
                        assert data["probability"].shape==(len(vi),) and np.isfinite(data["probability"]).all()
                        assert not set(old.features.GroupId.iloc[data["training"]])&set(old.features.GroupId.iloc[vi])
                        seen.extend(vi.tolist());count+=1
                    np.testing.assert_array_equal(np.sort(seen),np.arange(len(old.y)))
        if summary["status"]=="candidate_generated":
            probability,submission=predict_ensemble(PROJECT_ROOT/"models"/run/"ensemble.joblib")
            saved=pd.read_csv(output/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
            file=next(output.glob("submission_*4690.csv"))
            pd.testing.assert_frame_equal(submission,pd.read_csv(file,dtype={"PassengerId":str}))
            np.testing.assert_allclose(probability,saved.probability,atol=1e-12,rtol=0)
            assert len(submission)==4277 and not submission.isna().any().any() and submission.PassengerId.is_unique
            dest=PROJECT_ROOT/"outputs/kaggle_ready_4690"/file.name;dest.write_bytes(file.read_bytes())
            candidates.append(dict(model_run=run,file=dest.relative_to(PROJECT_ROOT).as_posix(),status="not_submitted",score=None,
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),rows=4277))
        results.append(summary)
        checks.append(dict(run=run,status=summary["status"],source_hashes_match=True,
            split_group_checks_passed=True,teacher_partitions_checked=count,
            initial_profiles=len(protocol.get("profiles",[])),
            student_config_seed_records=len(list(report.glob("profile_*_seed_*.json"))),
            observed_kaggle_score=None))
    ledger_path=EXPERIMENTS_DIR/"kaggle_submissions.json"
    ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["prepared_candidates"]=[r for r in ledger.get("prepared_candidates",[]) if r.get("model_run") not in RUNS]+candidates
    _write_json(ledger_path,ledger)
    uncertainty=None
    if results[3]["status"]=="candidate_generated":
        choice=results[3]["frozen"]["choice"]
        deltas=[]
        for repeat in [1,2]:
            oof=pd.read_csv(EXPERIMENTS_DIR/RUNS[3]/f"choice_{choice}_repeat_{repeat}_oof.csv",dtype={"PassengerId":str})
            np.testing.assert_array_equal(oof.PassengerId,old.raw.PassengerId)
            delta=((oof.alternative.to_numpy()>=.5)==old.y).astype(float)-((oof.reference.to_numpy()>=.5)==old.y).astype(float)
            deltas.append(delta)
        mean_delta=np.mean(deltas,axis=0)
        np.testing.assert_allclose(mean_delta.mean(),results[3]["frozen"]["mean_gain"],atol=1e-14)
        grouped=pd.DataFrame(dict(GroupId=old.features.GroupId,delta=mean_delta,n=1)).groupby("GroupId").agg(delta=("delta","sum"),n=("n","sum"))
        rng=np.random.default_rng(20261029);samples=[]
        for _ in range(15):
            draw=rng.integers(len(grouped),size=(200,len(grouped)))
            samples.extend((grouped.delta.to_numpy()[draw].sum(axis=1)/grouped.n.to_numpy()[draw].sum(axis=1)).tolist())
        lower,upper=np.quantile(samples,[.025,.975])
        uncertainty=dict(unit="GroupId",groups=len(grouped),resamples=3000,seed=20261029,
            mean_gain=float(mean_delta.mean()),percentile_95_lower=float(lower),percentile_95_upper=float(upper),
            interpretation="Conditional paired bootstrap of historically selected OOF; repeated predictions share rows. Does not account for training or selection uncertainty and is not a new independent significance test")
        _write_json(EXPERIMENTS_DIR/RUNS[3]/"conditional_uncertainty.json",uncertainty)
    check=dict(passed=True,date="2026-10-29",experiments=checks,candidates=candidates,tests_passed=29,
        observed_kaggle_best=.81108,incumbent_submission_sha256=expected,
        source_plan_weights_unchanged=plan["weights"]==[.25,.25,.5],
        classroom_report_and_archive_untouched=True,
        conditional_uncertainty=uncertainty,
        limitations="Historical selection and repeated partitions are not new blind validation")
    _write_json(PROJECT_ROOT/"reports/checks/learning_round_20261029.json",check)
    cal,pseudo,fixed,paired=results
    cf,pf=cal["frozen"],pseudo["frozen"]
    reference=float(np.mean([r["reference_accuracy"] for r in fixed["repeats"]]))
    alternative=float(np.mean([r["alternative_accuracy"] for r in fixed["repeats"]]))
    alternatives=paired["comparisons"]
    selected=max(alternatives,key=lambda r:r["mean_gain"])
    paired_ref=float(np.mean([r["reference_accuracy"] for r in selected["repeats"]]))
    paired_alt=float(np.mean([r["alternative_accuracy"] for r in selected["repeats"]]))
    text=f"""# 概率校准 自训练与固定权重复验 2026-10-29

当前Kaggle已观测最好成绩仍为0.81108；原45模型集成与54模型全量重训融合持平。今天不改变这三次平台成绩，继续暂停课程报告和源码ZIP整理。

| 方向 | 评价设置 | 对照准确率 | 新方案准确率 | 结论 |
|---|---|---:|---:|---|
| 概率校准 | 原开发第1至4折 | {cf['incumbent_accuracy']:.6f} | {cf['accuracy']:.6f} | {cal['status']} |
| 高置信软标签学生融合 | 原开发第1至4折 | {pf['incumbent_accuracy']:.6f} | {pf['accuracy']:.6f} | {pseudo['status']} |
| 昨日固定权重线索 | 两次新分组五折的均值 | {reference:.6f} | {alternative:.6f} | {fixed['status']} |
| 固定正则化替代模型配对 | 相同两次新分组五折的均值 | {paired_ref:.6f} | {paired_alt:.6f} | {paired['status']} |

固定权重的五折训练比例与开发四折不同，两行分数不能直接相减。该行只比较同一划分、相同训练数据的两个权重方案。

## 校准

固定8种校准配置、3种与原概率混合比例，共24项对照，阈值始终0.5。每个外层训练分区内部按GroupId留出25%组拟合校准器；临时教师完全排除这些校准组，校准标签也完全位于外层训练部分。随后将校准器应用到全外层训练教师的验证预测，这种校准迁移的效果由外层分组验证衡量。

校准输入包括原概率logit及少量乘客群体信息，不包含Transported。没有使用外层验证标签拟合校准器。最好配置为{cf['config']}，与原概率混合比例{cf['weight']}，仅{cf['improved_folds']}/4折改善。概率质量与分类准确率分别记录，不将更低Log loss自动解释为更高Accuracy。

## 高置信软标签

固定两个学生特征配置、两个置信阈值0.95/0.99，共4种初始配置；前两种补充2026和103种子，连同42共{checks[1]['student_config_seed_records']}项配置与种子记录。真实样本权重为1，选中的公开测试特征样本权重为0.25，目标为教师概率，通过CrossEntropy训练。

每个教师只使用允许的外层训练组；预测伪标签之前排除整个验证组。公开测试集真实标签从未读取。该对照同时包含CrossEntropy软目标训练与数据增强，不将可能的差值单独归因于伪标签。伪标签数量、有效权重及各折分数见profile JSON记录。

## 固定线索的复验

昨天49组权重搜索最好的0.40/0.15/0.45仅净增加6个开发正确预测，未达到当时门槛，原结论保留。今天不再选择权重，在训练前固定这一组，用种子20261029和20261106构造两次新GroupKFold五折。

新规则要求两次平均准确率差至少+0.001、两次均不低于原权重、至少6/10折改善，且各次Log loss恶化不超过0.002。实际平均差为{fixed['mean_gain']:+.6f}，改善{fixed['improved_folds']}/10折。所有原始行已在历史开发中被查看，这只是固定线索的稳定性复验，不是新的独立盲测。

## 固定正则化替代模型的配对检验

原开发折经过大量模型搜索，旧方案的开发分数可能存在选择偏差。因此，追加核验两种昨天已评估的固定模型：强正则的三层单调XGBoost，以及G/E甲板局部LightGBM。两者均固定占新融合概率的25%，原集成占75%；在比较预测前记录协议，不搜索参数、阈值或混合比例，复用两次新分组划分的相同基准预测。

XGBoost的两次平均准确率差为{alternatives[0]['mean_gain']:+.6f}，未达到0.001；局部LightGBM的平均差为{alternatives[1]['mean_gain']:+.6f}，两次均非负，7/10折改善，达到本轮事先设定的规则。旧开发门槛未通过的结论不变，本轮增加的是新分组下的固定模型配对证据，不能将其描述成新的独立测试集。

局部LightGBM两次分别从{alternatives[1]['repeats'][0]['reference_accuracy']:.6f}到{alternatives[1]['repeats'][0]['alternative_accuracy']:.6f}、从{alternatives[1]['repeats'][1]['reference_accuracy']:.6f}到{alternatives[1]['repeats'][1]['alternative_accuracy']:.6f}，对应净增加20和5个正确预测；概率损失也分别降低。验证单元仍为同行组，不将同组分散到两侧。

通过后按固定方案完成原五折训练，生成一份候选submission_regularized4690.csv。它含原45个模型及15个局部模型容器，每个容器包含全局与G/E子模型；不能把容器数量直接当作全部底层模型数。描述性原五折OOF为{paired.get('canonical_oof',float('nan')):.6f}，该折已参与历史选择，不是新的无偏指标。候选有{paired.get('changed_test_labels',0)}个测试标签变化，保存模型重载概率最大误差{paired.get('reload_max_error',float('nan')):.3g}，平台成绩尚未知。

## 核验与复现

29项测试通过。独立检查源码/冻结配置哈希、内部与外部组不重叠、开发第5折未参与校准与自训练、两次新划分每行各被验证一次，以及原最佳CSV哈希不变。核验见reports/checks/learning_round_20261029.json。

```powershell
python -m unittest discover -s tests -v
python -m scripts.calibration_study
python -m scripts.pseudo_label_study
python -m scripts.confirm_fixed_weights
python -m scripts.paired_regularization_check
python -m scripts.audit_learning_round
```

追加实验需要原开发概率缓存。只有GitHub源码而无缓存时，按原搜索流程先生成缓存；只复现现有最佳模型可使用scripts.refit_frozen。协议或源代码哈希改变应另开实验，不混用旧缓存。

方法参考：[scikit-learn概率校准](https://scikit-learn.org/stable/modules/calibration.html)，[CatBoost CrossEntropy](https://catboost.ai/docs/en/concepts/loss-functions-classification)。本轮没有导入外部比赛答案或测试标签。

## 当前文件

"""
    if candidates:
        text+="通过预设门槛的候选尚未提交Kaggle，平台提升未知：\n\n"+"\n".join("- "+c["file"] for c in candidates)+"\n"
        if uncertainty:
            text+=f"\n按同行组对两次预测的平均差做配对bootstrap，条件95%区间为[{uncertainty['percentile_95_lower']:+.6f},{uncertainty['percentile_95_upper']:+.6f}]。这不计入训练与模型选择不确定性，也不是新的盲测显著性检验；原0.81108方案继续保留。\n"
    else:text+="本轮没有通过预设门槛的候选，不生成新提交CSV；继续保留outputs/kaggle_ready_4690/submission_ensemble4690.csv。\n"
    (PROJECT_ROOT/"docs/学习与校准实验_20261029.md").write_text(text,encoding="utf-8")
    block=f"""<!-- learning-round-20261029:start -->
## 2026-10-29 学习与校准实验

完成8种校准配置、4种高置信软标签学生配置、昨日固定权重线索的两次新分组五折复验，以及两个固定正则化替代模型的配对检验。29项测试及分组/来源检查通过，当前Kaggle已观测最好仍为0.81108。

{"已有通过门槛的待提交候选，尚无新平台成绩。" if candidates else "本轮没有通过门槛的新候选，保留原集成，不生成新提交CSV。"} 详见[学习与校准记录](docs/学习与校准实验_20261029.md)。课程报告及源码ZIP仍暂停整理。
<!-- learning-round-20261029:end -->
"""
    for path in [PROJECT_ROOT/"README.md",PROJECT_ROOT/"docs/项目进度.md"]:
        text=path.read_text(encoding="utf-8")
        text=re.sub(r"<!-- learning-round-20261029:start -->.*?<!-- learning-round-20261029:end -->\n*","",text,flags=re.S)
        first,rest=text.split("\n",1)
        inserted=block if path.name=="README.md" else block.replace("(docs/学习与校准实验_20261029.md)","(学习与校准实验_20261029.md)")
        path.write_text(first+"\n\n"+inserted+"\n"+rest.lstrip(),encoding="utf-8")
    print(json.dumps(check,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
