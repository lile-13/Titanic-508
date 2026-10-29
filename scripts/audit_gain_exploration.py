"""Audit the completed negative EBM/spatial/threshold round without retraining."""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import GroupKFold

from src.accuracy_threshold import AccuracyThreshold
from src.additive_models import group_bags
from src.optimization import SearchRunner,config_id
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json

RUNS=["additive_study_20261029","spatial_study_20261029","threshold_study_20261029"]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=SearchRunner();dev=old.fold_of!=5
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    expected="270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    assert digest(old.output/"submission.csv")==expected
    assert digest(PROJECT_ROOT/"outputs/kaggle_ready_4690/submission_ensemble4690.csv")==expected
    ledger_path=EXPERIMENTS_DIR/"kaggle_submissions.json";ledger_digest=digest(ledger_path)
    ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["current_platform_best"]["model_run"]==old.name
    assert ledger["current_platform_best"]["score"]==.81108
    reference=np.full(len(old.y),np.nan);reference[dev]=0
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            reference[dev]+=weight/len(member["seeds"])*np.load(old.output/"search"/f"{trial}.npz")["probability"][dev]
    base_accuracy=float(accuracy_score(old.y[dev],reference[dev]>=.5))
    base_loss=float(log_loss(old.y[dev],reference[dev]))
    rows=[];audits=[]
    for run in RUNS:
        report=EXPERIMENTS_DIR/run;output=PROJECT_ROOT/"outputs"/run
        protocol=json.loads((report/"protocol.json").read_text(encoding="utf-8"))
        summary=json.loads((report/"summary.json").read_text(encoding="utf-8"))
        assert summary["status"]=="no_candidate","This audit is scoped to the completed negative round"
        assert not list(output.glob("submission*.csv"))
        assert digest(old.report/"frozen_plan.json")==protocol["source_plan_hash"]
        for path,expected_hash in protocol["sources"].items():assert digest(PROJECT_ROOT/path)==expected_hash,(run,path)
        frozen=summary["frozen"];development=frozen.get("development",frozen)
        count=0
        if not run.startswith("threshold"):
            for cache in output.glob("profile_*_seed_*.npz"):
                if "_fold_" in cache.stem:continue
                _,index,_,seed=cache.stem.split("_");seed=int(seed)
                data=np.load(cache)["probability"]
                assert np.isnan(data[~dev]).all() and np.isfinite(data[dev]).all()
                assert ((data[dev]>=0)&(data[dev]<=1)).all()
                for fold,(ti,vi) in enumerate(old.splits,1):
                    part=np.load(output/f"profile_{index}_seed_{seed}_fold_{fold}.npz")
                    np.testing.assert_array_equal(part["training"],ti);np.testing.assert_array_equal(part["validation"],vi)
                    np.testing.assert_array_equal(part["probability"],data[vi])
                    assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                    assert not (old.fold_of[ti]==5).any()
                    groups=old.features.GroupId.iloc[ti].to_numpy()
                    if run.startswith("additive"):
                        bags=group_bags(groups,seed)
                        for bag in bags.T:assert not set(groups[bag>0])&set(groups[bag<0])
                    else:
                        for a,b in GroupKFold(4,shuffle=True,random_state=20261030).split(ti,groups=groups):
                            assert not set(groups[a])&set(groups[b])
                            assert not set(ti[np.r_[a,b]])&set(vi)
                    count+=1
            predictions=np.mean([np.load(output/f"profile_{frozen['profile']}_seed_{s}.npz")["probability"] for s in frozen["seeds"]],axis=0)
            mixed=(1-frozen["weight"])*reference+frozen["weight"]*predictions
        else:
            mixed=np.load(output/f"profile_{frozen['profile']}.npz")["probability"]
            for fold,(ti,vi) in enumerate(old.splits,1):
                cached=np.load(PROJECT_ROOT/"outputs/calibration_study_20261029"/f"inner_fold_{fold}.npz")
                fit,cal=cached["fit_indices"],cached["cal_indices"]
                np.testing.assert_array_equal(np.sort(np.r_[fit,cal]),np.sort(ti))
                assert not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[cal])
                assert not set(np.r_[fit,cal])&set(vi)
                for index,config in enumerate(protocol["profiles"]):
                    threshold=AccuracyThreshold(**config).fit(old.features.iloc[cal],cached["probability"],old.y[cal])
                    expected_prediction=threshold.predict(old.features.iloc[vi],reference[vi])
                    np.testing.assert_allclose(expected_prediction,np.load(output/f"profile_{index}.npz")["probability"][vi],atol=1e-14)
                    saved=json.loads((report/f"profile_{index}_fold_{fold}_threshold.json").read_text(encoding="utf-8"))
                    assert saved==threshold.audit_
                    count+=1
        assert np.isnan(mixed[~dev]).all()
        accuracy=float(accuracy_score(old.y[dev],mixed[dev]>=.5));loss=float(log_loss(old.y[dev],mixed[dev]))
        np.testing.assert_allclose([accuracy,loss],[development["accuracy"],development["log_loss"]],atol=1e-14,rtol=0)
        rows.append(dict(run=run,accuracy=accuracy,log_loss=loss,gain=accuracy-base_accuracy,
            improved_folds=development["improved_folds"],status=summary["status"]))
        audits.append(dict(run=run,source_hashes_match=True,cached_outer_fits_or_thresholds_checked=count,
            group_disjoint=True,fold_5_excluded=True,development_metrics_reproduced=True,status=summary["status"]))
    assert digest(ledger_path)==ledger_digest
    check=dict(passed=True,date="2026-10-29",experiments=audits,new_tests=7,tests_passed=36,
        incumbent_csv_sha256=expected,observed_kaggle_best=.81108,platform_ledger_unchanged=True,
        new_submission_generated=False,baseline=dict(accuracy=base_accuracy,log_loss=base_loss),results=rows,
        limitation="Historical adaptive development; no new independent holdout or Kaggle result")
    _write_json(PROJECT_ROOT/"reports/checks/gain_exploration_20261029.json",check)
    names={RUNS[0]:"EBM加性提升",RUNS[1]:"邻近舱位统计",RUNS[2]:"训练折内阈值"}
    table="\n".join(f"| {names[r['run']]} | {r['accuracy']:.6f} | {r['log_loss']:.6f} | {r['gain']:+.6f} | {r['improved_folds']}/4 | 未通过 |" for r in rows)
    text=f"""# 增益探索实验 2026-10-29

在第四次提交0.81061之后，继续测试三个新方向，均未通过预设门槛。当前平台最佳保持0.81108，继续使用原45模型集成；本轮没有新增待提交CSV，也没有新增平台成绩。课程报告与源码ZIP继续暂停整理。

## 可比结果

表中使用原开发第1至4折合并样本的准确率和Log loss，第5折不参与训练或评分。原方案参考准确率{base_accuracy:.6f}、Log loss {base_loss:.6f}；这些数值不是全五折OOF或Kaggle分数。

| 方向 | 最佳三种子融合或阈值准确率 | Log loss | 准确率差 | 改善折数 | 结论 |
|---|---:|---:|---:|---:|---|
{table}

所有原始行已在历史开发中被查看，本轮属于追加开发。比较前固定有限配置、混合比例和晋级条件，训练完成后没有降低门槛。均在初始门槛失败，因此没有执行后续换分组复验或正式五折重训；协议中保留后续规则不代表那些步骤已经完成。

## EBM加性提升模型

固定4种配置：精简特征无交互、精简特征12项交互、消费组合特征12项交互、位置特征18项交互。每项先用种子42训练开发四折，前两项补充2026和103，共8项配置与种子记录、32次外层拟合。每次EBM包含6个内部bag，共192个内部bag拟合；内部早停按GroupId整组留出15%，没有使用外层验证标签。

最好三种子混合为原集成50%与精简特征12项交互的EBM50%。概率损失下降，但分类准确率下降，所以不生成新提交。EBM自然处理缺失值；库中关于缺失值图形显示的提示不表示训练失败。

本轮可选依赖为interpret-core==0.7.8，单独记录在requirements-ebm.txt。原requirements.txt及历史模型源码保持不变。方法参考：[InterpretML EBM文档](https://interpret.ml/docs/ebm.html)、[分组bag参数](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)。

## 邻近舱位目标统计

历史训练数据中没有跨同行组的同舱样本，因此没有将同舱目标统计当作新信号。改为在相同Deck、Side及原始CryoSleep类别内，计算舱号距离10、50、200以内的平滑目标均值和支持数量。先验及所有标签统计仅来自当前训练部分；未知区域或缺失舱号回退到训练先验。

训练特征采用4折GroupKFold交叉拟合，自己的整个同行组标签不进入自己的编码；验证特征只查询外层训练部分。测试标签从未读取。额外测试逐组翻转标签，确认该组自身编码完全不变。支持数量在内层训练和外层预测之间仍有样本量差异，因此它是受验证的编码设计，不能仅凭无泄漏就认定有效。

LightGBM和CatBoost各设置有/无邻近统计两组，共4项初始配置，前两项补齐三种子，共32次外层模型拟合。最好三种子融合含邻近统计LightGBM50%，但未超过原集成。不把模型参数或基础特征的变化全部归因于邻近统计。

## 嵌套准确率阈值

复用此前每个外层训练部分内部的25%整组留出预测。临时教师训练时完全排除这些校准组，外层验证组及第5折也不在其中。阈值只在这些训练内部样本上学习。

固定阈值网格0.35至0.65、步长0.005；按GroupId做150次bootstrap，各次选择准确率最高且尽量接近0.5的阈值，取中位数，再将阈值logit按0.25、0.5、1.0收缩。模型最终仍对变换后的概率使用0.5，未查看榜分选阈值。共3种配置、12个外层阈值变换。最好配置的准确率仍低于原固定0.5方案。

不同开发折学到的原始阈值为0.500、0.445、0.565、0.580，方向不稳定；这是本轮不采用阈值调整的证据，不能据此指定一个统一最优阈值。

## 核验与复现

36项测试通过，其中7项新增测试覆盖内部整组隔离、邻近统计的自身标签排除、未知区域回退及阈值变换。独立核验重建了开发指标、源码哈希、分组隔离和阈值结果；原最佳CSV哈希及四次平台成绩账本保持不变。详见reports/checks/gain_exploration_20261029.json。

```powershell
python -m pip install -r requirements-ebm.txt
python -m unittest discover -s tests -q
python -m scripts.additive_study
python -m scripts.spatial_study
python -m scripts.threshold_study
python -m scripts.audit_gain_exploration
```

上述命令用于已有本机缓存的增量复现。本轮依赖原冻结方案、confirmation阶段的三种子开发概率及嵌套校准缓存；全新克隆不附带这些缓存，不能直接运行上述追加实验。需按原冻结成员配置重建开发概率，再运行scripts.calibration_study生成内部留出预测；已有frozen_plan.json时，仅运行freeze会直接返回计划，不会补齐缓存。原流程见[成员1模型优化与复现](成员1_模型优化与复现.md)。仅复现现有平台最佳可使用scripts.refit_frozen，无需重做本轮。models和outputs按.gitignore留在本机，GitHub保存源码、冻结协议及实验结果。
"""
    (PROJECT_ROOT/"docs/增益探索实验_20261029.md").write_text(text,encoding="utf-8")
    block="""<!-- gain-exploration-20261029:start -->
## 2026-10-29 新方向探索

追加完成EBM、按组交叉拟合的邻近舱位目标统计及嵌套准确率阈值实验；本地准确率分别为0.819986、0.820417、0.821280，均未达到原开发口径0.821999。36项测试和来源/分组核验通过。本轮没有新增提交文件，平台最佳仍为原集成0.81108。

详见[增益探索实验](docs/增益探索实验_20261029.md)。课程报告与源码ZIP继续暂停整理。
<!-- gain-exploration-20261029:end -->
"""
    for path in [PROJECT_ROOT/"README.md",PROJECT_ROOT/"docs/项目进度.md"]:
        text=path.read_text(encoding="utf-8")
        text=re.sub(r"<!-- gain-exploration-20261029:start -->.*?<!-- gain-exploration-20261029:end -->\n*","",text,flags=re.S)
        first,rest=text.split("\n",1)
        inserted=block if path.name=="README.md" else block.replace("(docs/增益探索实验_20261029.md)","(增益探索实验_20261029.md)")
        path.write_text(first+"\n\n"+inserted+"\n"+rest.lstrip(),encoding="utf-8")
    print(json.dumps(check,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
