"""Recompute all three completed G-targeted comparisons without training."""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss

from src import preprocess as pp
from src.g_deck_models import blend_g,g_mask
from src.optimization import SearchRunner,config_id
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv

RUNS=["g_deck_roles_20261029_v2","g_deck_transfer_20261029","g_deck_offset_20261029"]
NAMES=["只用G训练的专用模型","全量训练并提高G权重","原概率加同行角色修正"]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=SearchRunner();dev=old.fold_of!=5;g=g_mask(old.features)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    ledger=EXPERIMENTS_DIR/"kaggle_submissions.json";ledger_hash=sha(ledger)
    expected="270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82"
    assert sha(old.output/"submission.csv")==expected
    assert sha(PROJECT_ROOT/"outputs/kaggle_ready_4690/submission_ensemble4690.csv")==expected
    reference=np.full(len(old.y),np.nan);reference[dev]=0
    for weight,member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            reference[dev]+=weight/len(member["seeds"])*np.load(old.output/"search"/f"{trial}.npz")["probability"][dev]
    public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
    assert pp.TARGET not in public
    rows=[];audits=[];subgroups=[];alternatives=[];flips=[]
    for run,name in zip(RUNS,NAMES):
        report=EXPERIMENTS_DIR/run;output=PROJECT_ROOT/"outputs"/run
        protocol=json.loads((report/"protocol.json").read_text(encoding="utf-8"))
        summary=json.loads((report/"summary.json").read_text(encoding="utf-8"))
        assert summary["status"]=="no_candidate","Candidate needs a separate full refit/prediction audit"
        assert not list(output.glob("submission*.csv"))
        for path,digest in protocol["sources"].items():assert sha(PROJECT_ROOT/path)==digest,(run,path)
        assert sha(old.report/"frozen_plan.json")==protocol["source_plan_hash"]
        frozen=summary["frozen"];count=0
        if run!=RUNS[2]:
            for part in output.glob("profile_*_seed_*_fold_*.npz"):
                data=np.load(part);ti,vi=data["training"],data["validation"]
                fold=int(part.stem.split("_")[-1]);expected_ti,expected_vi=old.splits[fold-1]
                np.testing.assert_array_equal(ti,expected_ti);np.testing.assert_array_equal(vi,expected_vi)
                assert not set(old.features.GroupId.iloc[ti])&set(old.features.GroupId.iloc[vi])
                assert not (old.fold_of[ti]==5).any()
                p=data["probability"]
                assert np.isfinite(p[g[vi]]).all() and np.isnan(p[~g[vi]]).all();count+=1
            index=frozen["profile"];control=frozen["control_profile"]
            p=np.mean([np.load(output/f"profile_{index}_seed_{s}.npz")["probability"] for s in frozen["seeds"]],axis=0)
            c=np.mean([np.load(output/f"profile_{control}_seed_{s}.npz")["probability"] for s in frozen["seeds"]],axis=0)
            assert np.isnan(p[~(dev&g)]).all()
            alternative=reference.copy();control_probability=reference.copy()
            alternative[dev]=blend_g(reference[dev],p[dev],old.features.loc[dev],frozen["weight"])
            control_probability[dev]=blend_g(reference[dev],c[dev],old.features.loc[dev],frozen["weight"])
        else:
            from src.g_offset_models import GOffsetCalibrator,corrected_probability,PROFILES
            p=np.load(output/f"profile_{frozen['profile']}.npz")["probability"]
            c=np.load(output/f"profile_{frozen['control_profile']}.npz")["probability"]
            for fold,(ti,vi) in enumerate(old.splits,1):
                source=PROJECT_ROOT/"outputs/calibration_study_20261029"/f"inner_fold_{fold}.npz"
                z=np.load(source);fit,cal=z["fit_indices"],z["cal_indices"]
                np.testing.assert_array_equal(np.sort(np.r_[fit,cal]),np.sort(ti))
                assert not set(fit)&set(cal) and not set(cal)&set(vi)
                assert not set(old.features.GroupId.iloc[fit])&set(old.features.GroupId.iloc[cal])
                for index,profile in enumerate(PROFILES):
                    estimator=GOffsetCalibrator(**profile).fit(old.features.iloc[cal],z["probability"],old.y[cal],public)
                    np.testing.assert_allclose(estimator.predict(old.features.iloc[vi],reference[vi]),
                        np.load(output/f"profile_{index}.npz")["probability"][vi],atol=1e-14,rtol=0)
                    saved=json.loads((report/f"profile_{index}_fold_{fold}.json").read_text(encoding="utf-8"))
                    assert saved["source_cache_sha256"]==sha(source)
                    count+=1
            alternative=reference.copy();control_probability=reference.copy()
            alternative[dev&g]=(1-frozen["weight"])*reference[dev&g]+frozen["weight"]*p[dev&g]
            control_probability[dev&g]=(1-frozen["weight"])*reference[dev&g]+frozen["weight"]*c[dev&g]
        np.testing.assert_array_equal(alternative[dev&~g],reference[dev&~g])
        d=frozen["development"]
        for actual,recorded in [(accuracy_score(old.y[dev],alternative[dev]>=.5),d["accuracy"]),
            (accuracy_score(old.y[dev&g],alternative[dev&g]>=.5),d["g_accuracy"]),
            (log_loss(old.y[dev],alternative[dev]),d["log_loss"]),
            (accuracy_score(old.y[dev&g],control_probability[dev&g]>=.5),frozen["matched_control"]["g_accuracy"])]:
            np.testing.assert_allclose(actual,recorded,atol=1e-14,rtol=0)
        before=reference>=.5;after=alternative>=.5;correct=before==old.y;new_correct=after==old.y
        fixed=int((dev&~correct&new_correct).sum());regressed=int((dev&correct&~new_correct).sum())
        rows.append(dict(run=run,name=name,accuracy=d["accuracy"],g_accuracy=d["g_accuracy"],gain=d["gain"],
            g_gain=d["g_gain"],g_log_loss=d["g_log_loss"],matched_control_g_accuracy=frozen["matched_control"]["g_accuracy"],
            role_added_g_gain=frozen["added_roles_g_gain"],g_improved_folds=d["g_improved_folds"],
            corrected=fixed,new_errors=regressed,net_correct=fixed-regressed,status="no_candidate"))
        audits.append(dict(run=run,source_hashes_match=True,outer_fits_or_nested_offsets_checked=count,
            development_metrics_reproduced=True,non_g_probabilities_unchanged=True,fold_5_excluded=True))
        for label,mask in [("全部开发样本",dev),("G甲板",dev&g),("非G甲板",dev&~g),
            ("Earth/G冷冻",dev&g&old.features.HomePlanet.eq("Earth").to_numpy()&old.features.CryoSleep.eq(True).fillna(False).to_numpy()),
            ("Earth/G未冷冻儿童",dev&g&old.features.HomePlanet.eq("Earth").to_numpy()&old.features.Age.le(12).to_numpy()&old.features.CryoSleep.eq(False).fillna(False).to_numpy())]:
            subgroups.append(dict(run=run,cohort=label,n=int(mask.sum()),
                reference_accuracy=float(accuracy_score(old.y[mask],before[mask])),alternative_accuracy=float(accuracy_score(old.y[mask],after[mask])),
                net_correct=int((new_correct[mask].sum()-correct[mask].sum()))))
        changed=dev&(before!=after)
        for i in np.flatnonzero(changed):
            flips.append(dict(run=run,PassengerId=old.raw.PassengerId.iloc[i],fold=int(old.fold_of[i]),
                Transported=int(old.y[i]),reference_probability=float(reference[i]),alternative_probability=float(alternative[i]),
                corrected=bool(new_correct[i]),Deck=str(old.features.Deck.iloc[i])))
        alternatives.append(alternative)
    aborted=EXPERIMENTS_DIR/"g_deck_roles_20261029"
    old_protocol=json.loads((aborted/"protocol.json").read_text(encoding="utf-8"))
    for path in ["src/g_deck_models.py","scripts/g_deck_study.py"]:
        assert sha(aborted/"aborted_sources"/path.split("/")[-1])==old_protocol["sources"][path]
    assert sha(ledger)==ledger_hash
    root_report=EXPERIMENTS_DIR/"g_deck_treatment_20261029";root_report.mkdir(exist_ok=True)
    _write_csv(pd.DataFrame(rows),root_report/"comparison.csv")
    _write_csv(pd.DataFrame(subgroups),root_report/"subgroup_comparison.csv")
    _write_csv(pd.DataFrame(flips),root_report/"development_prediction_changes.csv")
    _write_json(root_report/"summary.json",dict(status="no_candidate",results=rows,
        observed_kaggle_best=.81108,no_new_submission=True,historical_adaptive_development=True))
    check=dict(passed=True,experiments=audits,tests_passed=44,original_submission_sha256=expected,
        aborted_source_snapshots_verified=True,platform_ledger_unchanged=True,
        no_new_submission=True,no_five_fold_refit=True,no_repeat_validation_run=True,
        limitation="All original rows historically inspected; weak positive offset result is selected development evidence only")
    _write_json(PROJECT_ROOT/"reports/checks/g_deck_treatment_20261029.json",check)
    first=rows[0];transfer=rows[1];offset=rows[2]
    table="\n".join(f"| {r['name']} | {r['accuracy']:.6f} | {r['g_accuracy']:.6f} | {r['net_correct']:+d} | {r['g_improved_folds']}/4 |" for r in rows)
    subgroup_table="\n".join(f"| {r['cohort']} | {r['n']} | {r['reference_accuracy']:.2%} | {r['alternative_accuracy']:.2%} | {r['net_correct']:+d} |" for r in subgroups if r['run']==RUNS[2])
    text=f"""# G甲板定向处理实验 2026-10-29

本轮检验同行角色是否能改善G甲板。专用模型与G加权的全量模型未超过原集成；保留原概率的正则化修正得到小幅开发线索，但未达到事前门槛。尚未证明解决了G甲板问题，没有生成新的Kaggle提交CSV，平台最佳仍为0.81108。

## 同一开发口径的结果

本轮沿用开发第1至4折：共{int(dev.sum())}人，其中原始已知Deck=G的{int((dev&g).sum())}人。第5折完全不参与本轮训练和评分。原开发整体准确率{accuracy_score(old.y[dev],reference[dev]>=.5):.6f}，G甲板准确率{accuracy_score(old.y[dev&g],reference[dev&g]>=.5):.6f}。此前诊断的全五折G甲板共2559人，口径不同，不能直接混比。

| 方法 | 整体准确率 | G准确率 | 净多对人数 | G改善折数 |
|---|---:|---:|---:|---:|
{table}

所有方法都只改G的预测，非G开发样本的概率逐个完全不变；没有将缺失Deck按同行组强行填成G。阈值始终0.5，也未使用Kaggle榜分选择权重或规则。

## 新增的同行角色特征

从公开train+test无标签特征统一计算：同组其他已知乘客冷冻比例与支持人数、其他成年人平均年龄与人数、其他儿童人数、自身与其他成年人的年龄差，以及儿童/成年同行角色。每项都排除自身贡献，不读取同组Transported，不要求同行者全部来自训练集。

保留原冻结模型与历史源码接口，新增GDeckBuilder。新特征在单人和批量transform时保持一致，避免只看当前小批次导致同行信息丢失。未知或缺少其他同行者时保留未知状态，不把未知比例当成确定事实。

## 三种处理方式与消融

第一种只在外层训练部分的G乘客上训练LightGBM或CatBoost；每个家族各有不加角色和加入角色的匹配对照。先固定4种配置用种子42训练开发四折，再对初始最优的增强家族及其无角色对照补齐2026和103，共32次外层拟合。最好加入角色的混合相对无角色对照仅净多对3人，仍比原方案少对6人。

第二种用全部允许的外层训练样本学习共享关系，G样本权重固定2，其余为1，但预测时仅对G路由修正。仍为4项初始配置和匹配三种子对照，共32次外层拟合；新增角色相对同权重对照净多对1人，仍比原方案少对5人。这没有支持“仅靠扩大训练样本就解决G问题”的解释。

第三种保留原模型的logit概率，系数固定为1，只学习带L2惩罚的小幅修正。控制组使用CryoSleep、Side及截距；增强组增加同行角色数值和CryoSleep交互。固定两种惩罚强度0.01/0.1，混合比例0.25/0.5/1，共4种配置、12个组合；16个外层修正器均只拟合外层训练内部25%整组留出的G样本。临时教师完全排除这些校准组，外层验证标签不参与拟合。

第三种最好配置整体净多对{offset['net_correct']}人：修正{offset['corrected']}个旧错误，同时新增{offset['new_errors']}个错误。G准确率由71.87%到72.11%，仅两折改善；相对不含角色的同配置修正净多对6人。它是弱的、选择后的开发线索，不能直接当作真实提分。

## 概率修正对关键子群体的影响

| 群体 | 人数 | 原准确率 | 修正后准确率 | 净多对 |
|---|---:|---:|---:|---:|
{subgroup_table}

群体相互重叠，净多对人数不能相加。子群体用于解释结果，没有根据子群体验证答案另选阈值。变化逐人记录在g_deck_treatment_20261029/development_prediction_changes.csv中。

## 为什么没有生成新提交

训练前固定：整体准确率至少+0.0015、G至少+0.005、G的4折至少3折改善，且加入角色相对匹配对照的G增益至少+0.003；同时限制整体及G的Log loss恶化。概率修正仅整体+0.000719、G+0.002476、2/4折改善，因此未通过。没有因为结果接近门槛就放宽规则。

三个版本都未通过初始门槛，所以没有执行协议中预留的两次换分组复验，也没有正式五折重训或生成新CSV。原45模型与0.81108提交继续保留。不能把此前历史数据上所有人都被看过的开发过程称为独立盲测。

## 修复与核验

第一次尝试在新GroupAgeRole类别字段处遇到Pandas 3字符串类型推断问题，训练中断；原协议、源码快照和唯一已完成控制记录保存在g_deck_roles_20261029，不计入正式效果比较。修正后在全新v2目录重训，未混用中断缓存；新增回归测试确认编码器将该字段识别为类别。

44项测试通过，包括自身排除、标签翻转不影响特征、单人/批量一致、非G预测不变、训练权重、概率修正梯度与非G标签不影响修正器等8项新增测试。独立核验重新计算全部开发指标、分组隔离与概率修正；源码和原最佳CSV哈希一致，成绩账本未变化。详见reports/checks/g_deck_treatment_20261029.json。

已有本机缓存时可复现：

```powershell
python -m unittest discover -s tests -q
python -m scripts.g_deck_study
python -m scripts.g_deck_transfer_study
python -m scripts.g_offset_study
python -m scripts.audit_g_deck_studies
```

全新克隆还需要原冻结方案的confirmation开发缓存及scripts.calibration_study生成的内部教师缓存；models与outputs继续忽略。课程报告与源码ZIP按用户要求暂停整理。
"""
    (PROJECT_ROOT/"docs/G甲板定向实验_20261029.md").write_text(text,encoding="utf-8")
    block="""<!-- g-deck-treatment-20261029:start -->
## 2026-10-29 G甲板定向处理

完成G专用模型、G加权的全量训练模型及嵌套概率修正，均含同行角色消融。前两种未改善原方案；概率修正使2019名开发G乘客净多对5人，准确率71.87%到72.11%，但只在2/4折改善，未通过预设门槛。非G预测保持不变，44项测试通过，无新增提交文件，平台最佳仍为0.81108。

详见[G甲板定向实验](docs/G甲板定向实验_20261029.md)。课程报告和源码ZIP继续暂停整理。
<!-- g-deck-treatment-20261029:end -->
"""
    for path in [PROJECT_ROOT/"README.md",PROJECT_ROOT/"docs/项目进度.md"]:
        text=path.read_text(encoding="utf-8")
        text=re.sub(r"<!-- g-deck-treatment-20261029:start -->.*?<!-- g-deck-treatment-20261029:end -->\n*","",text,flags=re.S)
        first,rest=text.split("\n",1)
        inserted=block if path.name=="README.md" else block.replace("(docs/G甲板定向实验_20261029.md)","(G甲板定向实验_20261029.md)")
        path.write_text(first+"\n\n"+inserted+"\n"+rest.lstrip(),encoding="utf-8")
    print(json.dumps(dict(check=check,results=rows,subgroups=subgroups),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
