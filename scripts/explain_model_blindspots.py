"""Follow-up interpretation of the frozen OOF audit; no feature/model fitting."""
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import preprocess as pp
from src.competition_features import build_pool
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR,FIGURES_DIR
from src.train import _write_csv,_write_json

NAME="error_audit_20261029"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report=EXPERIMENTS_DIR/NAME;sub=report/"mechanisms";sub.mkdir(exist_ok=True)
    summary=json.loads((report/"summary.json").read_text(encoding="utf-8"))
    coverage=json.loads((report/"feature_coverage.json").read_text(encoding="utf-8"))
    a=pd.read_csv(report/"passenger_diagnostics.csv",dtype={"PassengerId":str,"GroupId":str},float_precision="round_trip")
    raw,test=pp.load_raw();d,t=build_pool(raw,test)
    np.testing.assert_array_equal(a.PassengerId,raw.PassengerId)
    public=pd.concat([d.drop(columns=pp.TARGET),t],ignore_index=True)
    y=raw.Transported.to_numpy(int)
    ps=[a.probability.to_numpy()];components=[a[["cat_spend_probability","cat_names_probability","lgb_probability"]].to_numpy()]
    for repeat in [1,2]:
        source=EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029"/f"repeat_{repeat}_oof.csv"
        p=pd.read_csv(source,dtype={"PassengerId":str},float_precision="round_trip")
        np.testing.assert_array_equal(p.PassengerId,raw.PassengerId);ps.append(p.reference.to_numpy())
        component=np.full((len(y),3),np.nan)
        for fold in range(1,6):
            data=np.load(PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029"/f"repeat_{repeat}_fold_{fold}.npz")
            component[data["validation"]]=data["probability"]
        np.testing.assert_allclose(component@np.array([.25,.25,.5]),ps[-1],atol=1e-14)
        components.append(component)
    probability=np.column_stack(ps)
    cohorts=[("Earth",lambda f:f.HomePlanet.eq("Earth")),
        ("Earth / G",lambda f:f.HomePlanet.eq("Earth")&f.Deck.eq("G")),
        ("Earth / Cryo=True",lambda f:f.HomePlanet.eq("Earth")&f.CryoSleep.eq(True)),
        ("Earth / G / child / awake",lambda f:f.HomePlanet.eq("Earth")&f.Deck.eq("G")&f.Age.le(12)&f.CryoSleep.eq(False)),
        ("Europa / Cryo=True",lambda f:f.HomePlanet.eq("Europa")&f.CryoSleep.eq(True)),
        ("Mars / Cryo=True",lambda f:f.HomePlanet.eq("Mars")&f.CryoSleep.eq(True))]
    protocol=dict(purpose="Exploratory follow-up cohorts chosen after first-stage OOF inspection, not confirmatory inference",
        cohort_names=[name for name,_ in cohorts],
        source_hash=sha(PROJECT_ROOT/"scripts/explain_model_blindspots.py"),
        base_audit_protocol_sha256=sha(report/"protocol.json"),
        input_predictions_sha256=sha(report/"passenger_diagnostics.csv"),
        new_candidate_features=["OtherKnownCryoShare","OtherAdultMeanAge","OtherChildCount"],
        no_fitting=True,no_new_submission=True,
        caveat="High error, a feature absent by name, and residual association do not prove missing causal information or an achievable gain. Historical repeated OOF rows are dependent.")
    path=sub/"protocol.json"
    if path.exists():assert json.loads(path.read_text(encoding="utf-8"))==protocol
    else:_write_json(path,protocol)
    rows=[]
    for name,condition in cohorts:
        mask=condition(d).fillna(False).to_numpy(bool);test_mask=condition(t).fillna(False).to_numpy(bool)
        p=probability[mask];actual=y[mask]
        rows.append(dict(cohort=name,n=int(mask.sum()),groups=int(d.GroupId[mask].nunique()),test_n=int(test_mask.sum()),
            positive_rate=float(actual.mean()),mean_probability=float(p[:,0].mean()),
            predicted_positive_rate=float((p[:,0]>=.5).mean()),
            errors=int(((p[:,0]>=.5)!=actual).sum()),error_rate=float(((p[:,0]>=.5)!=actual).mean()),
            repeat_error_rates=[float(((p[:,c]>=.5)!=actual).mean()) for c in [1,2]],
            roc_auc=float(roc_auc_score(actual,p[:,0])) if len(np.unique(actual))==2 else None,
            minority_rows=int(min(actual.sum(),len(actual)-actual.sum())),
            false_positive=int(((p[:,0]>=.5)&(actual==0)).sum()),false_negative=int(((p[:,0]<.5)&(actual==1)).sum())))
    _write_json(sub/"cohort_metrics.json",rows);_write_csv(pd.DataFrame(rows).drop(columns="repeat_error_rates"),sub/"cohort_metrics.csv")
    # Only public feature aggregation, excluding the query passenger's own value.
    group=public.GroupId
    known=public.CryoSleep.notna().astype(int);true=public.CryoSleep.eq(True).fillna(False).astype(int)
    other_known=known.groupby(group).transform("sum")-known
    other_true=true.groupby(group).transform("sum")-true
    adult=public.Age.ge(18).astype(int);adult_age=public.Age.where(adult.eq(1),0.).fillna(0.)
    other_adults=adult.groupby(group).transform("sum")-adult
    other_age=adult_age.groupby(group).transform("sum")-adult_age
    child=public.Age.le(12).astype(int)
    candidate=pd.DataFrame(dict(OtherKnownCryoShare=other_true/other_known.replace(0,np.nan),
        OtherAdultMeanAge=other_age/other_adults.replace(0,np.nan),
        OtherChildCount=child.groupby(group).transform("sum")-child)).iloc[:len(d)].reset_index(drop=True)
    categories=pd.DataFrame(index=d.index)
    categories["OtherKnownCryoShare"]=pd.cut(candidate.OtherKnownCryoShare,[-.01,0,.5,.999,1.],
        labels=["0","(0,.5]","(.5,1)","1"]).astype("string").fillna("No known other")
    categories["OtherAdultMeanAge"]=pd.cut(candidate.OtherAdultMeanAge,[-np.inf,25,40,np.inf],
        labels=["<=25","26-40",">40"]).astype("string").fillna("No known other adult")
    categories["OtherChildCount"]=pd.cut(candidate.OtherChildCount,[-1,0,1,np.inf],labels=["0","1","2+"]).astype("string")
    focus=(d.HomePlanet.eq("Earth")&d.Deck.eq("G")&d.CryoSleep.eq(True)).fillna(False).to_numpy(bool)
    context=[]
    for col in categories:
        for value in sorted(categories.loc[focus,col].unique()):
            mask=focus&categories[col].eq(value).to_numpy(bool)
            context.append(dict(feature=col,value=value,n=int(mask.sum()),groups=int(d.GroupId[mask].nunique()),
                positive_rate=float(y[mask].mean()),mean_probability=float(probability[mask,0].mean()),
                probability_bias=[float(probability[mask,c].mean()-y[mask].mean()) for c in range(3)],
                error_rate=float(((probability[mask,0]>=.5)!=y[mask]).mean()),
                explicitly_present_in_primary_inputs=any(col in r["columns"] for r in coverage)))
    _write_json(sub/"candidate_context_descriptions.json",context)
    mars_e=(d.HomePlanet.eq("Mars")&d.Deck.eq("E")).fillna(False).to_numpy(bool)
    crossover=[]
    for c,component in enumerate(components):
        acc=((component[mars_e]>=.5)==y[mars_e,None]).mean(axis=0)
        ensemble=float(((probability[mars_e,c]>=.5)==y[mars_e]).mean())
        crossover.append(dict(variant=c,n=int(mars_e.sum()),cat_spend_accuracy=float(acc[0]),
            cat_names_accuracy=float(acc[1]),lgb_accuracy=float(acc[2]),ensemble_accuracy=ensemble,
            retrospective_lgb_instead_of_ensemble_net_correct=int(round((acc[2]-ensemble)*mars_e.sum()))))
    _write_json(sub/"mars_e_component_comparison.json",crossover)
    confidence=pd.read_csv(report/"confidence_errors.csv")
    boundary_errors=int(confidence.loc[confidence.index[:2],"errors"].sum())
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    display=[rows[i] for i in [0,1,2,3,4,5]]
    labels=[r["cohort"] for r in display];rates=[r["error_rate"]*100 for r in display]
    bars=axes[0].barh(labels,rates,color=["#dc624f"]*4+["#508c9c"]*2)
    axes[0].invert_yaxis();axes[0].set_xlim(0,48);axes[0].set_xlabel("OOF error rate (%)")
    axes[0].axvline((1-summary["canonical_accuracy"])*100,color="#34495e",linestyle="--",label="Overall 18.43%")
    for bar,row in zip(bars,display):axes[0].text(bar.get_width()+.5,bar.get_y()+bar.get_height()/2,f"{row['error_rate']*100:.1f}% (n={row['n']})",va="center",fontsize=8)
    axes[0].legend(loc="lower right",fontsize=8);axes[0].set_title("Error concentration (overlapping cohorts)")
    comp=summary["complementarity"]
    axes[1].bar(["All 3 components wrong","Some component correct"],
        [comp["wrong_with_all_components_wrong"],comp["wrong_with_some_component_correct"]],color=["#dc624f","#508c9c"])
    axes[1].set_ylabel("Canonical ensemble errors");axes[1].set_ylim(0,1450)
    for i,v in enumerate([comp["wrong_with_all_components_wrong"],comp["wrong_with_some_component_correct"]]):axes[1].text(i,v+20,str(v),ha="center")
    axes[1].set_title("Shared mistakes dominate")
    fig.suptitle("Frozen ensemble blindspot audit: historical grouped OOF",fontsize=13)
    figure=FIGURES_DIR/"model_blindspots_20261029.png";fig.savefig(figure,dpi=170);plt.close(fig)
    shared=comp["wrong_with_all_components_wrong"]
    table="\n".join(f"| {r['cohort']} | {r['n']} | {r['errors']} | {r['error_rate']:.1%} | {r['test_n']} |" for r in rows)
    text=f"""# 现有模型的薄弱点 2026-10-29

已逐折重载原45个模型，重现8693名训练乘客的折外预测，并核对另外两次分组五折。发现了错误集中的群体和具体表征缺口；尚未证明添加某个字段就能提分。本轮只做诊断，没有训练新模型或生成提交，Kaggle最好仍为0.81108。

## 主要问题集中在哪

原集成全五折OOF准确率为{summary['canonical_accuracy']:.6f}，错1602人。Earth乘客占4602/8693人，却贡献1188/1602个错误（74.2%）；Earth/G甲板占2498人，贡献719个错误（44.9%），错误率28.8%。后者在两次换分组中错误率约29.5%和29.7%，不像单次划分造成的孤立现象。

| 观察群体 | 训练人数 | 原OOF错误 | 错误率 | 测试特征中人数 |
|---|---:|---:|---:|---:|
{table}

群体相互重叠，人数和错误数不能相加。测试列只统计公开特征中的覆盖人数，不知道这些测试乘客是否预测错误。所有分数都是历史OOF描述，不能当作新盲测或Kaggle分数。

![模型错误集中与成员共同错误](../reports/figures/model_blindspots_20261029.png)

## 模型漏掉的是群体内部区分，而不只是整体概率

冷冻乘客在不同星球的结果差别很大：Earth实际传送比例65.63%，Europa为98.90%，Mars为91.18%。不能把“冷冻就几乎必传送”当作通用规则。

但原模型并没有完全漏学这一点：Earth冷冻乘客的平均预测概率65.98%，已经接近实际65.63%；其群体内部ROC AUC约0.674，说明更突出的是个体排序较弱。平均数接近不代表所有子群体都已校准，也不能证明不存在其他可学信号。

Earth冷冻乘客有91.3%被判为True，高于实际65.6%，这本身不构成预测错误的规则：若大部分个体概率都大于0.5，准确率最优的分类器可以把多数人判为True。强行把预测正例比例压到65.6%，可能反而降低准确率。

具体还应检查Earth/G甲板中未冷冻的儿童：258人错99人，错误率38.4%。他们的五项消费通常缺少区分度，需要更清楚地表达同行组内部的年龄和冷冻角色，而不只是再堆消费函数。

## 能明确核实的输入缺口

原集成50%权重的LightGBM基础方案，以及25%权重的消费CatBoost，都没有输入连续CabinNum、GroupIdNum及完整同行年龄/消费汇总；这些主要集中在另外25%的姓名丰富CatBoost中。原方案已有GroupSize、GroupPos、GroupSurnameCount，不能说完全没有同行信息。

三个候选表征在当前输入中没有直接出现：同组其他已知乘客的冷冻比例、同组其他成年人的平均年龄、同组其他儿童人数。本次只用公开无标签特征计算它们，并将自身值排除；描述性条件统计保存在mechanisms/candidate_context_descriptions.json。丰富方案已有GroupMeanAge、GroupMinAge、GroupMaxAge等相关信息，因此“字段名不存在”不等于“具有独立新信息”。

这提供了下一轮可检验的假设：在Earth/G及儿童群体中，明确编码同行角色、舱位分布与年龄的交互。需要在同折消融中验证增量，不能直接写成已经找到可提升成绩的特征。已有G/E局部模型、全局关系编码及物理消费特征的失败记录仍保留，不将这些旧方向改名宣称为新发现。

## 为什么单纯融合越来越难

1602个错误中，有{shared}个（{shared/1602:.1%}）是三个成员类型共同犯的；只有349个错误存在某成员已经分对的情况。这349只是事后查看标签得到的可互补数量，不能把它当成可实现的提分上限或无泄漏的模型选择规则。

原始五折和两次换分组中都错的乘客有1364人，占原错误的85.1%；在三份预测中真实类别概率都不超过0.1的有74人。原OOF中有{boundary_errors}个错误位于预测概率0.35至0.65之间，占错误的{boundary_errors/1602:.1%}。大部分困难不能解释为少量极端离群点，但也不能据此断言这些错误不可约。

LightGBM三个种子的预测实际上相同，换种子没有增加这部分的模型多样性；这不改变其总权重，也不表示去重就会自动提分。

Mars/E甲板是一个可观察到的成员分工线索：330名训练乘客中，LightGBM在三种划分下都比姓名CatBoost更好。不过原五折若事后直接用LightGBM替代该群体的集成，也仅净多对3人，不能为此宣称全局稳定收益。条件融合的优先级低于寻找新判别信息。

## 已排除的简单解释

完整无缺失的6606人仍有18.07%错误；1项缺失为19.34%，2项及以上为21.36%。缺失样本稍难，但不能解释主要错误总量，也没有理由推翻此前“不在填补方法上过度投入”的结论。

舱号模2/3/4/5的探索未显示明显一致的交替规律；没有把偶然小分组差异当成新特征证据。对明显规律相反的74个高置信持久错误，已保存逐条清单，但没有外部证据证明标签有误，不改标签、不按验证答案制作规则。

## 证据与下一步

主要证据在reports/experiments/error_audit_20261029：slice_metrics.csv保留全部预定义群体；error_cases.csv保留1602个原OOF错误；confident_persistent_errors.csv保留74个高置信持久错误；feature_coverage.json保留实际模型输入。前12个高风险群体按GroupId做1000次条件bootstrap，但区间未经群体筛选或多重比较校正，不是因果或新增独立显著性证据。

下一轮应只测试少量同行角色和位置交互，优先观察Earth/G与儿童的错误是否下降、其余人群是否退化；保留同一分组基准和换分组复验，不根据测试集标签或榜分选择规则。当前只是明确了值得验证的薄弱环节，没有承诺突破0.82。

复现诊断需本机原模型和概率缓存：`python -m scripts.audit_model_blindspots`，再运行`python -m scripts.explain_model_blindspots`。该过程不拟合任何分类器。课程报告与源码ZIP继续暂停整理。
"""
    (PROJECT_ROOT/"docs/模型薄弱点诊断_20261029.md").write_text(text,encoding="utf-8")
    block="""<!-- model-blindspots-20261029:start -->
## 2026-10-29 模型薄弱点诊断

核对原45模型的全五折OOF及两次换分组预测。Earth/G甲板贡献719/1602个错误；Earth冷冻乘客平均概率已接近真实比例，主要问题是内部区分。1253个错误为三个成员类型共同犯错。提出同组其他乘客冷冻比例、成年年龄及儿童角色的待验证表征，尚无提分结论。

详见[模型薄弱点诊断](docs/模型薄弱点诊断_20261029.md)。本轮仅诊断，无新模型或提交；平台最佳仍为0.81108。
<!-- model-blindspots-20261029:end -->
"""
    for path in [PROJECT_ROOT/"README.md",PROJECT_ROOT/"docs/项目进度.md"]:
        text=path.read_text(encoding="utf-8")
        text=re.sub(r"<!-- model-blindspots-20261029:start -->.*?<!-- model-blindspots-20261029:end -->\n*","",text,flags=re.S)
        first,rest=text.split("\n",1)
        inserted=block if path.name=="README.md" else block.replace("(docs/模型薄弱点诊断_20261029.md)","(模型薄弱点诊断_20261029.md)")
        path.write_text(first+"\n\n"+inserted+"\n"+rest.lstrip(),encoding="utf-8")
    _write_json(PROJECT_ROOT/"reports/checks/blindspot_mechanisms_20261029.json",dict(passed=True,
        candidate_context_uses_unlabeled_features_only=True,own_values_excluded=True,
        repeated_components_reproduce_reference=True,analysis_only=True,cohort_count=len(rows),
        limitations=protocol["caveat"]))
    print(json.dumps(dict(cohorts=rows,candidate_context=context,mars_e_comparison=crossover),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
