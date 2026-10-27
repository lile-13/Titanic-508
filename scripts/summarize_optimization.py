"""从已完成实验生成结果报告与图表，不重新选择模型。"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import EXPERIMENTS_DIR, PROJECT_ROOT
from src.train import _write_csv, _write_json


def summarize(name):
    directory = EXPERIMENTS_DIR / name
    result = json.loads((directory / "final_summary.json").read_text(encoding="utf-8"))
    plan, confirmation = result["plan"], result["confirmation"]
    records = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("trial_*.json")]
    tune_counts = {family: sum(r["stage"] == "tune" and r["config"]["family"] == family for r in records)
                   for family in ["catboost", "lightgbm"]}
    lines = ["# 模型优化结果（2026-10-25）", "",
             f"实验：`{name}`；CatBoost参数试验{tune_counts['catboost']}组，LightGBM参数试验{tune_counts['lightgbm']}组。",
             "原始数据、核心源码及依赖哈希见实验目录，运行方法见[模型优化与复现](成员1_模型优化与复现.md)。", "",
             "## 冻结方案与确认", "",
             "| 指标 | 准确率 |", "|---|---|",
             f"| 本轮相同开发划分的固定CatBoost基线 | {plan['development_baseline']:.6f} |",
             f"| 冻结方案开发集（第1～4折） | {plan['development_accuracy']:.6f} |",
             f"| 原CatBoost第5折 | {confirmation['baseline_accuracy']:.6f} |",
             f"| 冻结方案第5折 | {confirmation['accuracy']:.6f} |", "",
             f"本轮规则判断：**{'生成新候选' if confirmation['promoted'] else '保留原CatBoost候选'}**。"
             f"阈值固定为{plan['threshold']}；融合权重和阈值均在第5折评估前保存。", "",
             "第5折未参与本轮搜索或开发训练，但在旧基线中已经被查看，不能称为从未查看的盲测。"
             "开发分数来自较小训练分区，不能与原完整五折均值直接相减。", "",
             "| 模型 | 特征 | 预处理 | 固定轮数 | 种子 | 权重 |", "|---|---|---|---:|---|---:|"]
    for weight, member in zip(plan["weights"], plan["members"]):
        c = member["config"]
        lines.append(f"| {c['family']} | {c['feature_set']} | {c['strategy']} | {c['params']['iterations']} | "
                     f"{', '.join(map(str,member['seeds']))} | {weight:.2f} |")
    lines += ["", "完整参数见frozen_plan.json。"]
    if result["status"] == "promoted_candidate":
        oof = pd.read_csv(directory / "oof_predictions.csv", dtype={"PassengerId": str, "GroupId": str})
        baseline_dir = EXPERIMENTS_DIR / "training/catboost_baseline_20261018"
        baseline = pd.read_csv(baseline_dir / "oof_predictions.csv", dtype={"PassengerId": str}).set_index("PassengerId").loc[oof.PassengerId]
        truth = oof.Transported.to_numpy(bool)
        old_correct = (baseline.probability.to_numpy() >= .5) == truth
        new_correct = oof.prediction.to_numpy(bool) == truth
        h = oof.fold.to_numpy() == 5
        groups = pd.DataFrame({"GroupId": oof.GroupId[h], "delta": new_correct[h].astype(int)-old_correct[h].astype(int), "n": 1})
        aggregated = groups.groupby("GroupId").agg(delta=("delta", "sum"), n=("n", "sum"))
        rng = np.random.default_rng(20261025)
        # 同行GroupId为重采样单位，保留组内错误相关性；仅量化确认折的不确定性。
        draws = rng.integers(len(aggregated), size=(3000, len(aggregated)))
        samples = aggregated.delta.to_numpy()[draws].sum(axis=1) / aggregated.n.to_numpy()[draws].sum(axis=1)
        low, high = np.quantile(samples, [.025, .975])
        uncertainty = dict(unit="GroupId", resamples=3000, seed=20261025, confirmation_groups=len(aggregated),
                           accuracy_difference=float(new_correct[h].mean()-old_correct[h].mean()),
                           percentile_95_lower=float(low), percentile_95_upper=float(high),
                           fixed_errors=int((~old_correct[h] & new_correct[h]).sum()),
                           newly_wrong=int((old_correct[h] & ~new_correct[h]).sum()),
                           interpretation="paired group bootstrap of frozen confirmation only; fold appeared in historical baselines")
        _write_json(directory / "confirmation_uncertainty.json", uncertainty)
        old_metrics = pd.read_csv(baseline_dir / "fold_metrics.csv").sort_values("fold")
        new_metrics = pd.read_csv(directory / "fold_metrics.csv").sort_values("fold")
        if result["selected_oof_mean"] <= old_metrics.accuracy.mean():
            lines[2:2] = ["**当前建议：继续保留原CatBoost为主力。新集成的最终五折均值略低，确认折仅小幅改善，作为独立榜分对照候选保留。**", ""]
            _write_json(directory / "recommendation.json",
                        dict(primary="catboost_baseline_20261018", challenger=name,
                             primary_submission="outputs/catboost_baseline_20261018/submission.csv",
                             challenger_submission=f"outputs/{name}/submission.csv",
                             reason=f"Challenger canonical OOF mean is lower; frozen confirmation net gain is {uncertainty['fixed_errors']-uncertainty['newly_wrong']} samples; see paired group-bootstrap interval",
                             interpretation="Candidate-generation gate passed; this is not automatic replacement of the incumbent. Frozen models unchanged.",
                             kaggle_score_available=False))
        comparison = old_metrics[["fold", "accuracy"]].rename(columns={"accuracy": "baseline_accuracy"}).merge(
            new_metrics[["fold", "accuracy"]].rename(columns={"accuracy": "candidate_accuracy"}), on="fold", validate="1:1")
        comparison["difference"] = comparison.candidate_accuracy - comparison.baseline_accuracy
        comparison["interpretation"] = np.where(comparison.fold == 5, "frozen_confirmation", "post_selection_descriptive")
        _write_csv(comparison, directory / "baseline_comparison.csv")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(5)
        ax.bar(x-.18, comparison.baseline_accuracy, width=.36, label="Original CatBoost")
        bars = ax.bar(x+.18, comparison.candidate_accuracy, width=.36, label="Frozen candidate")
        bars[4].set_hatch("//")
        ax.set_xticks(x, ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5\nconfirmation"])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(max(0,float(comparison[["baseline_accuracy", "candidate_accuracy"]].min().min())-.025),
                    min(1,float(comparison[["baseline_accuracy", "candidate_accuracy"]].max().max())+.025))
        ax.set_title("Folds 1-4 participated in selection; fold 5 was reserved this round")
        ax.legend(loc="best")
        fig.tight_layout()
        figure = PROJECT_ROOT / "reports/figures" / f"{name}_fold_comparison.png"
        fig.savefig(figure, dpi=170)
        plt.close(fig)
        lines += ["", "## 最终重训与不确定性", "",
                  f"冻结方案完成团队五折重训，共保存{result['models']}个模型，测试概率按5个训练分区平均。"
                  f"最终OOF五折均值为{result['selected_oof_mean']:.6f} ± {result['selected_oof_std']:.6f}，"
                  f"按样本加权准确率为{result['oof_accuracy']:.6f}。",
                  f"原CatBoost同一团队五折均值为{old_metrics.accuracy.mean():.6f}，"
                  f"新候选的描述性差值为{result['selected_oof_mean']-old_metrics.accuracy.mean():+.6f}。",
                  "这些OOF分数包含参与过选择的第1～4折，只用于描述最终模型，不能当成新的无偏验证分数。", "",
                  f"确认折修正旧模型{uncertainty['fixed_errors']}个错误，同时新增{uncertainty['newly_wrong']}个错误。"
                  f"按同行组配对bootstrap得到准确率差的95%区间为[{low:+.4f}, {high:+.4f}]。"
                  + ("区间跨过0，目前证据不足以断言稳定优于原模型。" if low <= 0 <= high else
                     "此区间仅反映当前确认折重采样的不确定性，不能保证测试集提升。"), "",
                  f"![各折准确率对比](../reports/figures/{figure.name})", "",
                  "## 提交与复现产物", "",
                  f"- 候选提交：`outputs/{name}/submission.csv`，4277行，PassengerId顺序与原test一致。",
                  f"- 测试概率：`outputs/{name}/test_predictions.csv`。",
                  f"- 完整模型：`models/{name}/ensemble.joblib`。",
                  f"- 校验摘要：`reports/checks/{name}.json`；校验成功后才能确认保存模型的预测复现通过。"]
        diagnostic_path = directory / "training_diagnostics.json"
        if diagnostic_path.exists():
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            lines += ["", f"保存模型重算的训练准确率均值为{diagnostic['training_accuracy_mean']:.6f}，"
                      f"验证准确率均值为{diagnostic['validation_accuracy_mean']:.6f}，"
                      f"差距为{diagnostic['training_gap_mean']:.6f}；逐折见training_gaps.csv。",
                      "此检查只描述冻结模型的拟合情况，没有据此修改模型或阈值。"]
    lines += ["", "## 诊断与边界", "",
              "翻转全部训练标签后派生特征完全不变。3折分组LightGBM对抗验证中，基础、消费、姓名丰富方案AUC均约0.50；"
              "这是指定诊断器没有发现明显来源指纹，不能证明所有分布偏移都不存在。设置不同，不能与成员2旧AUC直接比较。",
              "历史候选脚本已增加空列保护，防止把不存在的特征当作基线重复训练并覆盖旧结果。",
              "本轮未上传Kaggle，未产生实际榜分，未提交或推送GitHub；模板原件未修改。", "",
              "下一步：由另一位成员从原始数据复现；队长先提交原CatBoost主力文件，再用冻结集成文件做榜分对照并记录；报告引用保留折与选择后OOF时注明不同含义。"]
    path = PROJECT_ROOT / "docs" / "模型优化结果_20261025.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="optimization_20261025")
    summarize(parser.parse_args().name)
