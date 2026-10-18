# %% [markdown]
# # 05 成员4：独立对照模型与误差分析
#
# 对应任务：**开发随机森林和 LightGBM 对照模型 + 误差分析 + 概率平均融合**
#
# 本 notebook 遵循团队统一约定：
# - 使用统一的 5 折划分（`features/folds.csv`，按 GroupId 的 StratifiedGroupKFold，seed=42），**不自行重新切分**；
# - 每折通过 `src.preprocess.prepare_fold_matrices` 从原始特征独立拟合填补器，**不在 `features/train_clean.csv` 上直接切折**；
# - 对照模型：LogisticRegression（基线）、RandomForest、LightGBM；指标：Accuracy / AUC / LogLoss；
# - 输出：`reports/experiments/member4_*.csv`、`reports/figures/member4_*.png`。
#
# 数据来源：`data/raw/train.csv`、`data/raw/test.csv`（由 `preprocess.find_data_dir()` 自动定位）

# %%
import os
import sys
from pathlib import Path

# 从 Notebook 或分析源码所在位置向上定位项目根目录
START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [START, *START.parents]
            if (p / "src" / "preprocess.py").is_file())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))   # matplotlib 缓存放仓库内，不污染用户目录

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
import lightgbm as lgb

from src import preprocess as pp
from src.paths import EXPERIMENTS_DIR, FIGURES_DIR

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体：默认字体没有汉字，标题会变成方框
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 100)

SEED = 42
print("项目根目录:", ROOT)
print("pandas", pd.__version__, "| numpy", np.__version__,
      "| sklearn", sklearn.__version__, "| lightgbm", lgb.__version__)

# %% [markdown]
# ## 1. 加载数据与统一折划分
#
# `build_features_combined` 合并 train/test 计算人数类特征（统一口径）；
# `folds.csv` 是按 GroupId 分组的 5 折（seed=42），按 PassengerId 关联，不依赖行号。

# %%
train_raw, test_raw = pp.load_raw()
features, test_features = pp.build_features_combined(train_raw, test_raw)

fold_map = pd.read_csv(ROOT / "features" / "folds.csv",
                       dtype={"PassengerId": str, "GroupId": str})
assert fold_map.PassengerId.is_unique
fold_of = features.PassengerId.map(fold_map.set_index("PassengerId")["fold"])
assert fold_of.notna().all() and fold_of.isin(range(1, 6)).all()

print("train", features.shape, "| test", test_features.shape)
print("折分布:", fold_of.value_counts().sort_index().to_dict())

# %% [markdown]
# ## 2. 对照模型定义
#
# - LR：逻辑回归基线（对照下限）
# - RF：随机森林（500 棵树，控制深度防过拟合）
# - LGB：LightGBM（300 棵树，小 num_leaves 防 891 行级别数据过拟合）
#
# 全部固定 `random_state=42`，保证可复现。

# %%
def make_models() -> dict:
    """返回模型工厂：每次调用创建全新实例，保证每折独立训练。"""
    return {
        # LR 走 标准化 + lbfgs 管道：不缩放时 lbfgs 千次迭代不收敛
        "LR": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, random_state=SEED)),
        ]),
        "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=6,
                                             min_samples_leaf=4, random_state=SEED),
        "LGB": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                          num_leaves=15, random_state=SEED, verbosity=-1),
    }

# %% [markdown]
# ## 3. 5 折交叉验证
#
# 每折：`prepare_fold_matrices(strategy="rule_group")` 只在该折训练行上拟合填补器，
# 返回对齐后的建模矩阵（59 维特征 + 缺失标记，随成员3特征方案更新）。记录每折 Accuracy / AUC / LogLoss，
# 并保存验证集预测（供误差分析与融合使用）。

# %%
rows = []
fold_parts = []
for fold in range(1, 6):
    tr_idx = np.flatnonzero((fold_of != fold).to_numpy())
    va_idx = np.flatnonzero((fold_of == fold).to_numpy())
    Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(
        features, tr_idx, va_idx, strategy="rule_group")

    fold_df = pd.DataFrame({
        "PassengerId": features.iloc[va_idx]["PassengerId"].astype(str).to_numpy(),
        "fold": fold,
        "y_true": yva,
    })
    for name, factory in make_models().items():
        model = factory()
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        proba = model.predict_proba(Xva)[:, 1]
        fold_df[f"{name}_pred"] = pred
        fold_df[f"{name}_proba"] = proba
        rows.append({
            "fold": fold, "model": name,
            "accuracy": accuracy_score(yva, pred),
            "auc": roc_auc_score(yva, proba),
            "log_loss": log_loss(yva, proba),
        })
    fold_parts.append(fold_df)
    print(f"折 {fold}/5 完成")

df_cmp = pd.DataFrame(rows)
df_preds = pd.concat(fold_parts, ignore_index=True)
assert df_preds.PassengerId.is_unique, "验证折不应重叠"

# %% [markdown]
# ### 3.1 对比表（每折明细 + 跨折汇总）

# %%
summary = df_cmp.groupby("model").agg(
    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    loss_mean=("log_loss", "mean"), loss_std=("log_loss", "std"),
).round(5)

df_cmp.to_csv(EXPERIMENTS_DIR / "member4_model_comparison.csv",
              index=False, encoding="utf-8-sig")
summary.to_csv(EXPERIMENTS_DIR / "member4_model_summary.csv",
               encoding="utf-8-sig")

print(df_cmp.pivot(index="fold", columns="model", values="accuracy").round(4))
print("\n跨折汇总（Accuracy 均值±std，AUC，LogLoss）：")
print(summary)

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(data=df_cmp, x="fold", y="accuracy", hue="model", ax=ax)
ax.set_title("5 折 Accuracy 对比（LR / RF / LGB）")
ax.set_ylim(0.70, 0.85)
ax.axhline(df_cmp[df_cmp.model == "RF"].accuracy.mean(), ls="--",
           color="tab:blue", lw=1)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "member4_cv_accuracy.png", dpi=150)
plt.close(fig)
print("已保存:", FIGURES_DIR / "member4_cv_accuracy.png")

# %% [markdown]
# ## 4. 误差分析
#
# 合并原始/派生特征做分组说明（缺失单独成组），回答两个问题：
# 1. 各模型**在哪些群体上容易出错**（错误率切片）；
# 2. 两个对照模型**共同错 / 各自独错**的样本规模（判断融合是否有互补空间）。

# %%
df_e = df_preds.copy()
for name in ("LR", "RF", "LGB"):
    df_e[f"{name}_error"] = df_e[f"{name}_pred"] != df_e["y_true"]

GRP_COLS = ["HomePlanet", "Deck", "Side", "AgeBand",
            "IsAlone", "ZeroSpend", "VIP"]
grp = features[["PassengerId"] + GRP_COLS].copy()
grp["PassengerId"] = grp["PassengerId"].astype(str)
df_e = df_e.merge(grp, on="PassengerId", how="left")

# 4.1 混淆矩阵（RF / LGB）
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, name in zip(axes, ("RF", "LGB")):
    cm = confusion_matrix(df_e["y_true"], df_e[f"{name}_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["False", "True"], yticklabels=["False", "True"])
    ax.set_title(f"{name} 混淆矩阵")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "member4_confusion_matrix.png", dpi=150)
plt.close(fig)

# 4.2 分组错误率（长表存 CSV，供报告引用）
parts = []
for c in GRP_COLS:
    tmp = df_e.copy()
    tmp[c] = tmp[c].astype("object").fillna("缺失")   # category/bool 列先转 object 再补缺失组
    long = (tmp.groupby(c)[["RF_error", "LGB_error"]]
               .mean().round(4).reset_index()
               .melt(id_vars=c, var_name="model", value_name="error_rate"))
    long.insert(0, "group_col", c)
    parts.append(long)
df_gerr = pd.concat(parts, ignore_index=True)
df_gerr.to_csv(EXPERIMENTS_DIR / "member4_group_error_rate.csv",
               index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, c in zip(axes, ["Deck", "AgeBand", "IsAlone"]):
    tmp = df_e.copy()
    tmp[c] = tmp[c].astype("object").fillna("缺失")
    g = tmp.groupby(c)[["RF_error", "LGB_error"]].mean().sort_values("RF_error")
    g.plot.barh(ax=ax, legend=(ax is axes[0]))
    ax.set_title(f"分组错误率：{c}")
    ax.set_xlabel("错误率")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "member4_group_error_rate.png", dpi=150)
plt.close(fig)

# 4.3 共错 / 独错
df_e["both_wrong"] = df_e["RF_error"] & df_e["LGB_error"]
df_e["only_RF"] = df_e["RF_error"] & ~df_e["LGB_error"]
df_e["only_LGB"] = ~df_e["RF_error"] & df_e["LGB_error"]
both_ok = (~df_e["RF_error"] & ~df_e["LGB_error"]).sum()
tbl = {
    "两模型都对": both_ok,
    "仅 RF 错": int(df_e["only_RF"].sum()),
    "仅 LGB 错": int(df_e["only_LGB"].sum()),
    "两模型都错": int(df_e["both_wrong"].sum()),
}
print("共错/独错计数：", tbl)

cols_out = ["PassengerId", "fold", "y_true"] + GRP_COLS + \
    [f"{n}_{s}" for n in ("LR", "RF", "LGB") for s in ("pred", "proba")] + \
    ["both_wrong", "only_RF", "only_LGB"]
df_e[cols_out].to_csv(EXPERIMENTS_DIR / "member4_error_analysis.csv",
                      index=False, encoding="utf-8-sig")
print("误差分析明细已保存:", EXPERIMENTS_DIR / "member4_error_analysis.csv")

# %% [markdown]
# ## 5. 概率平均融合实验
#
# 对 RF / LGB 的验证概率做 **soft voting**：
# 1. 每折独立扫描权重 `w ∈ [0,1]`（步长 0.1），记录每折最优融合与简单平均(0.5)的表现；
# 2. 全部验证集拼接后扫描全局最优权重（注意：权重在验证集上挑选，存在轻微过拟合风险）；
# 3. 稳定性判断：融合是否**每一折都不差于**该折最优单模型。

# %%
W = np.arange(0, 1.01, 0.1)
fusion_rows = []
for fold in range(1, 6):
    f = df_e[df_e["fold"] == fold]
    rf_p = f["RF_proba"].to_numpy()
    lg_p = f["LGB_proba"].to_numpy()
    y = f["y_true"].to_numpy()
    accs = [accuracy_score(y, (w * rf_p + (1 - w) * lg_p) > 0.5) for w in W]
    fusion_rows.append({
        "fold": fold,
        "rf_acc": accuracy_score(y, f["RF_pred"].to_numpy()),
        "lgb_acc": accuracy_score(y, f["LGB_pred"].to_numpy()),
        "fusion05_acc": accs[5],                      # w=0.5 简单平均
        "best_w": float(W[int(np.argmax(accs))]),
        "best_fusion_acc": float(max(accs)),
    })
df_fus = pd.DataFrame(fusion_rows)
df_fus["best_single"] = df_fus[["rf_acc", "lgb_acc"]].max(axis=1)
df_fus["fusion05_no_worse"] = df_fus["fusion05_acc"] >= df_fus["best_single"] - 1e-9
df_fus.to_csv(EXPERIMENTS_DIR / "member4_fusion_experiment.csv",
              index=False, encoding="utf-8-sig")

print(df_fus.round(4))
print("\n融合(0.5)每折不差于最优单模型的折数：",
      int(df_fus["fusion05_no_worse"].sum()), "/ 5")

# 全局权重扫描曲线（拼接全部验证集）
rf_p_all = df_e["RF_proba"].to_numpy()
lg_p_all = df_e["LGB_proba"].to_numpy()
y_all = df_e["y_true"].to_numpy()
curve = pd.DataFrame({
    "w": [round(w, 1) for w in W],
    "acc": [accuracy_score(y_all, (w * rf_p_all + (1 - w) * lg_p_all) > 0.5)
            for w in W],
})
best_w_all = float(curve.loc[curve.acc.idxmax(), "w"])
print("全局最优权重 w =", best_w_all,
      "，全局 Accuracy =", round(curve.acc.max(), 4))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(curve["w"], curve["acc"], marker="o")
ax.axvline(0.5, ls="--", color="gray", lw=1, label="w=0.5")
ax.axvline(best_w_all, ls="--", color="tab:orange", lw=1, label=f"最优 w={best_w_all}")
ax.set_xlabel("RF 权重 w（LGB 权重 = 1-w）")
ax.set_ylabel("整体 Accuracy")
ax.set_title("概率平均融合：权重扫描（全部验证集拼接）")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "member4_fusion_weight_sweep.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(5)
ax.bar(x - 0.25, df_fus["rf_acc"], 0.25, label="RF")
ax.bar(x, df_fus["lgb_acc"], 0.25, label="LGB")
ax.bar(x + 0.25, df_fus["fusion05_acc"], 0.25, label="Fusion(0.5)")
ax.set_xticks(x, [f"fold {i}" for i in range(1, 6)])
ax.set_ylabel("Accuracy")
ax.set_ylim(0.70, 0.85)
ax.set_title("每折对比：单模型 vs 简单平均融合")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "member4_fusion_fold_compare.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 6. 结论与最终建议
#
# 判断标准：
# - RF / LGB 是否**跑赢 LR 基线**（同折同指标比较）；
# - 融合是否**每折都不差于最优单模型**（只看均值更高不算"稳定优于"）；
# - 最终建议：选单模型还是融合，给出测试集建议提交（供成员1整合）。

# %%
best_single_mean = df_fus[["rf_acc", "lgb_acc"]].max(axis=1).mean()
fusion_mean = df_fus["fusion05_acc"].mean()
print(f"RF    均值 {df_fus['rf_acc'].mean():.4f} | "
      f"LGB 均值 {df_fus['lgb_acc'].mean():.4f} | "
      f"Fusion(0.5) 均值 {fusion_mean:.4f} | 最优单模型均值 {best_single_mean:.4f}")
print(f"融合在 {int(df_fus['fusion05_no_worse'].sum())}/5 折上不差于最优单模型。")

# 最终建议：全量训练 RF + LGB，概率平均后预测 test（命名"建议"，最终提交由成员1整合）
prep = pp.Preprocessor("rule_group").fit(features)
train_clean = prep.transform(features)
test_clean = prep.transform(test_features)
X, y = pp.make_model_matrix(train_clean, target=True)
X_test = pp.make_model_matrix(test_clean).reindex(columns=X.columns, fill_value=0)

final_models = {name: factory() for name, factory in make_models().items()}
for m in final_models.values():
    m.fit(X, y)

# 选择最终建议方案：
# - 若简单平均融合每折都不差于最优单模型，且均值更高 → 用融合；
# - 否则 → 用 CV 均值最高的单模型（本数据上是 LGB）。
n_stable = int(df_fus["fusion05_no_worse"].sum())
best_name = summary["acc_mean"].idxmax()
if n_stable == 5 and fusion_mean >= best_single_mean - 1e-9:
    final_choice = f"Fusion(0.5)（{n_stable}/5 折稳定不差于单模型）"
    proba = 0.5 * final_models["RF"].predict_proba(X_test)[:, 1] \
        + 0.5 * final_models["LGB"].predict_proba(X_test)[:, 1]
else:
    final_choice = f"单模型 {best_name}（融合仅 {n_stable}/5 折不差于最优单模型）"
    proba = final_models[best_name].predict_proba(X_test)[:, 1]

survived = proba > 0.5
sub = pd.DataFrame({
    "PassengerId": test_raw["PassengerId"].astype(str),
    "Transported": survived.astype(bool),
})
sub_path = EXPERIMENTS_DIR / "member4_suggested_submission.csv"
sub.to_csv(sub_path, index=False, encoding="utf-8-sig")
print("最终建议：", final_choice)
print("建议提交已生成:", sub_path, sub.shape)
