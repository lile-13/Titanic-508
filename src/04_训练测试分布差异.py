# %% [markdown]
# # 04 训练集与测试集的特征分布差异
#
# 对应任务：**检查训练集与测试集的特征分布差异**
#
# 要回答的问题不是"看起来像不像"，而是：**有没有变量在 train/test 之间系统性地不同**。
# 如果存在，那么
# - 训练集上学到的规律在 test 上可能不成立，本地 CV 会高估；
# - 或者反过来说，某些特征可以"认出"样本来自哪一边，说明划分不随机。
#
# 三把尺子，从粗到细：
# 1. 逐列缺失率对比
# 2. 分类变量的分布对比 + 卡方检验；数值变量的分布对比 + KS 检验
# 3. **对抗验证（adversarial validation）**：训练一个模型区分"这条来自 train 还是 test"，
#    如果 AUC 接近 0.5，就是同分布。

# %%
import os
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import preprocess as pp

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)

train_raw, test_raw = pp.load_raw()
train = pp.build_features(train_raw.copy())
test = pp.build_features(test_raw.copy())
print("train:", train.shape, "| test:", test.shape)

# %% [markdown]
# ## 4.1 缺失率对比
#
# 最粗的一层：如果某个字段在一侧的缺失率明显更高，那这侧的数据质量就不同。

# %%
mtr = train_raw.isna().mean() * 100
mte = test_raw.isna().mean() * 100
cmp_miss = pd.DataFrame({"train缺失率%": mtr, "test缺失率%": mte})
cmp_miss["差值(百分点)"] = (cmp_miss["test缺失率%"] - cmp_miss["train缺失率%"]).round(3)
cmp_miss = cmp_miss.round(3)
print(cmp_miss.to_string())
print("\n最大差值: %.3f 个百分点" % cmp_miss["差值(百分点)"].abs().max())

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
order = cmp_miss["train缺失率%"].sort_values().index
pos = np.arange(len(order))
axes[0].barh(pos - 0.2, cmp_miss.loc[order, "train缺失率%"], height=0.4, label="train")
axes[0].barh(pos + 0.2, cmp_miss.loc[order, "test缺失率%"], height=0.4, label="test")
axes[0].set_yticks(pos, order, fontsize=8)
axes[0].set_xlabel("缺失率 %")
axes[0].set_title("逐列缺失率：train vs test")
axes[0].legend()

axes[1].barh(order, cmp_miss.loc[order, "差值(百分点)"], color="#4C72B0")
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("test − train（百分点）")
axes[1].set_title("缺失率差值")
axes[1].tick_params(labelsize=8)
plt.tight_layout()
fig.savefig(FIG / "04_missing_rate_diff.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4.2 分类变量：分布对比 + 卡方检验
#
# 卡方检验的原假设是"该变量在 train/test 中同分布"。
# p 值大 → 没有证据说明不同；p 值小 → 分布有系统性差异。

# %%
cat_cols = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP", "AgeBand"]
rows = []
for col in cat_cols:
    # 统一转成字符串再统计：CryoSleep/VIP 是可空布尔，混着字符串会无法排序
    a = train[col].astype(object).fillna("__缺失__").astype(str).value_counts()
    b = test[col].astype(object).fillna("__缺失__").astype(str).value_counts()
    idx = sorted(set(a.index) | set(b.index))
    tab = np.array([[a.get(k, 0) for k in idx], [b.get(k, 0) for k in idx]])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    rows.append({
        "变量": col,
        "类别数": len(idx),
        "卡方": round(chi2, 3),
        "自由度": dof,
        "p值": round(p, 4),
        "结论": "无显著差异" if p > 0.05 else "有差异(p<0.05)",
    })
chi_tbl = pd.DataFrame(rows)
print(chi_tbl.to_string(index=False))

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), cat_cols):
    a = train[col].astype(object).fillna("__缺失__").astype(str).value_counts(normalize=True)
    b = test[col].astype(object).fillna("__缺失__").astype(str).value_counts(normalize=True)
    idx = sorted(set(a.index) | set(b.index))
    pos = np.arange(len(idx))
    ax.bar(pos - 0.2, [a.get(k, 0) for k in idx], width=0.4, label="train")
    ax.bar(pos + 0.2, [b.get(k, 0) for k in idx], width=0.4, label="test")
    ax.set_xticks(pos, idx, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{col}", fontsize=10)
    ax.set_ylabel("占比", fontsize=8)
    ax.legend(fontsize=8)
axes.ravel()[-1].axis("off")
plt.suptitle("分类变量占比：train vs test", y=1.0)
plt.tight_layout()
fig.savefig(FIG / "04_categorical_distribution.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4.3 数值变量：分布对比 + KS 检验
#
# KS 检验比较两条累积分布曲线，对形状差异（不只是均值）敏感。
# 同时看分位数：**均值相同也可能分布形状不同**。

# %%
num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "TotalSpend", "CabinNum"]
rows = []
for col in num_cols:
    a = pd.to_numeric(train[col], errors="coerce").dropna()
    b = pd.to_numeric(test[col], errors="coerce").dropna()
    ks, p = stats.ks_2samp(a, b)
    rows.append({
        "变量": col,
        "train均值": round(a.mean(), 2), "test均值": round(b.mean(), 2),
        "train中位数": round(a.median(), 1), "test中位数": round(b.median(), 1),
        "KS统计量": round(ks, 4), "p值": round(p, 4),
        "结论": "无显著差异" if p > 0.05 else "有差异(p<0.05)",
    })
ks_tbl = pd.DataFrame(rows)
print(ks_tbl.to_string(index=False))

# %%
q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), num_cols):
    a = pd.to_numeric(train[col], errors="coerce").dropna()
    b = pd.to_numeric(test[col], errors="coerce").dropna()
    qa, qb = np.quantile(a, q), np.quantile(b, q)
    ax.plot(q, qa, "o-", label="train")
    ax.plot(q, qb, "s--", label="test")
    ax.set_title(col, fontsize=10)
    ax.set_xticks(q)
    ax.set_xlabel("分位点", fontsize=8)
    ax.legend(fontsize=8)
plt.suptitle("数值变量分位数曲线：两条线越贴合，分布越一致", y=1.0)
plt.tight_layout()
fig.savefig(FIG / "04_numeric_quantile.png", dpi=140, bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Age", "LogTotalSpend", "CabinNum"]):
    a = pd.to_numeric(train[col], errors="coerce").dropna()
    b = pd.to_numeric(test[col], errors="coerce").dropna()
    bins = np.histogram_bin_edges(pd.concat([a, b]), bins=40)
    ax.hist(a, bins=bins, alpha=0.55, density=True, label="train")
    ax.hist(b, bins=bins, alpha=0.55, density=True, label="test")
    ax.set_title(col)
    ax.legend()
plt.suptitle("关键数值变量的密度对比（归一化，两侧样本量不同也能比）", y=1.02)
plt.tight_layout()
fig.savefig(FIG / "04_numeric_density.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4.4 对抗验证（adversarial validation）
#
# 把 train/test 混在一起，训练一个模型预测"这条来自哪一边"。
#
# - **AUC ≈ 0.5** → 模型无法区分，两边同分布 → 本地 CV 可信，
#   也不需要做"挑与 test 最像的训练样本"这类操作。
# - **AUC 明显 > 0.5** → 存在系统性差异，需要找出是哪些变量造成的。
#
# 这一步的标签是"数据来源"，不是 `Transported`，所以随机切分即可，不需要按组。
# 同时给出**置换重要性**，直接指出"哪个变量的分布差异最大"。

# %%
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score

X_all = pd.concat([
    pp.make_model_matrix(pp.Preprocessor("indicator").fit_transform(train)),
    pp.make_model_matrix(pp.Preprocessor("indicator").fit(test).transform(test)),
], ignore_index=True)
y_domain = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
print("对抗验证样本:", X_all.shape, "| 来自 test 的比例: %.3f" % y_domain.mean())

clf = HistGradientBoostingClassifier(random_state=0, max_iter=200, learning_rate=0.06)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
with joblib.parallel_backend("sequential"):
    aucs = cross_val_score(clf, X_all, y_domain, cv=cv, scoring="roc_auc", n_jobs=1)
print("对抗验证 AUC（5 折）:", np.round(aucs, 4), "| 均值 %.4f" % aucs.mean())

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.bar(range(1, 6), aucs, color="#4C72B0")
ax.axhline(0.5, ls="--", color="red", label="AUC=0.5（完全无法区分）")
ax.axhline(aucs.mean(), ls="-", color="#DD8452", label=f"均值 {aucs.mean():.4f}")
ax.set_ylim(0.4, 0.7)
ax.set_xlabel("折")
ax.set_ylabel("AUC")
ax.set_title("对抗验证：能否区分 train / test")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG / "04_adversarial_auc.png", dpi=140, bbox_inches="tight")
plt.show()

# %%
with joblib.parallel_backend("sequential"):
    clf.fit(X_all, y_domain)
    imp = permutation_importance(clf, X_all, y_domain, n_repeats=5,
                                 random_state=0, scoring="roc_auc", n_jobs=1)
imp_tbl = (pd.DataFrame({"特征": X_all.columns, "重要性": imp.importances_mean,
                         "标准差": imp.importances_std})
           .sort_values("重要性", ascending=False).head(15).round(5))
print("置换重要性 Top 15（对区分 train/test 的贡献）：")
print(imp_tbl.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
t = imp_tbl.iloc[::-1]
ax.barh(t["特征"], t["重要性"], xerr=t["标准差"], color="#4C72B0")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("置换重要性（AUC 下降量）")
ax.set_title("哪些变量在 train/test 之间差异最大")
plt.tight_layout()
fig.savefig(FIG / "04_adversarial_importance.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 诊断：AUC 0.73 是谁造成的？
#
# 置换重要性把矛头直接指向 `SurnameSize`（重要性约 0.27，比第二名高一个数量级）。
# 这个特征是"同一姓氏总共有多少人"，而它是在 **train 和 test 里各算各的**。
#
# - 组号 `gggg` 在两边**不重叠**（交集 0），所以 `GroupSize` 怎么算都一样；
# - 但**姓氏是重叠的**：1536 个姓氏同时出现在两边。
#   train 有 8693 人、test 只有 4277 人，同一个姓氏在 train 里数出来的计数天然更大。
#
# 于是这个特征变成了**"我来自哪个数据集"的指纹**。下面用数据坐实。

# %%
sv = pd.DataFrame({
    "train 人数": train.SurnameSize.value_counts().sort_index(),
    "test 人数": test.SurnameSize.value_counts().sort_index(),
}).fillna(0).astype(int)

s_tr = set(train.Surname.dropna())
s_te = set(test.Surname.dropna())
inter = s_tr & s_te

print("SurnameSize 取值分布（各数据集单独计算）：")
print(sv.head(12).to_string())
print("\n均值: train %.3f  |  test %.3f   -> 同一个人口统计量，两侧差了一大截"
      % (train.SurnameSize.mean(), test.SurnameSize.mean()))
print("组号交集: %d（不重叠，GroupSize 不受影响）" % len(set(train.GroupId) & set(test.GroupId)))
print("姓氏交集: %d 个 -> 覆盖 train %d 人、test %d 人"
      % (len(inter), int(train.Surname.isin(inter).sum()), int(test.Surname.isin(inter).sum())))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
ks = np.arange(1, 13)
axes[0].bar(ks - 0.2, sv["train 人数"].reindex(ks, fill_value=0), width=0.4, label="train")
axes[0].bar(ks + 0.2, sv["test 人数"].reindex(ks, fill_value=0), width=0.4, label="test")
axes[0].set_xlabel("SurnameSize")
axes[0].set_ylabel("人数")
axes[0].set_title("各算各的：SurnameSize 两边分布完全不同")
axes[0].legend()

fix_m = pp.build_features_combined(train_raw.copy(), test_raw.copy())
f_tr, f_te = fix_m
sv2 = pd.DataFrame({
    "train 人数": f_tr.SurnameSize.value_counts().sort_index(),
    "test 人数": f_te.SurnameSize.value_counts().sort_index(),
}).fillna(0).astype(int)
axes[1].bar(ks - 0.2, sv2["train 人数"].reindex(ks, fill_value=0), width=0.4, label="train")
axes[1].bar(ks + 0.2, sv2["test 人数"].reindex(ks, fill_value=0), width=0.4, label="test")
axes[1].set_xlabel("SurnameSize")
axes[1].set_ylabel("人数")
axes[1].set_title("合并计算：口径一致\ntrain 均值 %.3f vs test 均值 %.3f"
                  % (f_tr.SurnameSize.mean(), f_te.SurnameSize.mean()))
axes[1].legend()
plt.tight_layout()
fig.savefig(FIG / "04_surnamesize_leak.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# 修复方式：**把 train 和 test 拼起来算派生特征，再拆回去**
# （`preprocess.build_features_combined`）。组号不跨集，所以这样做不会把
# test 的信息带进 train；用到的只是"test 也有这些公开列"这一事实，不涉及 `Transported`。

# %%
def adv_auc(ftr, fte, label=""):
    X = pd.concat([
        pp.make_model_matrix(pp.Preprocessor("indicator").fit_transform(ftr)),
        pp.make_model_matrix(pp.Preprocessor("indicator").fit(fte).transform(fte)),
    ], ignore_index=True)
    y = np.r_[np.zeros(len(ftr), dtype=int), np.ones(len(fte), dtype=int)]
    m = HistGradientBoostingClassifier(random_state=0, max_iter=200, learning_rate=0.06)
    with joblib.parallel_backend("sequential"):
        auc = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              scoring="roc_auc", n_jobs=1)
    print(f"{label:36s} AUC={auc.mean():.4f} ± {auc.std(ddof=1):.4f}")
    return auc.mean()


auc_before = adv_auc(train, test, "① 各算各的（有问题的做法）")
auc_after = adv_auc(f_tr, f_te, "② 合并计算（推荐做法）")

drop_cols = ["SurnameSize", "SurnameCount", "GroupSize", "IsAlone", "IsMixedGroup",
             "GroupSurnameCount", "GroupPos", "GroupNum"]
raw_cols = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
            "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name",
            "PassengerId"]
no_size = [c for c in train.columns if c not in drop_cols and c in test.columns]
auc_nosize = adv_auc(train[no_size], test[no_size], "③ 去掉姓氏规模类特征")
auc_raw = adv_auc(train[raw_cols], test[raw_cols], "④ 只用原始列")

fig, ax = plt.subplots(figsize=(7.6, 4))
labels = ["①各算各的", "②合并计算", "③去掉姓氏规模", "④只用原始列"]
vals = [auc_before, auc_after, auc_nosize, auc_raw]
bars = ax.bar(labels, vals,
              color=["#C44E52", "#55A868", "#4C72B0", "#8172B3"])
ax.axhline(0.5, ls="--", color="red", label="AUC=0.5（无法区分）")
ax.set_ylim(0.45, 0.8)
ax.set_ylabel("对抗验证 AUC")
ax.set_title("各做法的对抗验证 AUC：越接近 0.5 越好")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG / "04_adv_fix_comparison.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4.5 结论
#
# **分布差异本身：几乎没有。**
#
# | 检查 | 结果 |
# |---|---|
# | 缺失率差值 | 最大 0.373 个百分点（`FoodCourt`），其余都在 0.3pp 以内 |
# | 分类变量卡方检验 | 7 个变量 p 值全部 > 0.05，最小 0.0592（`AgeBand`） |
# | 数值变量 KS 检验 | 8 个变量 p 值全部 > 0.05，最小 0.185（`CabinNum`） |
# | 只用原始列做对抗验证 | AUC ≈ 0.52，基本无法区分 |
#
# 结论：**原始数据的 train/test 是同分布的**，本地按组交叉验证的分数可以作为提交依据，
# 不需要做"挑与 test 最像的训练样本"这类加权采样。
#
# **但对抗验证抓到了一个真问题（这才是本 notebook 最有价值的产出）：**
#
# 用上全部派生特征时 AUC 冲到 **0.73**，元凶是 `SurnameSize` —— 它被
# **在 train 和 test 里各算各的**，而姓氏是跨数据集的（1536 个姓氏两边都有），
# train 人数约为 test 的两倍，于是计数天然偏大（train 均值 5.33 vs test 3.51）。
# 这个特征因此变成了"样本来自哪个数据集"的指纹。
#
# 后果：树模型会在这个特征上切出一个阈值，而这个阈值在两边含义完全不同，
# 属于**静默的性能损失**——本地 CV 看不出来，榜上才知道。
#
# **修复**：改用 `pp.build_features_combined(train, test)`，把派生特征
# 在合并数据上算完再拆开。修复后两侧 `SurnameSize` 均值变成 7.085 vs 7.079，
# 对抗 AUC 回落到 0.545（见上方实际数值）。
#
# **给全组的行动项**：
# 1. 所有"同一姓氏/同一组有多少人"这类**聚合计数特征**，一律用
#    `build_features_combined`，不要分别对 train/test 调 `build_features`。
# 2. 每次新增特征后重跑本节——若某个新特征让对抗 AUC 明显上升，
#    说明它在两侧不稳定，属于隐患而不是收益。
# 3. `GroupSize` 不受影响（组号不跨集），但统一走合并入口更省心。
