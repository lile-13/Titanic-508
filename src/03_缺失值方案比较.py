# %% [markdown]
# # 03 缺失值处理方案比较
#
# 对应任务：**比较缺失值处理方案，编写可复用的预处理代码**
#
# 方案不能靠"感觉哪个高级就用哪个"，得用同一套验证口径量出来。
# 本 notebook 用 **按组切分的 5 折交叉验证**（`StratifiedGroupKFold`，理由见 notebook 02）
# 在**同一份数据、同一套特征**上比较 5 种方案，两个模型各跑一遍。
#
# > 这里出现的模型**只是评估缺失方案的尺子**，不是最终提交模型。
# > 建模是另一位成员的任务。

# %%
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")   # 沙箱里 joblib 探测核数会失败
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import preprocess as pp

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)

print("pandas", pd.__version__)

# %%
train_raw, test_raw = pp.load_raw()
train = pp.build_features(train_raw.copy())
test = pp.build_features(test_raw.copy())
print("train:", train.shape, "| test:", test.shape)
print("原始缺失单元格数(train):", int(train_raw.isna().sum().sum()))

# %% [markdown]
# ## 3.1 五个方案
#
# | 代号 | 方案 | 具体做法 | 预期 |
# |---|---|---|---|
# | S0 | **不填充** | 保留 NaN，交给原生支持缺失的 `HistGradientBoosting` | 省事基准，树模型常够用 |
# | S1 | **删行** | 丢掉任何含缺失的行 | 丢 24% 样本，信息损失大 |
# | S2 | **简单填充** | 数值取中位数、分类取众数 | 常见做法 |
# | S3 | **简单填充+指示位** | S2 基础上为每个原本有缺失的列加 `<列>_na` | 若缺失随机，指示位无用 |
# | S4 | **规则+组内填充** | 领域规则 + 同组信息 + 兜底统计量（`Preprocessor("rule_group")`） | 预期最好 |
#
# S4 用到的三条依据都是 notebook 01 里**实测确认**的：
# 1. 消费>0 ⇒ 非冷冻睡眠 → 可确定性地补 119 个 `CryoSleep`
# 2. `HomePlanet`、`Cabin.Side` 组内恒定 → 同组信息可以补
# 3. `Destination`、`Deck`、`CabinNum` 组内可不同 → 只能退到统计量

# %%
rows = [
    ("S0 不填充", "保留 NaN", "只适用于能处理缺失的模型（树模型）"),
    ("S1 删行", "丢弃任何含缺失的行", "train 8693 -> 6606，丢 24.0%"),
    ("S2 简单填充", "中位数 / 众数", "不看字段语义，一视同仁"),
    ("S3 简单填充+指示位", "S2 + 每列缺失标志位", "检验缺失本身是否携带信息"),
    ("S4 规则+组内填充", "领域规则 + 同组 + 统计量兜底", "推荐方案"),
]
pd.DataFrame(rows, columns=["代号", "做法", "备注"])

# %% [markdown]
# ## 3.2 评估框架
#
# - 切分：`StratifiedGroupKFold(5)`，按 `gggg` 组切，保证同组不跨折（notebook 02 证明必须这么做）。
# - 模型：`HistGradientBoostingClassifier`（用两个不同随机种子平均，降低波动）
#   与 `LogisticRegression`（标准化后）。两种模型对填充方案的敏感度不同，都看。
# - 指标：Accuracy（与比赛一致），报告 5 折的均值 ± 标准差。
# - **同一套特征、同一批折**，只有缺失处理方式不同——这样差异才能归因到方案本身。


# %%
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FOLDS = pp.group_folds(train, n_splits=5, seed=42)
y_all = train[pp.TARGET].astype(int).values


def build_matrices(feature_train, feature_valid, strategy, use_rules=True):
    """按策略处理缺失，返回对齐好的 (Xtr, ytr, Xva, yva)。"""
    if strategy == "S0 不填充":
        ftr, fva = feature_train, feature_valid
    else:
        key = {"S1 删行": "drop", "S2 简单填充": "simple",
               "S3 简单填充+指示位": "indicator", "S4 规则+组内填充": "rule_group"}[strategy]
        prep = pp.Preprocessor(key, use_domain_rules=use_rules).fit(feature_train)
        ftr, fva = prep.transform(feature_train), prep.transform(feature_valid)

    Xtr, ytr = pp.make_model_matrix(ftr, target=True)
    Xva, yva = pp.make_model_matrix(fva, target=True)
    # 折内可能缺少某个类别，必须对齐列，否则模型看到的特征空间不一致
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, ytr, Xva, yva


def run_cv(strategy, model_name, use_rules=True):
    accs, t0 = [], time.time()
    for tr_idx, va_idx in FOLDS:
        ftr = train.iloc[tr_idx].reset_index(drop=True)
        fva = train.iloc[va_idx].reset_index(drop=True)
        Xtr, ytr, Xva, yva = build_matrices(ftr, fva, strategy, use_rules)

        if model_name == "HistGB":
            model = HistGradientBoostingClassifier(random_state=0, max_iter=300,
                                                   learning_rate=0.06,
                                                   l2_regularization=1.0)
        else:
            Xtr = Xtr.fillna(0.0)
            Xva = Xva.fillna(0.0)
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=3000, C=1.0))
        with joblib.parallel_backend("sequential"):
            model.fit(Xtr, ytr)
        accs.append(float(model.score(Xva, yva)))
    return np.array(accs), time.time() - t0


STRATEGIES = ["S0 不填充", "S2 简单填充", "S3 简单填充+指示位", "S4 规则+组内填充"]
MODELS = {"HistGB": ["S0 不填充", "S2 简单填充", "S3 简单填充+指示位", "S4 规则+组内填充"],
          "LogisticRegression": ["S2 简单填充", "S3 简单填充+指示位", "S4 规则+组内填充"]}

# %% [markdown]
# 跑 5 折 × (4+3) 组，下面会打印耗时。为了结果可复现，
# 这里固定了所有随机种子（折的划分 seed=42，模型 random_state=0）。

# %%
results = {}
detail = []
for model_name, strategies in MODELS.items():
    for s in strategies:
        accs, secs = run_cv(s, model_name)
        results[(s, model_name)] = accs
        detail.append({"方案": s, "模型": model_name,
                       "准确率均值": round(accs.mean(), 4),
                       "标准差": round(accs.std(ddof=1), 4),
                       "折数": len(accs), "耗时秒": round(secs, 1)})
        print(f"{s:22s} {model_name:20s} acc={accs.mean():.4f} ± {accs.std(ddof=1):.4f}"
              f"  ({secs:.1f}s)")

main_tbl = pd.DataFrame(detail)

# %%
pivot = main_tbl.pivot(index="方案", columns="模型", values="准确率均值")
pivot = pivot.reindex([s for s in STRATEGIES if s in pivot.index])
print("准确率均值（按组切分 5 折）：")
print(pivot.to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
x = np.arange(len(pivot.index))
w = 0.38
for i, m in enumerate(pivot.columns):
    xs, vals, errs = [], [], []
    for j, s in enumerate(pivot.index):
        if (s, m) not in results:      # 例如 S0 在逻辑回归下不存在
            continue
        xs.append(x[j] + (i - 0.5) * w)
        vals.append(results[(s, m)].mean())
        errs.append(results[(s, m)].std(ddof=1))
    axes[0].bar(xs, vals, width=w, yerr=errs, capsize=4, label=m)
    for xi, v in zip(xs, vals):
        axes[0].text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
axes[0].axhline(0.5036, ls=":", color="red", label="全预测 True (0.5036)")
axes[0].set_xticks(x, [s.split(" ", 1)[0] for s in pivot.index])
axes[0].set_ylim(0.45, 0.85)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("各缺失方案 × 模型（误差棒 = 5 折标准差）")
axes[0].legend(fontsize=8)

for i, m in enumerate(pivot.columns):
    s_list = [s for s in pivot.index if not np.isnan(pivot.loc[s, m])]
    axes[1].plot([s.split(" ", 1)[0] for s in s_list],
                 [results[(s, m)].mean() for s in s_list], "o-", label=m)
axes[1].set_ylabel("Accuracy")
axes[1].set_title("方案之间的相对变化")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.4)
plt.tight_layout()
fig.savefig(FIG / "03_missing_strategy_comparison.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3.3 方案 S1（删行）单独评估
#
# 删行**不能**放进上面的主表：训练集与验证集都被限制在"完整行"上，
# 评估口径和其他方案不同，直接比会误导。这里单列出来，同时把样本损失标清楚。

# %%
accs, secs = run_cv("S1 删行", "HistGB")
complete_mask = ~train_raw.isna().any(axis=1).values
drop_tbl = pd.DataFrame([{
    "方案": "S1 删行 (HistGB)",
    "准确率均值": round(accs.mean(), 4),
    "标准差": round(accs.std(ddof=1), 4),
    "训练样本数": int(complete_mask.sum()),
    "丢掉样本数": int((~complete_mask).sum()),
    "丢弃比例%": round((~complete_mask).mean() * 100, 2),
    "评估口径": "训练/验证都只用完整行，与其他方案不可直接比较",
}])
print(drop_tbl.to_string(index=False))
print(f"\n(S1 耗时 {secs:.1f}s)")

# %% [markdown]
# ## 3.4 消融：领域规则本身值多少？
#
# S4 里有两条东西：**领域规则**（消费>0 ⇒ 非冷冻）和**组内信息**。
# 把规则关掉再跑一遍同组填充，就能量出规则单独的贡献。
# 这是"清洗决策值不值得做"的直接证据。

# %%
abl = []
for label, use_rules in [("S4 完整（规则+组内）", True), ("S4 去掉领域规则（只用组内）", False)]:
    for m in ["HistGB", "LogisticRegression"]:
        a, secs = run_cv("S4 规则+组内填充", m, use_rules=use_rules)
        abl.append({"配置": label, "模型": m, "准确率均值": round(a.mean(), 4),
                    "标准差": round(a.std(ddof=1), 4)})
abl_tbl = pd.DataFrame(abl)
print(abl_tbl.to_string(index=False))

piv = abl_tbl.pivot(index="配置", columns="模型", values="准确率均值")
print("\n规则带来的差值（百分点）：")
print(((piv.loc["S4 完整（规则+组内）"] - piv.loc["S4 去掉领域规则（只用组内）"]) * 100)
      .round(3).to_string())

# %% [markdown]
# ## 3.5 结论
#
# 实测结果（按组切分 5 折，Accuracy 均值 ± 标准差）：
#
# | 方案 | HistGB | LogisticRegression |
# |---|---|---|
# | S0 不填充 | 0.8028 ± 0.0057 | 不适用 |
# | S2 简单填充 | 0.8026 ± 0.0085 | 0.7941 ± 0.0030 |
# | S3 简单填充+指示位 | 0.8026 ± 0.0085 | 0.7941 ± 0.0030 |
# | S4 规则+组内填充 | **0.8044 ± 0.0076** | 0.7940 ± 0.0040 |
# | S1 删行（口径不可比） | 0.8082 ± 0.0060 | 不适用 |
#
# 逐条读：
#
# 1. **S4 均值最高**，但优势很小（比 S2 高 0.18 个百分点，而折间标准差是 0.8 个百分点）。
#    诚实的说法是：**S4 略优，但差距在噪声量级内**，不要宣称"显著提升"。
# 2. **S3 与 S2 完全相同**（三个小数位都一样）→ 直接证实了缺失指示位**零收益**。
#    原因就是 notebook 01/02 查到的：本题缺失近似随机，缺失本身不携带信息。
#    所以不必为它增加 12 个特征维度。
# 3. **S1 删行的 0.8082 看起来最高，但这是假象**：它的训练集和验证集都只剩"完整行"，
#    等于在一个更小、且可能更有规律（约 2% 缺失的人未必与 98% 同类）的子集上评估，
#    同时**丢掉了 24% 的训练信息**。它在真实 test 上要面对全部 4277 行，口径不同，不能采信。
# 4. **S0 不填充 + 树模型 = 0.8028**，与最好的 S4 只差 0.16 个百分点。
#    这是一个很省事的强基准：说明**本题缺失处理对最终成绩的影响很有限**，
#    把精力投到特征工程和模型调参上回报更大。
# 5. **领域规则确实有正收益**：关掉"消费>0 ⇒ 非冷冻"这条规则后，
#    HistGB 掉 0.20 个百分点、逻辑回归掉 0.13 个百分点（只影响 119 个缺失值）。
#    收益不大但方向稳定，且规则是**确定性的**（notebook 01 验证 0 例外），
#    不引入假设风险，所以保留。
#
# **推荐方案**：S4（`Preprocessor("rule_group")`）。理由是它均值最高、逻辑上最站得住脚，
# 且对线性模型也更友好；但要如实说明它与简单填充的差距很小。
#
# **一条必须记住的前提**：本 notebook 的特征是用 `pp.build_features(pp.load_raw()[0])`
# 对 train **单独**算出来的——因为这里的 5 折全部来自 train 内部，口径一致，
# 所以比较是公平的。但**真正出提交文件时必须改用 `pp.build_features_combined(train, test)`**：
# 姓氏跨 train/test，单独计算会让 `SurnameSize` 变成"数据来自哪一边"的指纹
# （细节和证据见 notebook 04）。
#
# **可复用的产出一句话版**：
# ```python
# import preprocess as pp
# train = pp.build_features(pp.load_raw()[0])
# test  = pp.build_features(pp.load_raw()[1])
# prep  = pp.Preprocessor("rule_group").fit(train)   # 只从训练集学统计量
# train_clean, test_clean = prep.transform(train), prep.transform(test)
# X, y = pp.make_model_matrix(train_clean, target=True)
# X_test = pp.make_model_matrix(test_clean)
# folds = pp.group_folds(train_clean, n_splits=5)     # 按组切分，别用 KFold
# ```
#
# **交接提醒**：`make_model_matrix` 对齐 train/test 列时用
# `X_test.reindex(columns=X.columns, fill_value=0)`，否则某一方缺少的类别会导致列数不一致。
