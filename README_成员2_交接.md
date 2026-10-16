# 成员2 交付：数据分析与清洗

Spaceship Titanic（Kaggle）数据分析与清洗阶段的完整产出。
四个 notebook 都已**实际执行过**，输出（含图）嵌在文件里，打开即可看结果，不需要重跑。

## 交付清单

| 文件 | 内容 | 对应任务 |
|---|---|---|
| `notebooks/01_数据体检.ipynb` | 缺失、重复、字段类型、异常、组内一致性 | 检查缺失值/重复记录/字段类型/异常 |
| `notebooks/02_属性消费与目标关系.ipynb` | 各属性与消费行为对 `Transported` 的影响 | 分析属性、消费与目标的关系 |
| `notebooks/03_缺失值方案比较.ipynb` | 5 种缺失处理方案的对照实验 | 比较缺失值处理方案 |
| `notebooks/04_训练测试分布差异.ipynb` | 分布检验 + 对抗验证 | 检查 train/test 分布差异 |
| `preprocess.py` | **可复用预处理模块**（下述 API） | 编写可复用的预处理代码 |
| `features/` | **清洗后的特征表 + 折划分**（见下） | 交给建模成员直接用 |
| `figures/` | 20 张图，notebook 里已嵌入，也可单独引用 | — |

`features/` 由 `python export_features.py` 生成，内容：

| 文件 | 内容 |
|---|---|
| `train_clean.csv` | 清洗后的训练集（8693 行，含 `Transported`） |
| `test_clean.csv` | 清洗后的测试集（4277 行） |
| `folds.csv` | `PassengerId → fold(1~5)` 的映射，**按组切分**，直接照用 |
| `feature_dict.md` | 每个派生列的含义 |

建模成员**不需要重跑清洗**，也不该自己重新随机切分——直接用 `folds.csv`，
否则同组乘客跨 fold 会造成泄漏（理由见 notebook 02）。

辅助工具（可选，不影响交付）：

- `src/*.py` — notebook 的源码版本（`# %%` 分块）。改分析先改这里，再 `python py2nb.py src/xx.py --outdir notebooks` 生成 ipynb。这样 git diff 才看得清。
- `export_features.py` — 一键生成 `features/`（清洗后的表 + 折划分），内部带一致性断言。
- `run_notebook.py` — 命令行执行 notebook 并把输出写回文件。
- `show_nb.py` — 打印 notebook 的文本输出，便于无 GUI 环境核对。
- `test_preprocess.py` — `preprocess.py` 的冒烟测试，改完模块跑一次确认没坏。

## 怎么用

把整个目录拷进 `D:\Titanic-508`（保持 `csv/` 在仓库根目录，notebook 会自动找到数据）。

正常情况：用 JupyterLab 打开 `notebooks/*.ipynb` 直接看，或自己改着跑。

命令行执行（改完 `src/` 之后）：

```powershell
D:\anaconda\python.exe py2nb.py src/01_数据体检.py --outdir notebooks
D:\anaconda\python.exe run_notebook.py "notebooks/01_数据体检.ipynb"
```

## 核心结论

### 数据体检（notebook 01）

- train 8693×14、test 4277×13，与题目描述一致。
- **无重复记录**：整行重复 0、`PassengerId` 重复 0、列名重复 0。
- `CryoSleep` / `VIP` 在 CSV 里读成 `object` 而不是 `bool`——它们是**三态列**（True/False/缺失），处理时要留意。
- 每列缺失率都在 **2%** 上下，非常均匀；含缺失的行 2087/8693（24.0%）。
- **缺失近似随机**：列间缺失相关最高只有 0.025，不存在"某几列一起缺"的结构。
- **组内一致性（与常见说法不同的地方）**：
  - `HomePlanet`、`Cabin.Side` 组内恒定 → 可当组级属性，可用来补缺；
  - `Destination`、`Cabin.Deck`、`CabinNum` 组内**可以不同** → 不能当组级属性；
  - 同姓氏 ≠ 同组：2217 个姓氏中 1664 个跨多个组，姓氏与组号是两套互补信息。
- `Age==0` 有 178 人，是婴儿（与同组成人年龄连贯），**不是脏数据，不要删**。
- **`CryoSleep=True ⇒ 五项消费恒为 0`，3037 人零例外**；反过来"消费>0 ⇒ 非冷冻"也成立，
  可直接用来补 119 个缺失的 `CryoSleep`。

### 属性/消费与目标的关系（notebook 02）

- 目标均衡：正例率 **0.5036**，Accuracy 是合理指标；"全预测 True"的基准分就是 0.5036。
- **最强单变量是 `CryoSleep`**：冷冻睡眠者目标率 0.8176，非冷冻者 0.3289。
- **消费有独立作用，不只是冷冻睡眠的影子**：在 `CryoSleep=False` 内部，
  无消费者目标率 0.6158、有消费者 0.2987，差异依然巨大。
- `LogTotalSpend` 与目标的点二列相关 **-0.4689**（绝对值最大）；
  而原始 `TotalSpend` 只有 -0.1995 → **必须做 log 变换**。
- `Deck`：B=0.7343、C=0.6801 明显高，E=0.3573 最低。
- `GroupSize`：单人 0.4524，4 人组 0.6408 最高。
- 各列 `__缺失__` 类别的目标率都贴近 0.5036 → 缺失本身几乎不携带信息。

### 缺失值方案比较（notebook 03）

按组切分 5 折，HistGB / LogisticRegression 各跑一遍：

| 方案 | HistGB | LogisticRegression |
|---|---|---|
| S0 不填充 | 0.8028 ± 0.0057 | 不适用 |
| S2 简单填充（中位数/众数） | 0.8026 ± 0.0085 | 0.7941 ± 0.0030 |
| S3 S2 + 缺失指示位 | 0.8026 ± 0.0085 | 0.7941 ± 0.0030 |
| **S4 规则+组内填充** | **0.8044 ± 0.0076** | 0.7940 ± 0.0040 |
| S1 删行（口径不可比） | 0.8082 ± 0.0060 | 不适用 |

- **S4 均值最高，但优势在噪声量级内**（高出 0.18pp，而折间标准差是 0.8pp）。不要写成"显著提升"。
- **S3 与 S2 结果完全一样** → 缺失指示位**零收益**，不必为它加 12 个特征维度。
- **S1 删行的 0.8082 是假象**：它训练和验证都只剩完整行，口径不同，且丢了 24% 的训练信息。
- **领域规则本身有正收益**：关掉"消费>0 ⇒ 非冷冻"后 HistGB 掉 0.20pp、逻辑回归掉 0.13pp。
- 总结论：**本题缺失处理对成绩影响很有限**（S0 与最好的 S4 只差 0.16pp），
  精力投到特征工程和调参上回报更大。

### train/test 分布差异（notebook 04）—— **这一节有一个必须转达的发现**

分布本身：缺失率最大差 0.373pp；7 个分类变量卡方检验 p 全 > 0.05；
8 个数值变量 KS 检验 p 全 > 0.05；只用原始列做对抗验证 AUC ≈ 0.52。
**原始数据是同分布的**，本地 CV 分数可以作为提交依据。

**但是**：用上全部派生特征后，对抗验证 AUC 冲到 **0.7336**。元凶是 `SurnameSize`——
它被在 train 和 test 里**各算各的**，而姓氏跨数据集（1536 个姓氏两边都有，
覆盖 train 6137 人、test 3759 人），train 人数约为 test 两倍，计数天然偏大
（train 均值 5.33 vs test 3.51）。这个特征于是变成**"样本来自哪个数据集"的指纹**。

修复：`pp.build_features_combined(train, test)` 把派生特征在合并数据上算完再拆开。
修复后 `SurnameSize` 两侧均值 7.085 vs 7.079，对抗 AUC 回落到 **0.5454**。

## 给全组的行动项

1. **验证一律按组切分**：用 `pp.group_folds()`（内部是 `StratifiedGroupKFold`）。
   同组结果一致性 0.4356，显著高于随机基准 0.3702（95% 区间 0.3463~0.3959），
   随机切分会让同组乘客跨 fold 泄漏，本地分数虚高、榜上掉分。
2. **特征构建统一用 `pp.build_features_combined(train, test)`**，不要分别对两边调
   `build_features`。凡是"同一姓氏/同一组有多少人"这类聚合计数特征都有这个坑。
3. **每天只有 10 次提交**。建议：第一次提交用来验证管线（能出分、格式对），
   之后每次提交前必须先跑本地按组 CV，只在 CV 有稳定提升时提交。
4. **本机是 pandas 3.0.3**，不是 1.x/2.x——字符串默认 dtype、copy-on-write 语义都变了，
   网上抄的老代码经常直接报错。写代码时按 3.0 的写法。
5. **`make_model_matrix` 对齐列**：predict 侧要
   `X_test = X_test.reindex(columns=X_train.columns, fill_value=0)`，
   否则某一方缺少的类别会导致列数不一致。

## `preprocess.py` API 速查

```python
import preprocess as pp

# 载入（自动定位 csv/ 目录，兼容在 notebooks/ 下运行）
train_raw, test_raw = pp.load_raw()

# 特征：出提交文件用这个（合并计算，避免 SurnameSize 指纹问题）
train, test = pp.build_features_combined(train_raw, test_raw)
# 仅在"只需 train 内部一致"的分析场景可以分别调 pp.build_features(df)

# 体检
pp.basic_report(train_raw)        # 逐列类型/缺失/唯一值
pp.duplicate_report(train_raw)    # 重复记录
pp.missing_summary(train_raw)     # 缺失总览
pp.consistency_report(train)      # 组内一致性
pp.apply_domain_rules(pp.normalize_types(train_raw))   # 领域规则 + 命中统计

# 缺失处理（fit 只在训练集上做，统计量不会从 test 学）
prep = pp.Preprocessor("rule_group").fit(train)
train_clean = prep.transform(train)
test_clean = prep.transform(test)
# strategy: "drop" | "simple" | "indicator" | "rule_group"
# 可加 use_domain_rules=False 做消融

# 建模矩阵
X, y = pp.make_model_matrix(train_clean, target=True)
X_test = pp.make_model_matrix(test_clean)

# 按组切分
folds = pp.group_folds(train_clean, n_splits=5, seed=42)
```

## 已知限制（如实说明）

- **没有做任何建模/调参**——那是其他成员的任务。notebook 03 里的模型只是评估缺失方案的尺子。
- notebook 03 的特征是在 train **内部**单独算的（因为 5 折都在 train 里，口径一致，比较公平）。
  真正出提交文件时必须走 `build_features_combined`。
- notebook 04 的对抗验证标签是"数据来源"而非 `Transported`，所以它只能说明**特征分布**一致，
  不能说明 test 的目标率与 train 相同（test 无标签，无法验证）。
- `Deck='T'` 只有 5 人、`VIP=True` 只占约 2%，这类稀有类别的统计波动大，结论别过度解读。
- 所有结论都基于当前这份数据（已核对与你提供的 `sample_submission.csv` 同源）。
