# Titanic-508
# Spaceship Titanic 四人团队项目

## 项目目标
根据乘客的个人信息、舱位和消费记录，预测其是否被传送。
评价指标为准确率（Accuracy）。

比赛地址：
https://www.kaggle.com/competitions/spaceship-titanic
## 团队分工

### 成员1（队长）：核心建模与性能优化
- 跑通数据读取、训练、验证和预测的完整流程。
- 建立逻辑回归基线，并开发 CatBoost 主力模型。
- 统一交叉验证划分，比较模型的平均准确率和稳定性。
- 调整模型参数，分析过拟合情况。
- 与特征负责人配合，验证新特征对主力模型的效果。
- 整合最终方案，完成全量训练和提交文件生成。
- 协调项目进度与代码合并。

交付：训练与预测代码、参数实验记录、最终模型方案和提交文件。

### 成员2：数据分析与清洗
- 检查缺失值、重复记录、字段类型和异常情况。
- 分析乘客属性、消费行为与预测目标的关系。
- 比较缺失值处理方案，编写可复用的预处理代码。
- 检查训练集与测试集的特征分布差异。

交付：数据分析 Notebook、预处理代码、数据质量分析结论。

### 成员3：特征工程与消融实验
- 构造舱位、同行群组、消费和缺失标记等特征。
- 编写统一的特征生成函数。
- 使用固定模型和验证划分，逐组检验新特征。
- 删除无效或不稳定的特征，整理最终特征方案。

交付：特征生成代码、特征说明、特征增删对比记录。

### 成员4：独立对照模型与误差分析
- 使用相同的验证划分，开发随机森林和 LightGBM 对照模型。
- 分析不同模型在哪些样本上容易出错。
- 比较模型预测差异，尝试概率平均融合。
- 判断融合是否稳定优于单模型，并提出最终建议。

交付：对照模型代码、误差分析、融合实验记录。

### 共同要求
- 所有实验使用统一的数据划分与评价指标。
- 每人维护实验记录，并完成自己负责部分的报告。
- 最终代码由另一名成员复现检查。


## 项目目录

```text
Titanic-508/
├── README.md                 项目入口和团队分工
├── requirements.txt          依赖版本
├── .gitignore
├── data/raw/                 原始比赛数据（不上传）
├── src/                      可复用模块：preprocess.py、paths.py
├── scripts/                  导出、Notebook 转换与执行工具
├── notebooks/                已执行的分析 Notebook
│   └── source/               与 Notebook 配对的分析源码
├── features/                 清洗表、统一折划分和特征字典
├── reports/
│   ├── figures/              分析图表
│   └── experiments/          实验结果 CSV
├── docs/                     团队交接说明
└── tests/                    回归与集成测试
```

所有运行命令从仓库根目录执行。原始数据、虚拟环境和缓存保持在本地。
正式模块统一用 `from src import preprocess as pp` 导入。

## 数据准备
加入 Kaggle 比赛后，在 Data 页面下载并解压：
- train.csv
- test.csv
- sample_submission.csv

将以上文件放入本项目的 data/raw/ 文件夹。
原始数据保持不变，处理步骤通过代码实现。

## 运行方法
先按 [成员2交接说明](docs/成员2_数据清洗交接.md) 创建环境并安装 requirements.txt。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.export_features
```

数据放在 data/raw/。正式交叉验证从原始数据构建特征，每折使用 `src.preprocess.prepare_fold_matrices` 独立拟合填补器；`features/train_clean.csv` 仅用于最终全量训练或数据检查。

## 当前进度
已完成数据分析与清洗交接，并修正缺失标记入模、组内补缺范围、数据路径和逐折预处理。
特征工程与消融实验已完成（见 [成员3交接说明](docs/成员3_特征工程交接.md)）：保留
`CabinZone` 舱号分箱（树模型 +1pp），弃用连续 `CabinNum`（指纹风险）与组内聚合特征；
建模矩阵由 53 列变为 59 列。模型方案比较见 reports/experiments/；
最终 CatBoost 开发、调参和比赛提交待完成。
