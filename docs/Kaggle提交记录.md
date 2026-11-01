# Kaggle提交记录

比赛：Spaceship Titanic。更新日期：2026-11-01。

## 四份探索候选的实际反馈 2026-11-01

用户截图确认四份均为Complete：TabPFN 25%与50%各0.81271，纯TabPFN为0.80921，TabICL 75%为**0.81318**。平台最佳更新为原树25%与TabICL 75%，比此前0.81178提高0.00140（0.14个百分点）；距离0.82为0.00682。

![四份候选实际成绩](../reports/figures/kaggle_manual_batch_20261101.png)

截图原图直接复制保存；账号Litianle4690、文件名、Complete和显示分数可见。相对时间不用于推断准确提交时刻，排名为空，上传字节仍未由服务器验证。结构化账本保留9次真实结果，前5次不改写；本批4份从待提交移至已完成。来源CSV哈希继续保留。

这次选择来自用户明确要求的公开榜单探索，不能把未通过本地稳定性门槛的TabPFN研究改写为成功，也不能保证提高TabICL权重会一直提高分数。50% TabPFN在4套历史分组均有正增益，但实际榜分仍低于75% TabICL，说明本地与平台排序并不一致。

最新最佳模型单独保存为`models/manual_submission_batch_20261031_tabicl75/ensemble.joblib`，50个顶层成员，原45树总权重0.25、5个TabICL上下文各0.15。未重新更新神经网络参数，原模型包和原CSV保留。以下旧章节按各自日期保留当时的状态。

## 第五次提交 2026-10-30

**用户截图确认`submission_tabicl_mean4690.csv`为Complete，得分0.81178，超过原最佳0.81108，差值0.00070，即0.07个百分点。** 当前平台最佳更新为TabICL概率平均与原45模型各50%的融合，阈值0.5，共50个顶层成员。原方案、权重和提交CSV继续保留。

![第五次提交与此前成绩](../reports/figures/kaggle_tabicl_mean_20261030.png)

原始截图直接复制保存，SHA256为`43ddeff9eb09a81a3a06e8f6f05dc928e145a3b4e18d5818a48fdda2250cf16f`。截图显示账号Litianle4690、文件名、Complete和得分；相对时间不作为准确提交时刻，排名为空。截图不能校验服务器上所上传文件的字节内容；账本保存本地预期CSV的SHA256。

本轮没有根据榜分再次调整模型、融合权重或阈值，也没有额外训练或提交。新分组验证分别净多对47/0人、8/10折改善；条件增益区间仍包含零。此次确认的是平台分数有所提高，不能由单次结果推断统计显著或泛化稳定性。原始实验摘要及提交前审计记录保留评分前状态，真实平台成绩以账本为准。细节见[诊断与概率平均实验](TabICL失稳诊断与概率平均_20261030.md)。

## 第四次提交 2026-10-29

新增`outputs/kaggle_ready_4690/submission_regularized4690.csv`，采用原集成75%与G/E甲板局部LightGBM25%，阈值0.5。两次新分组五折的准确率分别由0.812148到0.814448、由0.814218到0.814794，净多预测正确20人和5人；10折中7折改善，Log loss均降低。该证据属于历史数据上的固定模型配对复验，不是新的盲测。

新文件与原集成有76个标签不同，4277行ID及顺序已验证；独立CLI重载模型生成的CSV逐字节一致。SHA256为`8715f16bb14eb6d5e4cdec84cd55b39ac1977208be71a1220833274b8a4d8ce9`。模型保存在`models/paired_regularization_20261029/ensemble.joblib`，29项测试通过。

**用户截图确认新候选为Complete，得分0.81061，比原最佳0.81108低0.00047（0.047个百分点），继续保留原45模型集成。** 本地改善没有转化为平台提升。配对同行组bootstrap的条件95%区间约为[-0.000908,+0.003633]，仍跨过0；它也没有计入训练与选择不确定性。完整实验见[学习与校准记录](学习与校准实验_20261029.md)。课程报告与源码包继续暂停整理。

![第四次提交及此前成绩](../reports/figures/kaggle_regularized_20261029.png)

原始截图直接复制保存，未修改图片；可见文件名、Complete、账号Litianle4690和得分。相对时间不作为准确提交时刻，排名仍为空。截图不能校验服务器所上传文件的字节内容；结构化账本记录本地预期CSV哈希。训练期原始摘要保留，平台结果以提交账本为准。

## 已完成的平台提交

九份预测均已提交成功，当前最高为**0.81318**。本地分组验证与平台评分分别记录，使用真实榜分选择当前平台最佳。

| 模型 | 提交文件名 | 本地验证（口径见说明） | Kaggle得分 | 状态 |
|---|---|---:|---:|---|
| 原CatBoost | submission4690.csv | 0.8163 | 0.80313 | Complete，用户截图确认 |
| 冻结集成 | submission_ensemble4690.csv | 0.8157 | 0.81108 | Complete，原平台最佳 |
| 全量重训与五折集成各50% | submission_fullblend4690.csv | 无新增OOF指标 | 0.81108 | Complete，与原集成持平 |
| 原集成75%与局部LightGBM25% | submission_regularized4690.csv | 0.8179（选择后OOF） | 0.81061 | Complete，未替换原最佳 |
| 原集成50%与TabICL概率平均50% | submission_tabicl_mean4690.csv | 0.821235（选择后OOF） | 0.81178 | Complete，此前平台最佳 |
| 此前最佳75%与TabPFN25% | submission_pfn25_4690.csv | 0.816663（4套历史分组均值） | 0.81271 | Complete，探索对照 |
| 此前最佳50%与TabPFN50% | submission_pfn50_4690.csv | 0.816692（4套历史分组均值） | 0.81271 | Complete，探索对照 |
| 纯TabPFN | submission_pfn100_4690.csv | 0.816922（4套历史分组均值） | 0.80921 | Complete，探索对照 |
| 原树25%与TabICL75% | submission_tabicl75_4690.csv | 0.815397（4套历史分组均值） | 0.81318 | Complete，当前平台最佳 |

集成本地OOF包含参与过模型选择的折，属于选择后的描述性结果。4套历史分组使用同一批8693名乘客，均值不是4份独立数据。历次本地验证与平台排序并不完全一致，保留原始结果。全量重训对照未产生新的OOF分数，不能借用原OOF。九次平台得分不足以证明跨数据集稳定提升。

## 首次提交证据

![CatBoost提交成功与评分](../reports/figures/kaggle_catboost_20261027.png)

截图可见比赛名称、上传文件名、Complete状态、账号Litianle4690、模型描述和0.80313得分。未显示准确提交时刻和排名，因此记录中保持为空。模型关联依据提交描述及此前操作流程；截图不能验证上传文件的字节内容。

结构化记录：`reports/experiments/kaggle_submissions.json`。截图文件由用户原图直接复制，没有修改图片。

## 两次提交对照证据

![集成与CatBoost平台成绩对照](../reports/figures/kaggle_comparison_20261027.png)

该截图同时显示submission_ensemble4690.csv的0.81108分、submission4690.csv的0.80313分、两条Complete状态及账号信息，适合作为报告中的平台对照证据。截图中的相对时间不能确定准确提交时刻，记录中仍不填写时刻或排名。

## 当前候选与复现

当前平台最佳文件：`outputs/kaggle_ready_4690/next_batch_20261031/submission_tabicl75_4690.csv`，模型为`models/manual_submission_batch_20261031_tabicl75/ensemble.joblib`。4277行，SHA256为`abd9924480c5ab75bb677384e9251b81ab3195bc25c6e750834b48f55ac807cf`。

此前0.81178文件`outputs/kaggle_ready_4690/submission_tabicl_mean4690.csv`与`models/tabicl_probability_mean_20261030/ensemble.joblib`保留；下段为此前方案的复现检查。

提交前已通过56项测试、25个上下文及90个对照树模型重载、独立CLI生成及独立概率组合核验。来源、分组及格式检查见`reports/checks/tabicl_probability_mean_20261030.json`。本地`manifest.json`已将本候选从待提交移至已提交列表；模型和CSV继续由.gitignore忽略。

### 第三次提交 2026-10-28

![全量重训对照与前两次提交](../reports/figures/kaggle_fullblend_20261028.png)

用户截图确认submission_fullblend4690.csv为Complete、得分0.81108，同时显示前两次成绩。该方案增加9个全量模型，与原45模型集成各50%，共54个模型；本地有34个测试标签变化，但显示分数持平，不能将变化解读为34处修正，也不能推断隐藏测试结果完全相同。继续保留原方案，避免在无观测收益时增加模型数量。

截图中的now和19h ago不作为精确提交时刻；成绩账本仅记录观察日期。原始图片直接复制保存，未修改像素。本阶段按用户要求暂停报告和源码包整理，现有本地DOCX/ZIP尚未同步第三、四次成绩。

此前平台最佳文件：`outputs/kaggle_ready_4690/submission_ensemble4690.csv`，原始候选位于`outputs/optimization_20261025/submission.csv`，完整模型位于`models/optimization_20261025/ensemble.joblib`，现保留作为对照。

准备文件与原CSV逐字节一致，SHA256为`270a687c3f9a942dfeedea8aef26af70bd24bb2bac14c3d50a249214c3681c82`，含4277名测试乘客。`manifest.json`保留两份CSV哈希及已观察到的成绩；本机outputs按.gitignore忽略。截图本身不能校验Kaggle服务器上文件的字节哈希。

同机隔离目录重训及模型重载已完成，提交CSV逐字节一致；检查见reports/checks/isolated_reproduction_20261028.json。此检查不冒充另一位成员或另一台机器的独立复现。完整进展见[后续训练与交付](后续训练与交付_20261028.md)。

## 课程报告所需材料

作业要求上传的CSV文件名带组长学号后四位；九份提交文件均使用4690。最终报告需要上传文件名与得分截图，以及小组信息截图。当前图片没有展示四人Team成员列表，还需另行保存小组信息。现有本地DOCX和源码ZIP仍暂停更新，尚未纳入本批成绩。

作业中“组长提交即可，不要重复提交”针对雨课堂课程报告和成果物，由组长统一提交。两份作业文件没有规定Kaggle只能提交一次；实际日额度以比赛页面为准。
