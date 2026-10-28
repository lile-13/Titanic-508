"""汇总本轮实测结果；不修改历史训练分数或Kaggle成绩账本。"""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))
    relation=read("reports/experiments/relational_study_20261028/summary.json")
    diversity=read("reports/experiments/diversity_study_20261028/summary.json")
    full=read("reports/experiments/full_data_candidate_20261028/summary.json")
    repro=read("reports/checks/isolated_reproduction_20261028.json")
    ledger=read("reports/experiments/kaggle_submissions.json")
    observed=next((r for r in ledger['submissions'] if r['model_run']=='full_data_candidate_20261028'),None)
    platform_status=(f"已提交，截图确认得分{observed['score']:.5f}，与原集成持平，本次没有提分；没有新的OOF指标。" if observed
                     else "尚无Kaggle成绩，也没有新的OOF指标。")
    f=diversity["frozen"]
    count=len(list((ROOT/"reports/experiments/diversity_study_20261028").glob("profile_*_seed_*.json")))
    text=f"""# 后续训练与交付 2026-10-28

当前已观测的Kaggle最佳仍为0.81108，原冻结集成与全量重训融合持平，继续保留原45模型集成为主方案。本轮不改写原始本地指标和训练摘要，平台结果独立登记。用户已确认组成四人队伍，并选择自行填写四人姓名、完整学号、班级及贡献率。

## 本轮训练

| 实验 | 本地开发准确率 | 结论 |
|---|---:|---|
| 原冻结集成，同一开发口径 | 0.821999 | 当前参考 |
| 关系特征，最佳三种子融合 | {relation['frozen']['development']['accuracy']:.6f} | 未达到预设门槛 |
| 异构模型，最佳三种子融合 | {f['development']['accuracy']:.6f} | {diversity['status']} |

关系特征实验先比较8种配置，对前两种补充种子，共12项配置与种子记录。姓氏目标统计按GroupId交叉拟合；自己的整个同行组标签均不进入自身编码。

异构模型比较14种初始配置，包括6项SVM、2项ExtraTrees、2项HistGradientBoosting及4项CatBoost不同训练方式；连同前两种方案补充种子，共{count}项配置与种子记录。概率校准、缩放与填补均不读取外层验证标签。开发期排除第5折；历史折已经被查看，属于追加开发，不是新的盲测。协议在训练前固定，原始记录见各自实验目录。

## 新增一份训练方式对照文件

`outputs/kaggle_ready_4690/submission_fullblend4690.csv`

沿用原冻结参数，用全部8693条训练数据训练9个模型，与原45模型五折集成各占50%融合。组合共54个模型，阈值固定0.5；没有根据平台分数或测试标签调整参数。相比原提交有{full['changed_test_labels']}个预测标签变化，共{full['rows']}行。模型重载的概率最大误差为{full['probability_max_reload_error']:.3g}。

**{platform_status}** 这是单独的训练策略对照，未参加上述算法筛选门槛，不能借用原OOF宣称提升。当前原集成45个模型，对照54个模型；平台持平时保留原方案。原始截图见`reports/figures/kaggle_fullblend_20261028.png`，成绩账本见`reports/experiments/kaggle_submissions.json`；截图无法校验平台实际接收的文件字节。

原已验证文件：`outputs/kaggle_ready_4690/submission_ensemble4690.csv`。

## 复现与修复

新增 `scripts/refit_frozen.py`，从历史冻结配置在全新run-name下重训，避免只读到已完成摘要却没有生成本机权重。修复全新源码目录没有models父目录时保存失败的问题；源码哈希检查只允许Git的LF/CRLF换行差异。

已在无模型权重和无预测缓存的隔离源码目录完成训练，OOF与测试概率完全一致，提交CSV逐字节一致，SHA256为`{repro['submission_sha256']}`。隔离检查使用同一台电脑和Python环境，不冒充其他成员或跨机器复现。详见`reports/checks/isolated_reproduction_20261028.json`。

回归测试19项通过，命令为`python -m unittest discover -s tests -v`。新增测试覆盖自己的同行组标签排除、未知关系键回退、训练折缩放不被验证数据改变以及仅允许换行变化的哈希检查。当前环境未安装pytest，项目使用unittest执行测试。

## 课程交付 本阶段暂停整理

报告：`outputs/delivery_20261028_4690/大作业综合报告-待填班级-待填学号-待填姓名.docx`。

源码包：`outputs/delivery_20261028_4690/大作业源代码-待填班级-待填学号-待填姓名.zip`。

报告沿用学校模板、封面、页眉页脚和评分附录；替换错误的891条传统Titanic示例，保留真实数据、实验结果和成绩截图，清空模板预填评分。目录由Word更新。还需组长填写个人信息、补Team页面截图，并按真实信息重命名两份文件；贡献率不得由工具推断。

源码包包含源代码、公开原始数据、折划分、实验图表、当前最佳模型、训练方式对照模型、提交CSV和SHA256清单；排除.venv、Git和中间搜索缓存。保存模型预测可直接复现；重做搜索需按原搜索命令先生成开发缓存。报告重建需额外的python-docx与Pillow及Word，不属于模型训练依赖。

组长已手动完成三次Kaggle提交。本阶段按组长要求先同步GitHub源码、实验记录和成绩截图，暂停报告与源码包的后续整理；已有本地报告与ZIP尚未纳入第三次成绩，不应直接当作最新交付。models与outputs继续被Git忽略。
"""
    (ROOT/"docs/后续训练与交付_20261028.md").write_text(text,encoding="utf-8")
    block=f"""<!-- followup-20261028:start -->
## 2026-10-28 后续工作

用户确认四人队伍，后续由统一流程完成；个人信息由组长自行填写。关系特征和异构模型补充实验、隔离目录重训及报告交付见[后续训练与交付](docs/后续训练与交付_20261028.md)。当前Kaggle已观测最佳仍为0.81108。

全量重训对照`submission_fullblend4690.csv`采用全量模型与原五折集成各50%，有{full['changed_test_labels']}个测试标签变化。{platform_status} 继续保留原45模型集成为主方案。回归测试19项通过，隔离重训的原最佳CSV逐字节一致。本阶段暂停报告及源码包整理，先同步GitHub源码与实验记录；权重、CSV及本地文档保存在被忽略的models与outputs目录。
<!-- followup-20261028:end -->
"""
    path=ROOT/"README.md";old=path.read_text(encoding="utf-8")
    old=re.sub(r"<!-- followup-20261028:start -->.*?<!-- followup-20261028:end -->\n*","",old,flags=re.S)
    old=old.replace("# Titanic-508\n# Spaceship Titanic 四人团队项目","# Titanic-508 Spaceship Titanic 四人团队项目",1)
    first,rest=old.split("\n",1);path.write_text(first+"\n\n"+block+"\n"+rest.lstrip(),encoding="utf-8")
    path=ROOT/"docs/项目进度.md";old=path.read_text(encoding="utf-8")
    old=re.sub(r"<!-- followup-20261028:start -->.*?<!-- followup-20261028:end -->\n*","",old,flags=re.S)
    first,rest=old.split("\n",1)
    rest=rest.replace("以下保留此前阶段记录，待办状态以本次更新为准。\n\n", "",1)
    progress=block.replace("(docs/后续训练与交付_20261028.md)","(后续训练与交付_20261028.md)")
    path.write_text(first+"\n\n"+progress+"\n以下保留此前阶段记录，待办状态以本次更新为准。\n\n"+rest.lstrip(),encoding="utf-8")


if __name__=="__main__":main()
