"""Publish the completed routing rejection without changing course deliverables."""
import json
import re

from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR

NAME = "g_cohort_routing_20261029"


def main():
    report = EXPERIMENTS_DIR / NAME
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((PROJECT_ROOT / "reports/checks" / f"{NAME}.json").read_text(encoding="utf-8"))
    assert audit["passed"]
    if summary["status"] != "no_candidate":
        raise RuntimeError("A qualified candidate needs a separate submission/reload audit before documentation")
    rows = summary["repeats"]
    table = "\n".join(
        f"| {r['repeat']} | {r['reference_accuracy']:.6f} | {r['accuracy']:.6f} | {r['corrected']} | {r['new_errors']} | {r['net_correct']:+d} | {r['cold_net_correct']:+d} | {r['child_net_correct']:+d} | {r['g_improved_folds']}/5 |"
        for r in rows)
    controls = "\n".join(
        f"第{r['repeat']}次：G准确率 {r['g_reference_accuracy']:.6f} → {r['g_accuracy']:.6f}；"
        f"相对无角色对照的G差值 {r['added_roles_g_gain']:+.6f}；"
        f"整体Log loss {r['reference_log_loss']:.6f} → {r['log_loss']:.6f}，"
        f"G的Log loss {r['g_reference_log_loss']:.6f} → {r['g_log_loss']:.6f}。"
        for r in rows)
    stopped = ("第一次五折结果已触发事前的停止条件，因此没有运行第二次划分。"
               if summary["early_stop"] else "两次固定划分均已完成，但未通过完整门槛。")
    text = f"""# G甲板分群路由验证 2026-10-29

本轮将此前两个开发线索组合成一条固定规则，仍未通过事前门槛。没有生成新Kaggle提交或替换主模型，已观测最佳保持0.81108。{stopped}

## 为什么检验分群

前一轮只用G训练的专用模型在Earth/G未冷冻儿童中净多对5人，却在Earth/G冷冻乘客中少对9人；概率修正在冷冻群体净多对5人。此前各自的初始门槛仍判为失败。本轮根据这些已看过的开发结果提出互补假设，只检验一次固定组合，不把先前失败改写成成功。

## 训练前固定的规则

- 原始Deck=G且已知CryoSleep=True：原概率75%加正则化角色修正概率25%。
- 原始Deck=G且已知CryoSleep=False、已知Age≤12：原概率75%加G专用CatBoost三种子均值25%。
- 其他所有乘客：逐个保留原概率，包括其他G乘客、未知Deck、未知冷冻状态或未知年龄。

两个群体互不重叠；冷冻儿童走第一条。规则没有额外限制HomePlanet，不能把本轮人数与前一轮Earth/G人数混比。正则惩罚固定0.1，专用模型参数沿用前轮，种子42/2026/103，阈值0.5；不搜索年龄边界、权重、分群或配置。

## 验证与隔离

复用此前固定的GroupKFold两次划分，种子20261029/20261106，每次五折。原方案也使用同一划分的缓存概率，按原权重25%/25%/50%组合，保证配对比较。每折使用外层训练内部25%的整组留出数据拟合修正器，其教师模型完全排除这些校准组；外层验证标签不参与教师、专用模型或修正器训练。

公开train+test的无标签特征仅用于统一同行统计，排除自身贡献；不使用同行标签填补验证答案。匹配对照采用相同参数、种子和分群，只移除同行角色增量。

这些原始样本、划分和基准结果此前已被使用，本轮是选择后的配对稳定性检验，不能称为独立新盲测。重新分折也不会凭空产生新的独立样本。

## 实际结果

| 次数 | 原整体准确率 | 路由后准确率 | 修正旧错 | 新增错误 | 整体净多对 | 冷冻净多对 | 儿童净多对 | G改善折数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

{controls}

完整互斥群体结果见[subgroup_results.csv](../reports/experiments/{NAME}/subgroup_results.csv)，所有变化见[changed_predictions.csv](../reports/experiments/{NAME}/changed_predictions.csv)。保留8693人的每次折外预测、逐折结果和修正器系数，失败记录也完整上传。

## 门槛与停止决定

训练前固定：每次整体、冷冻、儿童净多对均不得为负；两次平均整体准确率至少+0.001；10个G折至少6折改善；角色相对对照的平均G准确率至少+0.003；每次整体Log loss恶化不超过0.002、G不超过0.005。

第一次若任一非退化条件或损失条件失败，或G改善折数为零，便提前停止。{stopped}没有事后放宽条件、改权重或用另一个随机划分替换失败结果。代码记录原因：`{summary['reason']}`。

分群能把修改范围限制清楚，却不能保证历史收益跨划分重现。错误集中在G只能说明这一群体难，不能证明增加G权重、同行角色或分群补丁有有效的新信息。本轮也不能证明0.81108已经达到理论上限。

## 当前决定与复现

保留原45模型和原提交，暂时结束这一组G定向补丁。后续若再探索，应先提出尚未检验的信息来源并冻结对照；不继续围绕同一批小幅波动反复选参数或按榜分微调。课程报告DOCX和源码ZIP依用户要求继续暂停整理。

46项测试通过。独立核验检查原始数据和冻结源码哈希、外层与内部组隔离、{audit['teacher_members_checked']}份教师缓存及{audit['specialist_caches_checked']}份专用模型缓存，重算路由与全部指标；未被路由的概率完全不变，原最佳CSV的SHA256仍为`{audit['incumbent_submission_sha256']}`，榜分账本没有变化。

本机复现：

```powershell
python -m unittest discover -s tests -q
python -m scripts.g_routing_study
python -m scripts.audit_g_routing
python -m scripts.summarize_g_routing
```

全新克隆需先下载原始竞赛数据，复现原冻结方案及`python -m scripts.confirm_fixed_weights`的基准缓存；该基准程序自身的失败结论保持不变。协议按文件字节校验源码哈希，克隆时需保留仓库LF换行，避免Git自动CRLF转换改变哈希。模型和概率缓存位于被忽略的models/outputs目录。协议和实验结果位于reports/experiments/{NAME}/，核验记录位于reports/checks/{NAME}.json。
"""
    (PROJECT_ROOT / "docs/G甲板分群路由验证_20261029.md").write_text(text, encoding="utf-8")
    last = rows[-1]
    block = f"""<!-- g-routing-20261029:start -->
## 2026-10-29 G甲板分群路由

固定冷冻乘客概率修正、未冷冻儿童专用模型、其他人沿用原预测。完成{len(rows)}次配对五折，最近一次整体净多对{last['net_correct']:+d}人，冷冻{last['cold_net_correct']:+d}、儿童{last['child_net_correct']:+d}；未通过事前门槛。{stopped}46项测试及缓存重算通过，无新提交，平台最佳保持0.81108。

详见[G甲板分群路由验证](docs/G甲板分群路由验证_20261029.md)。课程报告和源码ZIP仍暂停整理。
<!-- g-routing-20261029:end -->
"""
    for path in [PROJECT_ROOT / "README.md", PROJECT_ROOT / "docs/项目进度.md"]:
        current = path.read_text(encoding="utf-8")
        current = re.sub(r"<!-- g-routing-20261029:start -->.*?<!-- g-routing-20261029:end -->\n*", "", current, flags=re.S)
        first, rest = current.split("\n", 1)
        inserted = block if path.name == "README.md" else block.replace("(docs/G甲板", "(G甲板")
        path.write_text(first + "\n\n" + inserted + "\n" + rest.lstrip(), encoding="utf-8")
    print("Routing outcome and project progress documented")


if __name__ == "__main__":
    main()
