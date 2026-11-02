"""从课程参考模板制作报告；使用 Codex 捆绑 Python 运行，随后由 Word 更新域。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"outputs/report_20261102_4690"
WORK=DEST/"work"
SOURCE=ROOT/"outputs/delivery_20261028_4690/work/reference_report.docx"
FINAL=DEST/"大作业综合报告-待填班级-待填学号-待填姓名.docx"


def draw_workflow():
    canvas=Image.new("RGB",(1860,390),"white");draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",34)
    steps=[("原始数据","8693 / 4277"),("统一特征计数","公开无标签池"),("折内数据处理","填补与编码"),("分组验证冻结","参数与权重"),("模型与平台验证","重载及提交")]
    for i,lines in enumerate(steps):
        x=15+i*374
        draw.rectangle((x,55,x+330,290),outline="#202020",width=3)
        for j,text in enumerate(lines):draw.text((x+165,110+j*80),text,font=font,fill="black",anchor="mm")
        if i<4:
            draw.line((x+333,175,x+368,175),fill="black",width=4)
            draw.polygon([(x+368,175),(x+355,166),(x+355,184)],fill="black")
    canvas.save(ROOT/"reports/figures/course_workflow_20261028.png")


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    template=SOURCE if SOURCE.exists() else ROOT/"resources/course_report_template.docx"
    doc=Document(template)
    sources=json.loads((WORK/"source_manifest.json").read_text(encoding="utf-8"))
    for relative,digest in sources.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    ledger=json.loads((ROOT/"reports/experiments/kaggle_submissions.json").read_text(encoding="utf-8"))
    assert len(ledger["submissions"])==13 and ledger["current_platform_best"]["score"]==.81318
    metadata=json.loads((WORK/"report_metadata.json").read_text(encoding="utf-8"))
    members={m["number"]:m for m in metadata["members"]}
    assert set(members)=={2,3,4}
    source_paragraphs=list(doc.paragraphs)
    hprops={level:deepcopy(source_paragraphs[index]._p.pPr) for level,index in [(1,28),(2,46),(3,47)]}
    cover=doc.tables[0]
    def cell_text(cell,text):
        p=cell.paragraphs[0]
        fmt=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
        for child in list(p._p):
            if child.tag!=qn("w:pPr"):p._p.remove(child)
        r=p.add_run(text)
        if fmt is not None:r._r.insert(0,fmt)
        for old in list(cell._tc)[2:]:
            if old.tag==qn("w:p"):cell._tc.remove(old)
    cell_text(cover.cell(1,1),"Spaceship Titanic 乘客传送预测")
    selection=cover.cell(1,1).paragraphs[0]
    selection.style=doc.styles["Normal"]
    outline=OxmlElement("w:outlineLvl");outline.set(qn("w:val"),"9");selection._p.get_or_add_pPr().append(outline)
    for run in selection.runs:run.font.color.rgb=__import__('docx').shared.RGBColor(0,0,0)
    for row in range(6,10):
        for col in range(1,5):cell_text(cover.cell(row,col),"待填")
        cell_text(cover.cell(row,5),"")
    for member in members.values():
        row=member["cover_row"]
        cell_text(cover.cell(row,1),member["name"])
        cell_text(cover.cell(row,2),member["student_id"])
    rubric=doc.tables[2]
    for row in range(2,7):cell_text(rubric.cell(row,3),"")
    body=doc._element.body
    start=source_paragraphs[28]._p
    boundary=source_paragraphs[96]._p
    removing=False
    for child in list(body):
        if child is start:removing=True
        if child is boundary:break
        if removing:body.remove(child)
    # Replace cached TOC with a fresh field. Native Word constructs the page numbers.
    for p in source_paragraphs[11:27]:
        if p._p.getparent() is body:body.remove(p._p)
    toc=doc.add_paragraph()
    begin=OxmlElement("w:fldChar");begin.set(qn("w:fldCharType"),"begin")
    instruction=OxmlElement("w:instrText");instruction.set(qn("xml:space"),"preserve")
    instruction.text=' TOC \\o "1-3" \\h \\z \\u '
    end=OxmlElement("w:fldChar");end.set(qn("w:fldCharType"),"end")
    for e in [begin,instruction,end]:toc.add_run()._r.append(e)
    source_paragraphs[27]._p.addprevious(toc._p)

    def p(text="",level=0,newpage=False,center=False,bold=False,size=10.5,keep=False):
        item=doc.add_paragraph()
        boundary.addprevious(item._p)
        if level:
            item._p.insert(0,deepcopy(hprops[level]))
            item.paragraph_format.keep_with_next=True
            item.paragraph_format.first_line_indent=Pt(0)
        else:
            item.paragraph_format.line_spacing=1.5
            item.paragraph_format.first_line_indent=Pt(21 if not center else 0)
            item.paragraph_format.space_after=Pt(5)
        item.paragraph_format.widow_control=True
        item.paragraph_format.page_break_before=bool(newpage)
        if center:item.alignment=WD_ALIGN_PARAGRAPH.CENTER
        if keep:item.paragraph_format.keep_with_next=True
        r=item.add_run(text)
        r.font.name="Times New Roman"
        r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"),"黑体" if level else "宋体")
        r.font.size=Pt({1:15,2:14,3:12}.get(level,size))
        r.bold=bool(level or bold)
        r.font.color.rgb=__import__('docx').shared.RGBColor(0,0,0)
        return item

    def table(headers,rows,widths):
        t=doc.add_table(rows=1,cols=len(headers))
        boundary.addprevious(t._tbl)
        t.autofit=False
        for c,w in zip(t.columns,widths):c.width=Inches(w)
        borders=OxmlElement("w:tblBorders")
        for edge in ["top","left","bottom","right","insideH","insideV"]:
            e=OxmlElement("w:"+edge);e.set(qn("w:val"),"single");e.set(qn("w:sz"),"4");e.set(qn("w:color"),"000000");borders.append(e)
        t._tbl.tblPr.append(borders)
        margins=OxmlElement("w:tblCellMar")
        for edge in ["top","bottom","left","right"]:
            e=OxmlElement("w:"+edge);e.set(qn("w:w"),"85");e.set(qn("w:type"),"dxa");margins.append(e)
        t._tbl.tblPr.append(margins)
        for vals,row in [(headers,t.rows[0])]+[(vals,t.add_row()) for vals in rows]:
            no_split=OxmlElement("w:cantSplit");row._tr.get_or_add_trPr().append(no_split)
            for i,(c,text) in enumerate(zip(row.cells,vals)):
                c.width=Inches(widths[i]);c.vertical_alignment=1
                para=c.paragraphs[0];para.paragraph_format.line_spacing=1.2
                para.paragraph_format.space_after=Pt(2);para.paragraph_format.space_before=Pt(2)
                r=para.add_run(str(text));r.font.name="Times New Roman";r.font.size=Pt(10)
                r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"),"宋体")
                r.bold=row is t.rows[0]  # explicitly set below as row proxies are reconstructed
        for r in t.rows[0].cells:
            for paragraph in r.paragraphs:
                paragraph.paragraph_format.keep_with_next=True
                for run in paragraph.runs:run.bold=True
        repeat=OxmlElement("w:tblHeader");t.rows[0]._tr.get_or_add_trPr().append(repeat)
        return t

    def image(path,caption,width=5.9):
        item=p(center=True,keep=True)
        item.add_run().add_picture(str(ROOT/path),width=Inches(width))
        p(caption,center=True,size=10)

    p("一．问题简介、目标",1)
    p("本项目选择Kaggle Spaceship Titanic竞赛，根据乘客的出发星球、冷冻睡眠状态、舱位、年龄及消费等属性，预测其是否被传送至另一维度。目标变量为Transported，属于二分类问题。实验使用8693条带标签训练记录和4277条待预测记录，完成了数据探索、清洗、特征构造、模型比较、概率融合与平台提交。")
    p("截至2026年11月1日，小组共完成13次有截图确认的平台提交，最高得分为0.81318。最终采用原树模型集成25%与TabICL算术概率平均75%的融合，分类阈值为0.5。相比首次CatBoost提交的0.80313，提高0.01005，即1.005个百分点；后续探索未超过该成绩，本轮实验结束并转入成果整理。")
    p("项目同时记录本地验证、平台成绩和复现证据。曾提出超过0.82的探索目标，但实际尚未达到，报告以已经完成的实验与真实得分为结论。")

    p("二．小组成员工作",1)
    p("项目使用Litianle4690账户提交，文件名包含组长学号后四位4690。四人小组按下表分工开展工作；组长信息、班级及实际贡献率在提交前补齐。")
    p("表1 小组任务模块分工",center=True,size=10,keep=True)
    table(["角色","姓名及学号","职责范围"],[
        ["组长","待填","统一建模入口、集成优化、平台提交、实验与报告整合"],
        ["成员2",members[2]["name"]+"\n"+members[2]["student_id"],"数据体检、属性及消费分析、缺失值方案、训练测试分布检查与预处理"],
        ["成员3",members[3]["name"]+"\n"+members[3]["student_id"],"舱位及同行组特征、统一特征函数、固定划分消融、最终特征字典"],
        ["成员4",members[4]["name"]+"\n"+members[4]["student_id"],"随机森林与LightGBM对照、误差分析、概率平均融合及建议"]
    ],[.7,1.25,4.35])
    p("后续整合与新增实验使用了代码和文档辅助工具。表中的模块安排不作为个人贡献率的自动判定依据，报告中的数值均以保存的实验产物和评分截图为准。")

    p("三．问题分析流程设计",1,newpage=True)
    p("整个流程由数据检查开始，先确认字段含义、缺失模式和同行关系，再建立统一的特征口径与按组验证方案。模型开发中比较树模型和预训练表格模型，保存折外预测后进行概率融合；只有核验通过的文件才进入手动平台提交。每次真实评分单独登记，避免把准备期的候选状态误当作实际成绩。")
    image("reports/figures/course_workflow_20261028.png","图1 数据分析与模型实现流程",6.1)
    p("表2 主要阶段及其产物",center=True,size=10,keep=True)
    table(["阶段","主要操作","可复查产物"],[
        ["数据检查","字段、缺失、标签和分组关系分析","统计表、可视化、原始数据哈希"],
        ["特征构造","拆分舱位、汇总消费、统一无标签计数","特征字典、折内输入构造器"],
        ["分组验证","按完整GroupId隔离训练与验证","固定划分、逐折指标和OOF概率"],
        ["模型比较","树模型搜索、TabICL聚合及互补对照","冻结配置、成功与失败实验摘要"],
        ["预测核验","重载模型、校验CSV、登记平台结果","4277行预测、SHA256、原始截图"]
    ],[.85,2.6,2.85])
    p("训练集有6217个GroupId，测试集有3063个，交集为0。因此不能将某个训练组的标签直接传播到测试组。验证时将完整同行组分配到同一折，可以减少相关乘客同时出现在训练与验证两侧造成的乐观估计；姓氏可能跨组出现，不能直接把同姓等同于同行。")
    p("无标签的GroupSize、SurnameSize等人口统计在公开train与test特征池中统一计算，不使用Transported。填补器、编码器以及监督统计仅利用相应训练折拟合，验证折和测试集复用。此设计针对本竞赛已公开的待预测批次，属于利用待预测特征分布的处理方式，不能直接假设对任意未来单行输入同样有效。")

    p("四．问题分析实现步骤",1,newpage=True)
    p("4.1 数据集初步探索",2)
    p("1. 数据获取与字段说明",3)
    p("原始数据来自Kaggle Spaceship Titanic竞赛[1]，包含train.csv、test.csv和sample_submission.csv。训练集有14列，测试集有13列，差异为训练集包含目标Transported。PassengerId按字符串读取，保留组号和组内序号的前导零；其前半部分用于构造GroupId。")
    p("表3 原始字段及建模用途",center=True,size=10,keep=True)
    table(["字段类别","字段","处理方式"],[
        ["标识与姓名","PassengerId、Name","拆出同行组与姓名类别；不将原始ID作为唯一身份标签"],
        ["类别属性","HomePlanet、CryoSleep、Destination、VIP","原生类别处理或按训练折编码"],
        ["舱位信息","Cabin","拆成Deck、CabinNum、Side，并构造组合特征"],
        ["年龄","Age","保留连续值，剩余缺失由折内方案处理"],
        ["五项消费","RoomService、FoodCourt、ShoppingMall、Spa、VRDeck","保留分项，增加总消费、零消费及变换特征"],
        ["预测目标","Transported","True为已传送，False为未传送；仅训练时使用"]
    ],[.8,2.6,2.9])
    p("2. 标签与缺失分布",3)
    p("训练集中True为4378人、False为4315人，类别接近平衡。Age有8514个非缺失值，均值约28.83岁，中位数27岁，范围为0至79岁。五项消费的中位数均为0，均值明显高于中位数，说明分布右偏；仅凭高消费数值不能认定记录错误，因此不直接删除这些乘客。")
    image("reports/figures/01_missing_rate.png","图2 原始字段缺失率",5.8)
    p("HomePlanet缺失201条，CryoSleep缺失217条，Cabin缺失199条，Age缺失179条，五项消费分别缺失181、183、208、183、188条。缺失比例总体较低，但涉及类别和数值两类字段，需要区分未知值与真实的零消费。")
    p("进一步按原始非缺失CryoSleep分组，3037名冷冻乘客的Transported正类比例约为81.76%，5439名非冷冻乘客约为32.89%。在五项消费均有记录的7785人中，3247名总消费为0的乘客正类比例约为78.38%，其余4538人约为30.12%。这些关联提示冷冻状态和消费信息值得保留，但不是因果结论，也不能据此把某一类乘客全部预测为同一标签。")

    p("4.2 数据预处理与特征工程",2)
    p("1. 缺失值处理",3)
    p("优先采用可以从现有字段核验的规则：已知CryoSleep为True时，将缺失消费填为0；已知某项消费为正时，将缺失CryoSleep填为False。同组分类补缺只传播观察到唯一值的HomePlanet和Side，其余缺失采用训练折统计量或模型原生缺失机制处理。规则不会用验证标签或测试标签决定填充值。")
    p("同行组内的Deck和Destination可能不同，不能按组强制覆盖。修正缺失标记确实进入模型后，HistGB五折对照中，不填充、简单填充、增加缺失标记、规则及组内填补的平均准确率分别约为0.8031、0.8058、0.8063、0.8050。增加标记相对简单填充约提高0.05个百分点，规则填补则略低于简单填充，差异不足以证明稳定改善或退化。这些结果属于特定模型与划分，不能直接叠加到最终模型成绩上。")
    p("对照结果来自修正后的missing_strategy_comparison.csv。早期实现曾生成缺失标记但未将其纳入建模矩阵，因此旧版所谓“缺失标记零收益”不能作为有效结论。修正后检查每折输入，保证标记列真正进入模型；随后将主要精力转向特征口径和模型验证。")
    p("2. 特征构造与统计口径",3)
    p("Cabin拆分为甲板Deck、舱位编号CabinNum和左右舷Side，便于模型分别学习类别与位置关系。消费侧构造TotalSpend等汇总特征，保留五个原始分项，以免总量掩盖消费结构；对树模型进一步比较spend和rich_names等特征视图。TabICL最终使用固定的18列精简输入，未将尝试过的所有扩展列都加入最终方案。")
    p("SurnameSize最初在train和test分别计算，两侧均值约为5.33和3.51，由于样本量不同形成了区分数据来源的指纹。在保留有效缺失标记的修正后对照中，统一计数使对抗验证AUC由0.7404降至0.5505，与图3及adversarial_comparison.csv一致。这里的对抗验证训练模型识别一行来自train还是test，AUC接近0.5表示在该检验下较难区分来源；它不是Transported预测准确率，也不能证明两侧分布完全相同。")
    image("reports/figures/04_adv_fix_comparison.png","图3 统一姓氏计数前后的对抗验证对照",5.6)
    p("3. 防止监督信息进入验证折",3)
    p("对需要标签的姓名目标统计、残差效应或难例权重，均在外层训练折内部再次按组构造折外预测或交叉拟合统计。外层验证乘客及其同行组标签不参与自身输入。全局错误名单只用于事后诊断，不作为人工修改测试标签或训练路由规则的依据。")

    p("4.3 构建分析模型及预测",2)
    p("1. 基线模型与树模型集成",3)
    p("先使用逻辑回归建立线性参考，再比较随机森林、LightGBM和CatBoost。CatBoost使用梯度提升树并支持类别特征处理[2]，适合本任务中类别与数值混合的输入；LightGBM使用直方图分裂等机制提高训练效率[3]。两者的预处理和特征视图不同，因此下表比较的是完整建模方案，不是完全相同矩阵上的纯算法差异。")
    p("表4 早期模型五折验证结果",center=True,size=10,keep=True)
    table(["方案","五折准确率均值","说明"],[
        ["逻辑回归 59列","0.7954","线性参考"],
        ["随机森林 59列","0.7874","树模型对照"],
        ["LightGBM 59列","0.8108","独立对照流程"],
        ["CatBoost 30列","0.8163","原始固定参数基线"],
        ["原45成员树集成","0.8157","经模型选择后的描述性结果"]
    ],[2.3,1.5,2.5])
    p("树模型优化阶段比较32组CatBoost和48组LightGBM配置，并考察基础、消费、位置和姓名信息。最终冻结两个CatBoost配置与一个LightGBM配置，分别为spend/native的555轮、rich_names/native的748轮及base/rule_group的327轮；内部家族权重为0.25、0.25、0.50。三个配置在五折与种子42、2026、103上训练，共45个树成员。LightGBM三种子的预测相同，未将其描述为额外的独立多样性来源。")
    p("2. 按组验证与指标解释",3)
    p("以准确率作为主要指标，即预测正确人数占参与评估样本的比例，同时记录Log loss观察概率质量。每个方案保留逐折结果、整体折外预测及修正旧错与新增错误人数；净收益必须同时考虑两者，不能只列被新模型改对的样本。")
    p("树模型开发阶段主要在第1至4折选择配置，再评估历史确认折。该第5折曾在早期基线中被查看，因此属于历史确认，不是从未使用的盲测。相同开发口径下，CatBoost准确率为0.811646、原树集成为0.821999；历史确认折则由0.807825到0.808976，只净多对2人，包含32个修正和30个新增错误。")
    image("reports/figures/optimization_20261025_fold_comparison.png","图4 原CatBoost与树集成的逐折准确率",5.7)
    p("原树集成最终五折均值为0.815713，标准差0.011160，样本加权OOF准确率为0.815714。它在本地并未全面优于CatBoost，但平台成绩从0.80313提高到0.81108。此差异说明，本地某次验证排序不能保证与平台排序一致。开发与确认结果应连同各自划分、模型选择过程一起解释。")

    p("3. 引入TabICL及概率聚合改进",3)
    p("在树模型之外，引入作者公开的TabICLv2预训练分类权重。TabICL把带标签的训练样本作为上下文，在预测前向计算中利用这些样本，不需要对每个新数据集重新预训练神经网络[4-5]。本项目的最终方案没有更新预训练网络参数，而是保存每折训练上下文、输入构造器和固定推理设置。")
    p("每个TabICL上下文保留8个生成视图，使用18列精简输入、种子42和算术概率平均。早期诊断发现，在一个表现较差的历史折中，有一个视图对264名冷冻G甲板乘客给出了极低的正类概率，平均约0.001185；默认的logit聚合放大了这一极端视图的影响。使用同样全部8个视图进行概率平均后，该子群平均概率由约0.3730变为0.5072。")
    p("改动只涉及视图聚合，不根据标签删除视图，也未把该已知坏折作为独立验证。随后在预先固定的分组对照中检查整个方案，原树模型与TabICL各占50%。下表记录两套历史分组及两套随后使用的分组；每套均包含全部8693名乘客，属于同一数据上的重复划分。")
    p("表5 树模型与TabICL各50%的分组对照",center=True,size=10,keep=True)
    aggregation=json.loads((ROOT/"reports/experiments/tabicl_probability_mean_20261030/summary.json").read_text(encoding="utf-8"))
    comparison=[]
    for stage,title in [("historical","历史分组"),("fresh","随后分组")]:
        for row in aggregation["stages"][stage]:
            comparison.append([title+str(row["repeat"]),f"{row['reference_accuracy']:.6f}",f"{row['accuracy']:.6f}",f"{row['corrected']-row['new_errors']:+d}",str(row["improved_folds"])+"/5"])
    table(["划分","原树准确率","新融合准确率","净多对人数","改善折数"],comparison,[1.05,1.25,1.3,1.35,1.35])
    p("该50%融合随后在平台得到0.81178。两套随后分组平均准确率增益约0.002703，合计8/10折改善，但按同行组配对重采样得到的条件95%区间约为[-0.000753, 0.006212]，仍包含0。该区间固定了已有预测，没有计入全部模型选择与重训不确定性，因此不能据此宣称稳定显著提高。")

    p("4. 最终融合模型与平台选择",3)
    p("在后续公开榜单探索中，原树25%与TabICL75%的方案得到0.81318，为本项目最高已观测成绩。因此最终保留该方案。原45个树成员与5个TabICL上下文构成50个顶层成员；每个TabICL上下文在总概率中权重为0.15，五个合计0.75。各上下文内部仍保留全部8个视图，最终以0.5为阈值输出True或False。")
    p("表6 最终模型的概率权重",center=True,size=10,keep=True)
    table(["组件","构成","最终总权重"],[
        ["CatBoost spend","5折×3种子，555轮","6.25%"],
        ["CatBoost rich_names","5折×3种子，748轮","6.25%"],
        ["LightGBM base","5折×3种子，327轮","12.50%"],
        ["TabICLv2","5个训练上下文，每个8视图","75.00%"]
    ],[2.15,2.75,1.4])
    p("这一权重是查看公开榜单反馈后的观察性选择，不是新的独立盲测胜出。该方案在四套历史分组上的平均准确率约为0.815397，增益方向并不全部为正；因此没有把平台胜出追认为所有本地稳定性规则均已通过，也不将这一均值与采用其他划分的早期五折分数直接排序。")
    p("最终模型保存在models/manual_submission_batch_20261031_tabicl75/ensemble.joblib，对应提交文件为submission_tabicl75_4690.csv。独立重载可以复现已保存提交文件；原树模型、50%融合模型、失败方案及准备期摘要均保留，平台结果通过独立成绩台账追加。")

    p("5. 后续实验与停止依据",3)
    p("为寻找互补信号，后续比较了TabM、TabPFN、更多TabICL视图或种子、姓名残差和不同训练上下文。各研究先固定配置与准入条件，再计算结果；若首套失败则停止，不根据已见答案挑选最有利的一折、检查点或错误个体。下表保留代表性结论，所有净变化均只相对于该研究自己的基线。")
    p("表7 代表性后续对照及结论",center=True,size=10,keep=True)
    table(["方向","主要观察","采用结论"],[
        ["TabM互补","四套历史对照净变化0、+21、+15、-4","未通过后续稳定性门槛"],
        ["TabPFN V2 default","25%混合四套净变化+11、+30、+11、-7","本地未晋级；另作平台探索"],
        ["TabICL 32视图","首套净少对18人，0/5折提高","停止，保留8视图"],
        ["固定三种子上下文","80个修正、102个新增错，净-22；G净-18","停止"],
        ["训练内姓名残差","60个修正、59个新增错，净+1；弱于截距对照9人","停止"],
        ["三份不同小权重TabPFN","53个修正、53个新增错，净0；弱于default对照2人","停止，未生成CSV"],
        ["更高TabICL占比及全上下文","四份最新平台成绩均低于0.81318","保留当前最佳"]
    ],[1.65,3.05,1.6])
    p("其中，姓名残差在每个外层训练集内部使用三折TabICL教师生成训练内折外概率，再拟合有收缩的姓名效应，并设置仅截距的匹配对照。该方案整体只净多对1人，且比对照少对9人，不能把局部子群改善写成发现了有效姓名信号。")
    p("最后一项不同权重研究先核查作者TabPFN V2仓库和文件哈希[6]，排除同内容的finetuned别名，固定平均三份小于14 MB且张量状态确实不同的检查点。完整首套五折整体持平、G净少对1人，比同输入同划分的default对照少对2人，按预设门槛停止。权重不同并不自动带来有效互补，Log loss改善也不等同于准确率提高。")
    p("历史持续错误诊断基于较早50%融合方案：同一批乘客在四套划分中有1203人四次均预测错误。该结果提示有难以利用现有信息区分的区域，但不能证明标签噪声或准确率上限；也不能由训练错例身份推断测试集中哪些乘客错误。本轮停止的依据是近期方案未获得足够稳定净收益，而不是已经证明所有方法都无法提高。")

    p("6. 预测核验与复现",3)
    p("测试预测按sample_submission.csv的PassengerId顺序输出，共4277行，仅含PassengerId和Transported两列。提交前检查ID唯一、顺序一致、没有缺失及布尔标签合法；文件名带有4690。最终文件SHA256为abd9924480c5ab75bb677384e9251b81ab3195bc25c6e750834b48f55ac807cf，便于确认本地使用的具体版本。")
    p("实验截至11月1日的已有66项测试通过。最终最佳模型已独立加载并重放CSV；最近三小权重研究另外重载15个新上下文与5个default上下文，检查8693行验证分组、训练输入、缓存哈希和概率重放通过。这些核验能够证明实现与保存产物一致，不代表新的独立测试得分。")
    p("项目区分常规树模型环境和GPU基础模型环境。已有GPU环境为outputs/tabicl_env，使用torch 2.14.1+cu130、tabicl 2.2.0及RTX 5060 Laptop 8GB。TabICL使用固定作者检查点tabicl-classifier-v2-20260212.ckpt，来源、修订号及SHA256保存在相应pretrained_source.json；本项目未重新预训练该网络。")
    p("在已保存本机数据、训练上下文、作者权重及GPU环境的条件下，可在项目根目录使用以下命令重载最终模型。输出目录必须尚不存在，避免覆盖原预测。")
    command=(r".\outputs\tabicl_env\Scripts\python.exe -B -m scripts.predict_ensemble "
             r"--bundle models/manual_submission_batch_20261031_tabicl75/ensemble.joblib "
             r"--output-dir outputs/report_best_replay")
    item=p(command,size=9);item.paragraph_format.first_line_indent=Pt(0)
    item.alignment=WD_ALIGN_PARAGRAPH.LEFT
    p("公开仓库主要保存源码、协议、实验记录和截图，data、models和outputs目录通常被忽略，仅clone仓库并不具备全部复现条件。该最终混合模型已完成同机重载复现；先前隔离源码目录从零重训核验针对历史树模型，不将两者混同为另一台机器复现最终混合模型。")

    p("五．问题分析结果",1,newpage=True)
    p("1. 平台成绩与最终选择",3)
    p("13次提交均由组长手动完成，文件名、Complete状态和显示分数以原始截图为依据。最高得分0.81318，相比最初0.80313提高1.005个百分点，距离探索目标0.82仍差0.00682。截图中的相对时间不足以确定准确提交时刻，报告不填写未知排名，也不推断私有测试成绩或平台接收文件的字节哈希。")
    p("表8 全部已完成平台提交",center=True,size=10,keep=True)
    labels=["原CatBoost","原树集成","全量树与五折各半","局部LightGBM混合","树与TabICL各半","TabPFN25%探索","TabPFN50%探索","纯TabPFN","树25%与TabICL75%","TabICL87.5%","全训练上下文混合","三模型互补","纯TabICL"]
    scores_table=table(["序号","方案","提交文件名","Kaggle得分"],[[str(i),label,row["uploaded_filename"],f"{row['score']:.5f}"] for i,(label,row) in enumerate(zip(labels,ledger["submissions"]),1)],[.4,1.7,3.25,.95])
    for cell in scores_table.rows[9].cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:run.bold=True
    p("后四份文件分别得到0.81295、0.81271、0.81295、0.81155，均未超过当前最佳。提高TabICL占比、增加全量训练上下文、加入小比例TabPFN没有在本批平台实验中产生进一步提升，因此最终保留树25%与TabICL75%的方案。完整上下文方案没有对应其最终训练规模的独立OOF指标，不能借用其他方案的OOF成绩。")
    p("2. 平台评分证据",3)
    image("reports/figures/kaggle_manual_batch_20261101.png","图5 四份探索提交成绩及0.81318最佳记录",6.15)
    p("图5保留原始截图，可见账户、文件名及Complete评分。TabPFN25%和50%均为0.81271，纯TabPFN为0.80921，TabICL75%为0.81318。选择当前最佳使用了公开平台反馈，属于观察性结果，不将其表述为未受选择影响的独立泛化估计。")
    image("reports/figures/kaggle_next_batch_feedback_20261101.png","图6 后续四份候选的实际平台成绩",6.15)
    p("原始图片与本地预期CSV哈希一并保存，但截图不能证明服务器上所上传文件的实际字节内容，因此结构化记录中的uploaded_bytes_verified仍为false。此限制不影响如实报告截图显示的分数。")
    p("3. 小组信息",3)
    member_suffixes="、".join(members[n]["student_id"][-4:] for n in [2,3,4])
    p(f"待插入Kaggle Team页面的四人小组截图，并核对队名以组长学号后四位结尾、三名组员显示名分别以{member_suffixes}结尾。封面三名组员的姓名和学号已填写；组长信息、班级和贡献率在最终提交前由小组补齐。")

    p("4. 结果讨论",3)
    p("本项目最有依据的改进包括统一姓氏统计口径、按完整同行组验证，以及发现TabICL内部极端视图对logit聚合的影响并改用算术概率平均。原树集成提供了稳定的基础预测，TabICL提供了不同的建模方式，最终通过平台反馈选择两者的融合比例。这些改进均能对应到代码、保存预测和评分证据。")
    p("本地与平台排序不完全一致，是实验中需要保留的重要现象。例如，原树集成本地五折均值略低于CatBoost，但平台明显提高；TabPFN若干混合方案在历史划分上优于部分对照，平台仍低于TabICL75%。因此，报告同时展示本地证据与平台观测，不只选择最有利的数字。")
    p("主要局限包括样本规模有限、历史数据和划分被反复查看、参数与融合选择存在选择偏差，以及公开榜单只反映特定评分样本。四套分组重复仍是同一批8693人，不是四份独立数据；子群误差分析也不能用来宣称已经知道测试错误位置。未达到0.82的目标，不等同于已经证明该数据集的理论上限。")
    p("本轮实验完成了从原始表格到最终提交文件的可复查流程，确认的最高平台成绩为0.81318。后续没有找到足够稳定的新净收益，因而结束继续提分，保留成功与失败记录，并将工作转入课程报告及成果物整理。")
    p("参考资料与产物索引",bold=True,size=12)
    for text in [
        "[1] Kaggle. Spaceship Titanic竞赛与数据：https://www.kaggle.com/competitions/spaceship-titanic。",
        "[2] CatBoost. How training is performed：https://catboost.ai/docs/en/concepts/algorithm-main-stages。",
        "[3] LightGBM. Features：https://lightgbm.readthedocs.io/en/latest/Features.html。",
        "[4] Qu J, Holzmüller D, Varoquaux G, Le Morvan M. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. ICML 2025. https://arxiv.org/abs/2502.05564。",
        "[5] Qu J, Holzmüller D, Varoquaux G, Le Morvan M. TabICLv2: A better, faster, scalable, and open tabular foundation model. https://arxiv.org/abs/2602.11139。",
        "[6] Prior Labs. TabPFN V2公开权重与许可：https://huggingface.co/Prior-Labs/TabPFN-v2-clf。Built with PriorLabs-TabPFN。",
        "[7] 成绩台账：reports/experiments/kaggle_submissions.json；原始截图：reports/figures/kaggle_*.png。",
        "[8] 关键实验：reports/experiments/optimization_20261025、tabicl_probability_mean_20261030、manual_submission_batch_20261031、tabpfn_diversity_20261101。",
        "[9] 核验：reports/checks/next_directions_20261101.json、kaggle_next_batch_result_20261101.json、continuation_20261101.json与tabpfn_diversity_20261101.json。"
    ]:
        item=p(text,size=9.5);item.alignment=WD_ALIGN_PARAGRAPH.LEFT
    p("实验与平台记录截至2026年11月1日，报告整理日期为2026年11月3日。封面保留课程模板预印日期。",size=9.5)

    staged=WORK/"staged_report.docx";doc.save(staged)
    replacements={"word/document.xml","word/_rels/document.xml.rels","[Content_Types].xml"}
    with zipfile.ZipFile(template) as original,zipfile.ZipFile(staged) as generated,zipfile.ZipFile(FINAL,"w",zipfile.ZIP_DEFLATED) as target:
        generated_names=set(generated.namelist());source_names=set(original.namelist())
        for name in original.namelist():target.writestr(name,generated.read(name) if name in replacements else original.read(name))
        for name in sorted(generated_names-source_names):target.writestr(name,generated.read(name))
    with zipfile.ZipFile(template) as original,zipfile.ZipFile(FINAL) as result:
        untouched=[name for name in original.namelist() if name not in replacements]
        assert all(original.read(name)==result.read(name) for name in untouched)
    (WORK/"package_preservation.json").write_text(json.dumps(dict(preserved_entries=len(untouched),
        changed_entries=sorted(replacements),source_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        report_sha256_before_word=hashlib.sha256(FINAL.read_bytes()).hexdigest()),indent=2),encoding="utf-8")
    # A plain-text copy supports editorial review without changing the Word template.
    text=[]
    for element in doc._element.body:
        if element.tag==qn("w:p"):
            value="".join(element.xpath(".//w:t/text()"))
            if value:text.append(value)
        elif element.tag==qn("w:tbl"):
            from docx.table import Table
            t=Table(element,doc)
            text.extend(" | ".join(c.text for c in row.cells) for row in t.rows)
    (DEST/"报告正文校对稿.txt").write_text("\n\n".join(text),encoding="utf-8")
    print(str(FINAL))


if __name__=="__main__":main()

