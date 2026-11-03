"""Create the final model diagram and platform-score comparison from records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    out = ROOT / "reports/figures"
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set(xlim=(0, 12), ylim=(0, 5.2)); ax.axis("off")
    def box(x, y, width, height, text, color):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.08", fc=color, ec="#526278", lw=1.2))
        ax.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=12, linespacing=1.65)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color="#526278", lw=1.4))
    box(.2, 1.75, 2.25, 1.55, "原始数据与统一特征\n完整同行组五折\n折内拟合填补与编码", "#EDF3F8")
    box(3.05, 2.8, 4.05, 1.85, "45个树成员  合计25%\nCatBoost spend 6.25%\nCatBoost rich_names 6.25%\nLightGBM base 12.50%", "#E8F1FA")
    box(3.05, .45, 4.05, 1.65, "5个TabICLv2训练上下文  合计75%\n每个上下文8视图  算术概率平均\n固定预训练权重  不更新网络参数", "#E7F3EE")
    box(8.0, 1.75, 3.55, 1.55, "按固定权重加权概率\n阈值0.5输出True或False\n4277行  保持PassengerId顺序", "#F2F1F8")
    arrow((2.45, 2.7), (3.0, 3.65)); arrow((2.45, 2.3), (3.0, 1.3))
    arrow((7.2, 3.65), (7.95, 2.8)); arrow((7.2, 1.3), (7.95, 2.25))
    ax.text(6, 5.0, "最终模型的训练与预测结构", ha="center", fontsize=17, weight="bold")
    ax.text(6, .04, "树成员在每折、每种子下拟合；TabICL使用作者预训练模型进行上下文学习。", ha="center", fontsize=10, color="#4E5A67")
    fig.savefig(out / "final_model_architecture_20261103.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    ledger = json.loads((ROOT / "reports/experiments/kaggle_submissions.json").read_text(encoding="utf-8"))
    rows = ledger["submissions"]
    names = ["原CatBoost", "原树集成", "全量树混合", "局部LGB混合", "TabICL50%", "TabPFN25%", "TabPFN50%", "纯TabPFN", "TabICL75%", "TabICL87.5%", "全上下文混合", "三模型互补", "纯TabICL"]
    scores = [r["score"]*100 for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    colors = ["#27816A" if r["score"] == max(x["score"] for x in rows) else "#6084AD" for r in rows]
    ax.barh(names, scores, color=colors, height=.64)
    for i, value in enumerate(scores): ax.text(value+.012, i, f"{value:.3f}%", va="center", fontsize=10)
    ax.axvline(82, color="#9B5462", ls="--", lw=1.2, label="探索目标 82%")
    ax.set(xlim=(80, 82.15), xlabel="Kaggle公开评分  Accuracy（%）", title="13次已完成提交的实际平台成绩")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(loc="lower right", frameon=False)
    fig.text(.02, .015, "分数来自截图确认的提交台账；横轴从80%起，不表示独立盲测或统计显著性。", fontsize=9, color="#4E5A67")
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(out / "kaggle_scores_20261103.png", dpi=200, facecolor="white"); plt.close(fig)
    print("Generated two figures from the fixed model and recorded platform outcomes.")


if __name__ == "__main__":
    main()
