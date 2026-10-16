"""导出清洗后的特征表与折划分，供建模成员直接读取（成员2 交付物之一）。

产出（写入 features/）：
    train_clean.csv   清洗后的训练集特征 + Transported
    test_clean.csv    清洗后的测试集特征（无标签）
    folds.csv         PassengerId -> fold 的映射（按组切分的 5 折）
    feature_dict.md   每个派生特征的含义

用法：
    python export_features.py            # 用默认配置（rule_group + 合并构建特征）
    python export_features.py --strategy simple
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

import preprocess as pp

HERE = Path(__file__).resolve().parent
OUT = HERE / "features"

DERIVED_DOC = {
    "GroupId": "同行组号（PassengerId 下划线前的部分）",
    "GroupNum": "组内序号（1~8）",
    "GroupSize": "同组人数：单人为 1。与目标强相关（单人 45% vs 四人组 64%）",
    "IsAlone": "是否独自出行（GroupSize == 1）",
    "IsFirstInGroup": "是否为组内第一个（GroupNum == 1）",
    "IsLastInGroup": "是否为组内最后一个",
    "GroupPos": "组内相对位置 0~1，比绝对序号稳定",
    "Deck": "Cabin 拆出的舱位牌面 A~G、T（T 仅 5 人）",
    "CabinNum": "Cabin 拆出的舱号（0~1894）",
    "Side": "Cabin 拆出的舷侧：P=Port、S=Starboard",
    "Surname": "姓氏（同姓 ≠ 同组，两者互补）",
    "FirstName": "名",
    "SurnameSize": "同姓氏人数。**必须合并 train+test 计算**，否则是数据集指纹（见 notebook 04）",
    "SurnameCount": "同姓氏计数（同 SurnameSize，含缺失处理差异）",
    "GroupSurnameCount": "本组内出现的不同姓氏数，>1 表示拼组",
    "IsMixedGroup": "是否拼组（GroupSurnameCount > 1）",
    "TotalSpend": "五项消费合计",
    "LogTotalSpend": "log1p(TotalSpend)。与目标相关性 -0.47，远强于原始值",
    "ZeroSpend": "五项消费是否为 0",
    "LuxurySpend": "Spa + VRDeck + ShoppingMall",
    "BasicSpend": "RoomService + FoodCourt",
    "NZerosAmenities": "五项中为 0 的项数",
    "NAmenitiesUsed": "五项中消费>0 的项数",
    "LuxuryShare": "豪华消费占比；总消费为 0 时记 0（用 ZeroSpend 区分）",
    "MaxAmenity": "消费最多的项目；无消费时为 __None__",
    "IsChild": "是否 13 岁以下",
    "AgeBand": "年龄段分箱",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="rule_group", choices=list(pp.STRATEGIES))
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    train_raw, test_raw = pp.load_raw()

    # 关键：合并算派生特征，避免 SurnameSize 变成"来自哪个数据集"的指纹
    train, test = pp.build_features_combined(train_raw, test_raw)
    print(f"特征构建完成（合并计算）: train {train.shape}, test {test.shape}")

    prep = pp.Preprocessor(args.strategy).fit(train)
    train_clean = prep.transform(train)
    test_clean = prep.transform(test)

    # 建模必需列做一次硬校验
    X, y = pp.make_model_matrix(train_clean, target=True)
    X_test = pp.make_model_matrix(test_clean)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    assert X.isna().sum().sum() == 0, "训练矩阵仍有缺失"
    assert X_test.isna().sum().sum() == 0, "测试矩阵仍有缺失"
    assert list(X.columns) == list(X_test.columns), "两侧列不一致"
    print(f"建模矩阵: X {X.shape}, X_test {X_test.shape}, 特征数一致 ✓, 无缺失 ✓")

    train_clean.drop(columns=["Cabin", "Name"]).to_csv(
        OUT / "train_clean.csv", index=False, encoding="utf-8-sig")
    test_clean.drop(columns=["Cabin", "Name"]).to_csv(
        OUT / "test_clean.csv", index=False, encoding="utf-8-sig")

    # 折划分：按组切分，直接发给建模成员，避免各自切法不一致
    folds = pp.group_folds(train_clean, n_splits=5, seed=42)
    fold_of = pd.Series(-1, index=train_clean.index, dtype=int)
    for i, (_, va_idx) in enumerate(folds):
        fold_of.iloc[va_idx] = i
    fold_df = pd.DataFrame({
        "PassengerId": train_clean["PassengerId"],
        "fold": fold_of.values + 1,
        "GroupId": train_clean["GroupId"],
    })
    fold_df.to_csv(OUT / "folds.csv", index=False, encoding="utf-8-sig")
    sizes = fold_df.fold.value_counts().sort_index()
    print(f"折划分: {dict(sizes)} | 正例率 {[round(train_clean.Transported.values[va].mean(), 4) for _, va in folds]}")

    # 交叉检查：同一个组不能出现在两个 fold（按组切分的定义）
    chk = fold_df.groupby("GroupId").fold.nunique()
    assert (chk == 1).all(), "存在跨 fold 的组，切分有误"
    print("按组切分校验通过：没有任何组跨 fold ✓")

    with open(OUT / "feature_dict.md", "w", encoding="utf-8") as fh:
        fh.write("# 派生特征说明\n\n")
        fh.write("由 `export_features.py` 生成，与 `notebooks/*.ipynb` 中的口径一致。\n\n")
        fh.write("## 清洗后表里的派生列\n\n| 列名 | 含义 |\n|---|---|\n")
        for k, v in DERIVED_DOC.items():
            fh.write(f"| `{k}` | {v} |\n")
        fh.write("\n## 建模矩阵（`preprocess.make_model_matrix`）\n\n")
        fh.write(f"- 数值/布尔列（{len(pp.NUM_FEATURES) + len(pp.BOOL_FEATURES)} 个）原样进入\n")
        fh.write(f"- 分类列（{', '.join(pp.CAT_FEATURES)}）做 one-hot\n")
        fh.write(f"- 最终特征维度：**{X.shape[1]}**\n")
        fh.write("\n## 折划分（`folds.csv`）\n\n")
        fh.write("按 `GroupId` 切的 5 折（`StratifiedGroupKFold`，seed=42）。\n")
        fh.write("**不要自己重新随机切分**——同组乘客跨 fold 会造成泄漏，本地分数虚高。\n")

    print(f"\n已写入 {OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:22s} {f.stat().st_size/1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
