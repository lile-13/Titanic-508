"""特征分组消融实验（成员3 交付物）。

固定模型 + 固定折划分（features/folds.csv），逐组加减特征，
量化每组特征的贡献，输出均值±标准差。

用法（在仓库根目录）：
    python -m scripts.feature_ablation --model histgb --seeds 42,2026
    python -m scripts.feature_ablation --model logreg

输出：reports/experiments/feature_group_ablation.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocess as pp
from src.paths import EXPERIMENTS_DIR, FEATURES_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# 特征分组：基于 make_model_matrix 输出的列名，各组互斥且恰好覆盖全部列。
GROUP_SPECS = {
    "base": {  # 原始字段，不含任何派生特征
        "exact": ["CryoSleep", "VIP", "Age",
                  "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"],
        "prefix": ["HomePlanet_", "Destination_", "Deck_", "Side_"],
    },
    "cabin": {  # 舱位派生（舱号分箱）
        "exact": [],
        "prefix": ["CabinZone_"],
    },
    "group": {  # 同行组 + 姓氏相关
        "exact": ["IsAlone", "GroupSize", "GroupPos", "SurnameSize",
                  "GroupSurnameCount", "IsMixedGroup"],
        "prefix": [],
    },
    "spend": {  # 消费派生
        "exact": ["ZeroSpend", "TotalSpend", "LogTotalSpend", "LuxurySpend",
                  "BasicSpend", "NZerosAmenities", "NAmenitiesUsed", "LuxuryShare"],
        "prefix": ["MaxAmenity_"],
    },
    "age": {  # 年龄派生
        "exact": ["IsChild"],
        "prefix": ["AgeBand_"],
    },
}

GROUP_ORDER = ["base", "cabin", "group", "spend", "age"]
GROUP_ALIAS = {
    "base": "基础",
    "cabin": "舱位",
    "group": "群组",
    "spend": "消费",
    "age": "年龄",
}


def columns_of(spec: dict, all_cols: list[str]) -> list[str]:
    """按 exact 名称 + prefix 前缀收集列，保持 all_cols 原始顺序。"""
    out = []
    for c in all_cols:
        if c in spec["exact"] or any(c.startswith(p) for p in spec["prefix"]):
            out.append(c)
    return out


def build_configs(all_cols: list[str]) -> dict[str, list[str]]:
    cols = {g: columns_of(GROUP_SPECS[g], all_cols) for g in GROUP_ORDER}
    full: list[str] = []
    for g in GROUP_ORDER:
        full += cols[g]
    assert set(full) == set(all_cols), (
        f"分组未覆盖全部列。缺失: {sorted(set(all_cols) - set(full))}，"
        f"多余: {sorted(set(full) - set(all_cols))}"
    )
    assert len(full) == len(set(full)), "分组之间有重叠列"

    configs: dict[str, list[str]] = {}
    acc: list[str] = []
    acc_name: list[str] = []
    for i, g in enumerate(GROUP_ORDER):
        acc = acc + cols[g]
        acc_name.append(GROUP_ALIAS[g])
        if i == len(GROUP_ORDER) - 1:
            configs["全量"] = acc
        else:
            configs["+".join(acc_name)] = acc
    for g in GROUP_ORDER:
        configs[f"全量-{GROUP_ALIAS[g]}"] = [
            c for c in full if c not in set(cols[g])]
    return configs


def make_model(model: str, seed: int):
    """固定模型与超参数，只随 seed 变化。"""
    if model == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=200, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    raise ValueError(f"未知模型: {model!r}")


def evaluate(features: pd.DataFrame, fold_map: pd.DataFrame,
             model: str, seeds: list[int], strategy: str = "rule_group"):
    fold_of = features.PassengerId.map(fold_map.set_index("PassengerId")["fold"])
    assert fold_of.notna().all() and fold_of.isin(range(1, 6)).all(), "fold 映射不完整"

    all_cols = None
    configs = None
    accs = None

    # 每折独立拟合填补器；同一折复用矩阵做不同列子集。
    for fold in range(1, 6):
        tr_idx = np.flatnonzero((fold_of != fold).to_numpy())
        va_idx = np.flatnonzero((fold_of == fold).to_numpy())
        Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(
            features, tr_idx, va_idx, strategy=strategy)
        if all_cols is None:
            all_cols = list(Xtr.columns)
            configs = build_configs(all_cols)
            accs = {name: [] for name in configs}
        for name, cols in configs.items():
            Xtr_s, Xva_s = Xtr[cols], Xva[cols]
            for seed in seeds:
                clf = make_model(model, seed)
                clf.fit(Xtr_s, ytr)
                accs[name].append(float((clf.predict(Xva_s) == yva).mean()))

    rows = []
    for name in configs:
        v = np.asarray(accs[name])
        rows.append({
            "模型": model,
            "特征集": name,
            "特征数": len(configs[name]),
            "准确率均值": round(float(v.mean()), 4),
            "准确率标准差": round(float(v.std(ddof=1)), 4),
            "运行次数": int(v.size),
        })
    return pd.DataFrame(rows), configs, all_cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="histgb", choices=["histgb", "logreg"])
    ap.add_argument("--seeds", default="42,2026", help="逗号分隔的随机种子")
    ap.add_argument("--strategy", default="rule_group",
                    choices=["simple", "indicator", "rule_group"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    train_raw, test_raw = pp.load_raw()
    features, _ = pp.build_features_combined(train_raw, test_raw)
    fold_map = pd.read_csv(FEATURES_DIR / "folds.csv",
                           dtype={"PassengerId": str, "GroupId": str})

    df, configs, all_cols = evaluate(
        features, fold_map, args.model, seeds, strategy=args.strategy)

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out = EXPERIMENTS_DIR / "feature_group_ablation.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"建模矩阵共 {len(all_cols)} 列，分组规模：")
    for g in GROUP_ORDER:
        print(f"  {g:6s} {len(columns_of(GROUP_SPECS[g], all_cols)):2d} 列")
    print()
    print(df.to_string(index=False))
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
