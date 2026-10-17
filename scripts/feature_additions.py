"""新特征逐个加入的配对检验（成员3 交付物）。

基线 = 旧的 53 列完整特征（不含成员3 新增列）。
对每个新增特征单元，在同一折下做配对比较，用 5 折配对 t 检验（df=4）
判断增益是否稳定。HistGB / LogReg 在本项目的超参数下是确定性模型，
因此不引入随机种子维度的伪重复；方差只来自统一折划分。

用法（仓库根目录）：
    python -m scripts.feature_additions --model logreg
    python -m scripts.feature_additions --model histgb

输出：reports/experiments/feature_additions.csv
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

from scipy import stats

from src import preprocess as pp
from src.paths import EXPERIMENTS_DIR, FEATURES_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# 成员3 新增的特征单元；彼此互斥，并与旧 53 列互斥。
NEW_UNITS = {
    "CabinNum": lambda cols: [c for c in cols if c == "CabinNum"],
    "CabinZone": lambda cols: [c for c in cols if c.startswith("CabinZone_")],
    "GroupMeanAge": lambda cols: [c for c in cols if c == "GroupMeanAge"],
    "GroupAgeSpan": lambda cols: [c for c in cols if c == "GroupAgeSpan"],
    "GroupHasChild": lambda cols: [c for c in cols if c == "GroupHasChild"],
    "GroupTotalSpend": lambda cols: [c for c in cols if c == "GroupTotalSpend"],
    "GroupHasSpender": lambda cols: [c for c in cols if c == "GroupHasSpender"],
}


def make_model(model: str, seed: int):
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


def run_config(Xtr, ytr, Xva, yva, cols, model, seed):
    clf = make_model(model, seed)
    clf.fit(Xtr[cols], ytr)
    return float((clf.predict(Xva[cols]) == yva).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg", choices=["histgb", "logreg"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_raw, test_raw = pp.load_raw()
    features, _ = pp.build_features_combined(train_raw, test_raw)
    fold_map = pd.read_csv(FEATURES_DIR / "folds.csv",
                           dtype={"PassengerId": str, "GroupId": str})
    fold_of = features.PassengerId.map(fold_map.set_index("PassengerId")["fold"])
    assert fold_of.notna().all() and fold_of.isin(range(1, 6)).all()

    # 第一折确定列名
    tr_idx = np.flatnonzero((fold_of != 1).to_numpy())
    va_idx = np.flatnonzero((fold_of == 1).to_numpy())
    Xtr0, _, _, _ = pp.prepare_fold_matrices(features, tr_idx, va_idx,
                                             strategy="rule_group")
    all_cols = list(Xtr0.columns)

    new_cols = sorted({c for unit in NEW_UNITS.values()
                       for c in unit(all_cols)})
    old_full = [c for c in all_cols if c not in set(new_cols)]
    units = {name: fn(all_cols) for name, fn in NEW_UNITS.items()}
    assert set(old_full) | set(new_cols) == set(all_cols), "新旧列未覆盖全部"
    assert not (set(old_full) & set(new_cols)), "新旧列有重叠"

    # 收集每个配置在每折的 accuracy（5 折，顺序一致用于配对）
    configs = {"基线(旧53列)": old_full,
               **{f"+{name}": old_full + c for name, c in units.items()},
               "全量(旧+全部新)": old_full + new_cols}
    results = {name: [] for name in configs}
    for fold in range(1, 6):
        tr_idx = np.flatnonzero((fold_of != fold).to_numpy())
        va_idx = np.flatnonzero((fold_of == fold).to_numpy())
        Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(
            features, tr_idx, va_idx, strategy="rule_group")
        for name, cols in configs.items():
            results[name].append(
                run_config(Xtr, ytr, Xva, yva, cols, args.model, args.seed))

    base = np.asarray(results["基线(旧53列)"])
    rows = []
    for name in configs:
        if name == "基线(旧53列)":
            rows.append({"模型": args.model, "配置": name,
                         "准确率均值": round(float(base.mean()), 4),
                         "准确率标准差": round(float(base.std(ddof=1)), 4),
                         "相对基线": 0.0, "正折数": int(base.size),
                         "配对p值": None, "折数": int(base.size)})
            continue
        arr = np.asarray(results[name])
        delta = arr - base
        t_stat, p_val = stats.ttest_rel(arr, base)
        rows.append({"模型": args.model, "配置": name,
                     "准确率均值": round(float(arr.mean()), 4),
                     "准确率标准差": round(float(arr.std(ddof=1)), 4),
                     "相对基线": round(float(delta.mean()), 4),
                     "正折数": int((delta > 0).sum()),
                     "配对p值": round(float(p_val), 4),
                     "折数": int(arr.size)})

    df = pd.DataFrame(rows)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out = EXPERIMENTS_DIR / "feature_additions.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"新特征单元：{list(units)}")
    print(f"旧基线 {len(old_full)} 列 + 新 {len(new_cols)} 列 = 全量 {len(all_cols)} 列\n")
    print(df.to_string(index=False))
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
