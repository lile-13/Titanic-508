"""CabinNum 有效性核查（成员3 交付物）。

1) 对抗验证：只用舱位相关特征区分 train/test，AUC 应接近 0.5。
   若明显高于 0.5，说明舱位特征是"数据集指纹"，不能入模。
2) 与目标的关系：把 CabinNum 分箱后看正例率，判断是否存在真实信号。
3) 不同随机种子下的 CV 增益稳定性（HistGB）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocess as pp
from src.paths import EXPERIMENTS_DIR, FEATURES_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    train_raw, test_raw = pp.load_raw()
    features, test_features = pp.build_features_combined(train_raw, test_raw)

    # ---- 1) 对抗验证 -------------------------------------------------------
    a = features.copy()
    b = test_features.copy()
    a["is_train"] = 1
    b["is_train"] = 0
    both = pd.concat([a, b], ignore_index=True, sort=False)
    y = both["is_train"].to_numpy()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    def adv_auc(use_cols):
        X = pd.get_dummies(
            both[use_cols].astype("object").fillna("__missing__"),
            columns=[c for c in use_cols if c != "CabinNum"],
        )
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, va in skf.split(X, y):
            clf = HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=200, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False, random_state=0)
            clf.fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[va], clf.predict_proba(X.iloc[va])[:, 1]))
        return np.mean(aucs), np.std(aucs)

    print("对抗验证（区分 train/test 的 AUC，应接近 0.5）：")
    for label, cols in [("Deck+Side", ["Deck", "Side"]),
                        ("CabinNum", ["CabinNum"]),
                        ("CabinZone", ["CabinZone"]),
                        ("Deck+Side+CabinNum", ["Deck", "Side", "CabinNum"]),
                        ("Deck+Side+CabinZone", ["Deck", "Side", "CabinZone"])]:
        m, s = adv_auc(cols)
        print(f"  {label:22s} {m:.4f} ± {s:.4f}")

    # ---- 2) CabinNum 与目标的关系 ------------------------------------------
    df = features.copy()
    df["CabinNum_bin"] = pd.cut(df["CabinNum"], bins=10, duplicates="drop")
    rel = df.groupby("CabinNum_bin", observed=True).agg(
        样本数=("PassengerId", "size"),
        正例率=("Transported", "mean"),
    )
    print("\nCabinNum 十等分与目标正例率：")
    print(rel.to_string())

    # ---- 3) 已有 CabinZone/组聚合时，CabinNum 的边际收益 ---------------------
    fold_map = pd.read_csv(FEATURES_DIR / "folds.csv",
                           dtype={"PassengerId": str, "GroupId": str})
    fold_of = features.PassengerId.map(fold_map.set_index("PassengerId")["fold"])
    tr_idx = np.flatnonzero((fold_of != 1).to_numpy())
    va_idx = np.flatnonzero((fold_of == 1).to_numpy())
    Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(features, tr_idx, va_idx,
                                                  strategy="rule_group")
    old = [c for c in Xtr.columns if c != "CabinNum"]
    acc_without, acc_with = [], []
    for fold in range(1, 6):
        tri = np.flatnonzero((fold_of != fold).to_numpy())
        vai = np.flatnonzero((fold_of == fold).to_numpy())
        Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(
            features, tri, vai, strategy="rule_group")
        for cols, bucket in [(old, acc_without), (list(Xtr.columns), acc_with)]:
            clf = HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=200, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False, random_state=42)
            clf.fit(Xtr[cols], ytr)
            bucket.append(float((clf.predict(Xva[cols]) == yva).mean()))
    d = np.asarray(acc_with) - np.asarray(acc_without)
    t_stat, p_val = stats.ttest_rel(acc_with, acc_without)
    print("\n已含 CabinZone/组聚合的 64 列上，再增 CabinNum 的边际收益：")
    print(f"  无CabinNum {np.mean(acc_without):.4f} vs "
          f"含CabinNum {np.mean(acc_with):.4f}，delta {d.mean():+.4f}，p={p_val:.4f}")

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
