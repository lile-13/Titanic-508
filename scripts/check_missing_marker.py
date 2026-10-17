"""缺失标记特征的独立复现（成员3 补充交付）。

成员2 已给出结论：S3（简单填充+缺失指示位）与 S2（简单填充）几乎相同。
这里用成员3 的固定模型与折划分独立复现一次，确认"缺失标记不作为重点"的结论
不依赖成员2 的口径，并把结果落到 reports/experiments/missing_marker_recheck.csv。

用法（仓库根目录）：
    python -m scripts.check_missing_marker
"""
from __future__ import annotations

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


def make_model(model: str, seed: int = 42):
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
    raise ValueError(model)


def main() -> int:
    train_raw, test_raw = pp.load_raw()
    features, _ = pp.build_features_combined(train_raw, test_raw)
    fold_map = pd.read_csv(FEATURES_DIR / "folds.csv",
                           dtype={"PassengerId": str, "GroupId": str})
    fold_of = features.PassengerId.map(fold_map.set_index("PassengerId")["fold"])

    rows = []
    for model in ["histgb", "logreg"]:
        for strategy in ["simple", "indicator"]:
            accs = []
            for fold in range(1, 6):
                tri = np.flatnonzero((fold_of != fold).to_numpy())
                vai = np.flatnonzero((fold_of == fold).to_numpy())
                Xtr, ytr, Xva, yva = pp.prepare_fold_matrices(
                    features, tri, vai, strategy=strategy)
                clf = make_model(model)
                clf.fit(Xtr, ytr)
                accs.append(float((clf.predict(Xva) == yva).mean()))
            rows.append({
                "模型": model,
                "策略": "S2 简单填充" if strategy == "simple" else "S3 简单填充+指示位",
                "准确率均值": round(float(np.mean(accs)), 4),
                "准确率标准差": round(float(np.std(accs, ddof=1)), 4),
            })

    df = pd.DataFrame(rows)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out = EXPERIMENTS_DIR / "missing_marker_recheck.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
