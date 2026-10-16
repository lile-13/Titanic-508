import sys, traceback
import numpy as np, pandas as pd
import preprocess as pp

print("dir:", pp.find_data_dir())
tr, te = pp.load_raw()
print("raw:", tr.shape, te.shape)
print("dup:", pp.duplicate_report(tr))

Ftr = pp.build_features(tr)
Fte = pp.build_features(te)
print("features:", Ftr.shape, Fte.shape)

print("\n-- domain rules --")
_, st = pp.apply_domain_rules(pp.normalize_types(tr))
print(st)

print("\n-- Preprocessor 各策略 --")
for s in pp.STRATEGIES:
    prep = pp.Preprocessor(s).fit(Ftr if s != "drop" else tr)
    out = prep.transform(Ftr if s != "drop" else tr)
    print(f"[{s}] rows={len(out)} 剩余缺失列数={int(out.isna().any().sum())} "
          f"总缺失={int(out.isna().sum().sum())}")

print("\n-- 组内切分 --")
folds = pp.group_folds(Ftr, 5)
print("folds:", [len(v) for _, v in folds])
gid = Ftr.GroupId.values
ok = True
for a, b in folds:
    if set(gid[a]) & set(gid[b]):
        ok = False
print("不同 fold 之间组号无重叠:", ok)
print("各 fold 正例率:", [round(Ftr.Transported.values[v].mean(), 4) for _, v in folds])

print("\n-- 模型矩阵 --")
X, y = pp.make_model_matrix(pp.Preprocessor("rule_group").fit_transform(Ftr), target=True)
Xt = pp.make_model_matrix(pp.Preprocessor("rule_group").fit(Ftr).transform(Fte))
print("X:", X.shape, "y:", y.shape, "Xt:", Xt.shape)
print("列对齐:", list(X.columns) == list(Xt.columns))
print("X 中 NaN:", int(X.isna().sum().sum()), "| inf:", int(np.isinf(X.select_dtypes('number')).sum().sum()))

print("\n-- sklearn 打通 --")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
a, b = folds[0]
clf = LogisticRegression(max_iter=2000).fit(X.iloc[a].fillna(0), y[a])
print("LR acc:", round(clf.score(X.iloc[b].fillna(0), y[b]), 4))
h = HistGradientBoostingClassifier(random_state=0).fit(X.iloc[a], y[a])
print("HistGB acc:", round(h.score(X.iloc[b], y[b]), 4))
