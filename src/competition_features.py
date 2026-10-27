"""比赛优化特征；仅统计公开的无标签特征，不改变成员2/3的基线接口。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import preprocess as pp

FEATURE_SETS = ("base", "geo", "spend", "rich", "rich_noid", "rich_names")
PROFILES = (
    ("base", "rule_group"), ("geo", "rule_group"), ("rich", "rule_group"),
    ("rich_noid", "rule_group"), ("base", "native"), ("spend", "native"),
    ("rich", "native"), ("rich_noid", "native"), ("rich_names", "native"),
)
GROUP_EXTRAS = ["GroupMeanAge", "GroupMinAge", "GroupMaxAge", "GroupAgeSpan",
                "GroupHasChild", "GroupTotalSpend", "GroupMeanSpend", "GroupZeroShare"]


def build_pool(train, test):
    """同一公开特征池内的计数/同行统计；Transported不参与任何统计。"""
    tr, te = pp.build_features_combined(train, test)
    both = pd.concat([tr, te], ignore_index=True)
    both["GroupIdNum"] = pd.to_numeric(both.GroupId)
    cabin_known = both[["Deck", "CabinNum", "Side"]].notna().all(axis=1)
    key = (both.Deck.astype("string") + "/" + both.CabinNum.astype("string") +
           "/" + both.Side.astype("string")).where(cabin_known)
    both["CabinKey"] = key
    both["CabinOccupancy"] = key.map(key.value_counts()).fillna(1).astype(float)
    deck_max = both.groupby("Deck", observed=True).CabinNum.transform("max")
    both["CabinRelative"] = both.CabinNum / deck_max.replace(0, np.nan)
    for c in pp.MISSING_INDICATOR_COLS:
        both[f"raw_missing_{c}"] = both[c].isna().astype(float)
    both["RawMissingCount"] = both[pp.MISSING_INDICATOR_COLS].isna().sum(axis=1)
    both["SpendMissingCount"] = both[pp.SPEND_COLS].isna().sum(axis=1)
    both["ConfirmedZeroSpend"] = ((both.SpendMissingCount == 0) &
                                   (both[pp.SPEND_COLS].sum(axis=1) == 0)).astype(float)
    both["AnyKnownPositiveSpend"] = (both[pp.SPEND_COLS].fillna(0).sum(axis=1) > 0).astype(float)
    group = both.groupby("GroupId", observed=True)
    both["GroupMeanAge"] = group.Age.transform("mean")
    both["GroupMinAge"] = group.Age.transform("min")
    both["GroupMaxAge"] = group.Age.transform("max")
    both["GroupAgeSpan"] = both.GroupMaxAge - both.GroupMinAge
    both["GroupHasChild"] = (both.GroupMinAge < 13).astype(float)
    raw_total = both[pp.SPEND_COLS].sum(axis=1, min_count=1)
    both["GroupTotalSpend"] = raw_total.groupby(both.GroupId).transform("sum")
    both["GroupMeanSpend"] = raw_total.groupby(both.GroupId).transform("mean")
    both["GroupZeroShare"] = both.ConfirmedZeroSpend.groupby(both.GroupId).transform("mean")
    n = len(tr)
    tr = both.iloc[:n].copy().reset_index(drop=True)
    te = both.iloc[n:].drop(columns=[pp.TARGET], errors="ignore").copy().reset_index(drop=True)
    tr[pp.TARGET] = train[pp.TARGET].to_numpy(dtype=bool)
    return tr, te


@dataclass
class FeatureBuilder:
    feature_set: str = "base"
    strategy: str = "rule_group"

    def fit(self, frame):
        if self.feature_set not in FEATURE_SETS or self.strategy not in ("rule_group", "simple", "indicator", "native"):
            raise ValueError("未知优化特征/预处理组合")
        self.preprocessor_ = None if self.strategy == "native" else pp.Preprocessor(self.strategy).fit(frame)
        return self

    def transform(self, frame):
        if self.preprocessor_ is not None:
            d = self.preprocessor_.transform(frame)
        else:
            d, _ = pp.apply_domain_rules(frame.copy())
            # 仅对已观测到唯一组值的分类属性补缺；保留其他未知类别。
            for c in ["HomePlanet", "Side"]:
                known = d.groupby("GroupId", observed=True)[c]
                values = known.first().where(known.nunique() == 1)
                d[c] = d[c].astype(object).fillna(d.GroupId.map(values))
            # 公开数据中已核对：已知年龄<=12的儿童没有正消费；仅补缺，不改已知值。
            child = d.Age.le(12)
            d.loc[child, pp.SPEND_COLS] = d.loc[child, pp.SPEND_COLS].fillna(0.0)
            d = pp.add_spend_features(d)
            d = pp.add_age_features(d)
            d = pp.add_cabin_zone_features(d)

        numeric = list(pp.BOOL_FEATURES) + list(pp.NUM_FEATURES)
        categorical = list(pp.CAT_FEATURES)
        if self.strategy == "native":
            numeric = [c for c in numeric if c not in ["CryoSleep", "VIP"]]
            categorical += ["CryoSleep", "VIP"]
        if self.strategy == "indicator":
            numeric += [f"{c}_na" for c in pp.MISSING_INDICATOR_COLS if f"{c}_na" in d]
        geometry = self.feature_set in ("geo", "rich", "rich_noid", "rich_names")
        spending = self.feature_set in ("spend", "rich", "rich_noid", "rich_names")
        if geometry:
            numeric += ["CabinNum", "CabinRelative", "CabinOccupancy", "GroupNum", "IsFirstInGroup", "IsLastInGroup"]
            if self.feature_set != "rich_noid":
                numeric += ["GroupIdNum"]
            for width in (50, 100):
                c = f"CabinZone{width}"
                d[c] = np.floor(d.CabinNum / width)
                categorical.append(c)
            categorical.append("CabinKey")
        if spending:
            numeric += ["RawMissingCount", "SpendMissingCount", "ConfirmedZeroSpend", "AnyKnownPositiveSpend"]
            numeric += [f"raw_missing_{c}" for c in pp.MISSING_INDICATOR_COLS]
            for c in pp.SPEND_COLS:
                d[f"log_{c}"] = np.log1p(d[c])
                d[f"share_{c}"] = d[c] / d.TotalSpend.replace(0, np.nan)
                d[f"zero_{c}"] = (d[c] == 0).astype(float)
                numeric += [f"log_{c}", f"share_{c}", f"zero_{c}"]
        if self.feature_set in ("rich", "rich_noid", "rich_names"):
            numeric += GROUP_EXTRAS
            categorical += ["Surname"]
        if self.feature_set == "rich_names":
            categorical.append("FirstName")
        X = pd.DataFrame(index=d.index)
        for c in numeric:
            X[c] = pd.to_numeric(d[c], errors="raise").astype(float)
        for c in categorical:
            X[c] = d[c].astype(object).where(d[c].notna(), "__missing__").map(str).astype(object)
        return X.replace([np.inf, -np.inf], np.nan)


@dataclass
class InputEncoder:
    family: str = "catboost"

    def fit(self, X):
        self.columns_ = list(X)
        self.cat_columns_ = [c for c in X if X[c].dtype == object]
        self.categories_ = {c: sorted(X[c].unique().tolist()) for c in self.cat_columns_}
        return self

    def transform(self, X):
        X = X.loc[:, self.columns_].copy()
        if self.family == "lightgbm":
            for c in self.cat_columns_:
                X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        elif self.family != "catboost":
            raise ValueError("未知模型家族")
        return X


def development_splits(fold_of):
    """本轮第5折不参与开发期训练或评分；公开无标签特征池仍保持一致。"""
    fold_of = np.asarray(fold_of)
    return [(np.flatnonzero((fold_of != fold) & (fold_of != 5)), np.flatnonzero(fold_of == fold))
            for fold in range(1, 5)]
