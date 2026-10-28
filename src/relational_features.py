"""同行组交叉拟合的目标统计；验证/测试标签不参与拟合。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .competition_features import FeatureBuilder


def relation_keys(frame, spatial=False):
    keys = pd.DataFrame(index=frame.index)
    keys["Surname"] = frame.Surname.astype("string")
    if spatial:
        band = np.floor(frame.CabinNum / 100).astype("Int64").astype("string")
        keys["DeckBandSide"] = frame.Deck.astype("string") + "/" + band + "/" + frame.Side.astype("string")
    return keys


@dataclass
class GroupCrossFitEncoder:
    smoothing: float = 10.0
    spatial: bool = False
    folds: int = 4
    seed: int = 20261028

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        if len(y) != len(frame) or not np.isfinite(y).all() or self.smoothing <= 0:
            raise ValueError("标签长度、数值或平滑参数不合法")
        self.prior_ = float(y.mean())
        self.statistics_ = {}
        for col, key in relation_keys(frame, self.spatial).items():
            data = pd.DataFrame({"key": key.to_numpy(), "y": y})
            grouped = data.groupby("key", dropna=True).y.agg(["sum", "count"])
            self.statistics_[col] = grouped
        return self

    def transform(self, frame):
        out = pd.DataFrame(index=frame.index)
        for col, key in relation_keys(frame, self.spatial).items():
            stats = self.statistics_[col]
            count = key.map(stats["count"]).fillna(0).to_numpy(dtype=float)
            total = key.map(stats["sum"]).fillna(0).to_numpy(dtype=float)
            out[f"te_{col}"] = (total + self.smoothing*self.prior_) / (count+self.smoothing)
            out[f"te_count_{col}"] = np.log1p(count)
        return out

    def fit_transform(self, frame, y):
        y = np.asarray(y, dtype=float)
        groups = frame.GroupId.to_numpy()
        n = min(self.folds, len(np.unique(groups)))
        if n < 2:
            raise ValueError("交叉拟合至少需要两个同行组")
        out = pd.DataFrame(index=np.arange(len(frame)))
        splits = GroupKFold(n_splits=n, shuffle=True, random_state=self.seed)
        for train, valid in splits.split(frame, groups=groups):
            encoder = GroupCrossFitEncoder(self.smoothing, self.spatial, self.folds, self.seed)
            encoded = encoder.fit(frame.iloc[train], y[train]).transform(frame.iloc[valid])
            for col in encoded:
                out.loc[valid, col] = encoded[col].to_numpy()
        if out.isna().any().any():
            raise RuntimeError("交叉拟合编码存在漏行")
        self.fit(frame, y)
        out.index = frame.index
        return out


@dataclass
class RelationalBuilder:
    feature_set: str = "base"
    strategy: str = "rule_group"
    smoothing: float | None = 10.0
    spatial: bool = False

    def fit_transform(self, frame, y):
        self.base_ = FeatureBuilder(self.feature_set,self.strategy).fit(frame)
        X = self.base_.transform(frame)
        self.target_encoder_ = None
        if self.smoothing is not None:
            self.target_encoder_ = GroupCrossFitEncoder(self.smoothing,self.spatial)
            encoded = self.target_encoder_.fit_transform(frame,y)
            for col in encoded:X[col] = encoded[col].to_numpy()
        return X

    def transform(self, frame):
        X = self.base_.transform(frame)
        if self.target_encoder_ is not None:
            encoded = self.target_encoder_.transform(frame)
            for col in encoded:X[col] = encoded[col].to_numpy()
        return X
