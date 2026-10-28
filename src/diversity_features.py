"""折内拟合的距离模型编码；保存后可用同一接口复现集成预测。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from . import preprocess as pp


@dataclass
class DiversityEncoder:
    """先对消费做对数变换，再按训练折中位数和尺度编码。"""

    def numeric_frame(self, X):
        X = X.copy()
        for c in [*pp.SPEND_COLS, "TotalSpend", "LuxurySpend", "BasicSpend"]:
            if c in X:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def fit(self, X):
        X = self.numeric_frame(X)
        self.columns_ = list(X)
        cats = [c for c in X if X[c].dtype == object]
        nums = [c for c in X if c not in cats]
        self.transformer_ = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                       RobustScaler()), nums),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ], sparse_threshold=0)
        self.transformer_.fit(X)
        return self

    def transform(self, X):
        result = self.transformer_.transform(self.numeric_frame(X.loc[:, self.columns_]))
        return np.clip(np.asarray(result, dtype=float), -8, 8)
