"""Unlabeled cabin/boarding coordinates from one shared public feature pool."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tabicl_inputs import TabICLInputBuilder


COLUMNS = ["SequenceExpectedCabin", "CabinSequenceResidual", "CabinResidualRobustZ",
           "PublicCabinOccupancy", "NearestOtherCabinGap", "PublicCabinNeighborCount10"]


def regions(frame):
    return frame.Deck.astype("string") + "/" + frame.Side.astype("string")


class PublicCabinGeometry:
    def fit(self, public):
        d = pd.DataFrame(dict(region=regions(public), x=pd.to_numeric(public.GroupId),
                              cabin=pd.to_numeric(public.CabinNum)))
        self.regions_ = {}
        for key, group in d.dropna().groupby("region", observed=True):
            x, cabin = group.x.to_numpy(dtype=float), group.cabin.to_numpy(dtype=float)
            slope = float(np.dot(x-x.mean(), cabin-cabin.mean())/np.dot(x-x.mean(), x-x.mean())) if np.ptp(x)>0 else 0.
            intercept = float(cabin.mean()-slope*x.mean())
            residual = cabin-(slope*x+intercept)
            center = float(np.median(residual))
            scale = max(float(np.median(np.abs(residual-center))), 1.)
            counts = group.cabin.value_counts().to_dict()
            self.regions_[str(key)] = dict(slope=slope, intercept=intercept, center=center, scale=scale,
                unique=np.sort(np.unique(cabin)), passengers=np.sort(cabin), counts=counts)
        return self

    def transform(self, frame):
        result = pd.DataFrame(np.nan, index=frame.index, columns=COLUMNS)
        keys = regions(frame).fillna("__unknown__").to_numpy(dtype=str)
        group = pd.to_numeric(frame.GroupId).to_numpy(dtype=float)
        cabin = pd.to_numeric(frame.CabinNum).to_numpy(dtype=float)
        for key in np.unique(keys):
            stat = self.regions_.get(key)
            if stat is None:
                continue
            idx = np.flatnonzero(keys==key)
            expected = stat["slope"]*group[idx]+stat["intercept"]
            residual = cabin[idx]-expected
            result.iloc[idx, 0] = expected
            result.iloc[idx, 1] = residual
            result.iloc[idx, 2] = (residual-stat["center"])/stat["scale"]
            idx = idx[np.isfinite(cabin[idx])]
            query = cabin[idx]
            result.iloc[idx, 3] = [stat["counts"].get(v, 0) for v in query]
            before = np.searchsorted(stat["unique"], query, side="left")-1
            after = np.searchsorted(stat["unique"], query, side="right")
            left, right = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
            known = before>=0
            left[known] = query[known]-stat["unique"][before[known]]
            known = after<len(stat["unique"])
            right[known] = stat["unique"][after[known]]-query[known]
            result.iloc[idx, 4] = np.fmin(left, right)
            lo = np.searchsorted(stat["passengers"], query-10, side="left")
            hi = np.searchsorted(stat["passengers"], query+10, side="right")
            result.iloc[idx, 5] = hi-lo
        return result


@dataclass
class SequenceGeometryBuilder:
    enabled: bool = True

    def prepared(self, frame):
        d = frame.copy()
        d["Side"] = d.Side.astype(object).fillna(d.GroupId.map(self.base_.group_side_))
        return d

    def fit(self, frame, public):
        self.base_ = TabICLInputBuilder("compact").fit(frame, public)
        self.geometry_ = PublicCabinGeometry().fit(self.prepared(public)) if self.enabled else None
        return self

    def transform(self, frame):
        X = self.base_.transform(frame)
        if self.geometry_ is not None:
            extra = self.geometry_.transform(self.prepared(frame))
            for c in COLUMNS:
                X[c] = extra[c].to_numpy()
        return X
