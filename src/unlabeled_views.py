"""Two target-free views of existing names and public ship coordinates."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .competition_features import FeatureBuilder


def region_key(frame):
    return (frame.Deck.astype("string") + "/" + frame.Side.astype("string")).to_numpy()


@dataclass
class UnlabeledViewBuilder:
    view: str

    def fit(self, frame, context):
        if self.view not in ("names", "coordinates"):
            raise ValueError("Unknown unlabeled view")
        self.base_ = FeatureBuilder("rich_names", "native").fit(frame)
        # An explicit feature allowlist prevents accidental target access.
        public = context[["FirstName", "GroupId", "GroupIdNum", "Deck", "Side", "CabinNum"]].copy()
        first = public.FirstName.astype("string").str.lower()
        self.first_count_ = first.value_counts().to_dict()
        self.first_group_count_ = public.groupby(first, observed=True).GroupId.nunique().to_dict()
        self.max_group_ = float(public.GroupIdNum.max())
        self.regions_ = {}
        public["region"] = region_key(public)
        for key, part in public.dropna(subset=["region", "CabinNum"]).groupby("region", observed=True):
            part = part.sort_values("CabinNum")
            self.regions_[key] = dict(cabins=part.CabinNum.to_numpy(dtype=float),
                                     groups=part.GroupId.astype(str).to_numpy())
        return self

    def transform(self, frame):
        X = self.base_.transform(frame)
        if self.view == "names":
            first = frame.FirstName.astype("string").str.lower()
            surname = frame.Surname.astype("string").str.lower()
            for label, name in [("First", first), ("Surname", surname)]:
                for width in [2, 3]:
                    X[f"{label}Prefix{width}"] = name.str[:width].fillna("__missing__").astype(object)
                    X[f"{label}Suffix{width}"] = name.str[-width:].fillna("__missing__").astype(object)
                length = name.str.len().astype(float)
                X[f"{label}Length"] = length
                X[f"{label}VowelFraction"] = name.str.count("[aeiou]").astype(float) / length.replace(0, np.nan)
            X["FirstNamePublicLogCount"] = np.log1p(first.map(self.first_count_).fillna(0).astype(float))
            X["FirstNamePublicLogGroupCount"] = np.log1p(first.map(self.first_group_count_).fillna(0).astype(float))
        else:
            progress = frame.GroupIdNum.to_numpy(dtype=float) / self.max_group_
            X["CabinMinusBoardingProgress"] = frame.CabinRelative.to_numpy(dtype=float) - progress
            rank = np.full(len(frame), np.nan)
            neighbors = np.full(len(frame), np.nan)
            keys = region_key(frame)
            cabins = frame.CabinNum.to_numpy(dtype=float)
            groups = frame.GroupId.astype(str).to_numpy()
            for i, (key, cabin, own_group) in enumerate(zip(keys, cabins, groups)):
                if pd.isna(key) or not np.isfinite(cabin):
                    continue
                region = self.regions_.get(key)
                if region is None:
                    continue
                values = region["cabins"]
                left = np.searchsorted(values, cabin, side="left")
                right = np.searchsorted(values, cabin, side="right")
                rank[i] = (left + right) / (2 * len(values))
                lo = np.searchsorted(values, cabin - 10, side="left")
                hi = np.searchsorted(values, cabin + 10, side="right")
                neighbors[i] = len(set(region["groups"][lo:hi]) - {own_group})
            X["CabinRankWithinDeckSide"] = rank
            X["CabinRankMinusBoardingProgress"] = rank - progress
            X["OtherGroupsWithin10CabinsLogCount"] = np.log1p(neighbors)
        return X.replace([np.inf, -np.inf], np.nan)
