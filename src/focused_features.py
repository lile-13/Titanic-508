"""精简舱位与消费特征；不改历史预处理或冻结模型接口。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .competition_features import FeatureBuilder
from . import preprocess as pp


def unique_value_map(frame, key, value):
    grouped=frame.groupby(key,observed=True)[value]
    return grouped.first().where(grouped.nunique()==1).dropna().to_dict()


@dataclass
class FocusedBuilder:
    mode: str="location"
    resolve_planet: bool=True

    def fit(self, frame, context=None):
        if self.mode not in ("compact","location","spend_location","physical"):
            raise ValueError("未知精简特征方案")
        public=frame if context is None else context
        self.surname_planets_=unique_value_map(public,"Surname","HomePlanet")
        self.deck_planets_=unique_value_map(public,"Deck","HomePlanet")
        self.base_=FeatureBuilder("spend" if self.mode=="spend_location" else "base","native")
        self.base_.fit(frame)
        return self

    def transform(self, frame):
        d=frame.copy()
        if self.resolve_planet:
            # 仅填未知HomePlanet。只映射公开特征中全部已知值一致的姓氏/甲板。
            planet=d.HomePlanet.astype(object)
            planet=planet.fillna(d.Surname.astype(object).map(self.surname_planets_))
            d["HomePlanet"]=planet.fillna(d.Deck.astype(object).map(self.deck_planets_))
        X=self.base_.transform(d)
        if self.mode in ("compact","physical"):
            X=X[[*pp.SPEND_COLS,"Age","TotalSpend","GroupSize","SurnameSize",
                 "HomePlanet","Destination","Deck","Side","CryoSleep","VIP"]].copy()
        X["CabinNum"]=pd.to_numeric(d.CabinNum,errors="coerce").astype(float)
        X["GroupIdNum"]=pd.to_numeric(d.GroupId,errors="raise").astype(float)
        X["DeckSide"]=(d.Deck.astype("string")+"_"+d.Side.astype("string")).fillna("__missing__").astype(object)
        if self.mode=="physical":
            clean,_=pp.apply_domain_rules(d)
            personal=clean[["RoomService","Spa","VRDeck"]].sum(axis=1,min_count=1)
            social=clean[["FoodCourt","ShoppingMall"]].sum(axis=1,min_count=1)
            X["PersonalSpend"]=personal.astype(float)
            X["SocialSpend"]=social.astype(float)
            X["PersonalSocialLogDifference"]=(np.log1p(personal)-np.log1p(social)).astype(float)
            X["AmenityPattern"]=clean[pp.SPEND_COLS].map(lambda v:"?" if pd.isna(v) else ("1" if v>0 else "0")).agg("".join,axis=1).astype(object)
        return X.replace([np.inf,-np.inf],np.nan)
