"""Cross-fitted cabin-neighborhood statistics, excluding the complete query group."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .competition_features import InputEncoder
from .focused_features import FocusedBuilder


RADII=(10,50,200)
PROFILES=[dict(family=family,spatial=enabled) for family in ["lightgbm","catboost"] for enabled in [False,True]]


def region_keys(frame):
    # Treat unknown keys as unknown rather than borrowing arbitrary cabin labels.
    return (frame.Deck.astype("string")+"/"+frame.Side.astype("string")+"/"+
            frame.CryoSleep.astype("string").fillna("unknown")).fillna("unknown")


@dataclass
class SpatialTargetEncoder:
    smoothing: float=20.
    folds: int=4
    seed: int=20261030

    def fit(self,frame,y):
        y=np.asarray(y,dtype=float)
        if len(y)!=len(frame) or not np.isfinite(y).all() or self.smoothing<=0:
            raise ValueError("Invalid target or smoothing")
        self.prior_=float(y.mean());self.regions_={}
        data=pd.DataFrame(dict(region=region_keys(frame).to_numpy(),cabin=frame.CabinNum.to_numpy(dtype=float),y=y))
        for key,group in data.groupby("region",observed=True):
            known=group.dropna(subset="cabin").sort_values("cabin")
            self.regions_[key]=dict(cabin=known.cabin.to_numpy(),
                total=np.r_[0.,known.y.to_numpy().cumsum()],prior=float((group.y.sum()+self.smoothing*self.prior_)/(len(group)+self.smoothing)))
        return self

    def transform(self,frame):
        result={f"spatial_mean_{r}":np.full(len(frame),self.prior_) for r in RADII}
        result.update({f"spatial_log_count_{r}":np.zeros(len(frame)) for r in RADII})
        keys=region_keys(frame).to_numpy();cabins=frame.CabinNum.to_numpy(dtype=float)
        for key in np.unique(keys):
            stat=self.regions_.get(key)
            if stat is None:continue
            query=np.flatnonzero((keys==key)&np.isfinite(cabins));x=cabins[query]
            for radius in RADII:
                lo=np.searchsorted(stat["cabin"],x-radius,side="left")
                hi=np.searchsorted(stat["cabin"],x+radius,side="right")
                count=hi-lo;total=stat["total"][hi]-stat["total"][lo]
                result[f"spatial_mean_{radius}"][query]=(total+self.smoothing*stat["prior"])/(count+self.smoothing)
                result[f"spatial_log_count_{radius}"][query]=np.log1p(count)
        return pd.DataFrame(result,index=frame.index)

    def fit_transform(self,frame,y):
        y=np.asarray(y,dtype=float);groups=frame.GroupId.to_numpy()
        splits=min(self.folds,len(np.unique(groups)))
        if splits<2:raise ValueError("At least two groups required")
        result=None
        for train,valid in GroupKFold(splits,shuffle=True,random_state=self.seed).split(frame,groups=groups):
            assert not set(groups[train])&set(groups[valid])
            encoder=SpatialTargetEncoder(self.smoothing,self.folds,self.seed).fit(frame.iloc[train],y[train])
            encoded=encoder.transform(frame.iloc[valid])
            if result is None:result=pd.DataFrame(np.nan,index=np.arange(len(frame)),columns=encoded.columns)
            result.iloc[valid]=encoded.to_numpy()
        assert np.isfinite(result.to_numpy()).all()
        self.fit(frame,y);result.index=frame.index
        return result


@dataclass
class SpatialBuilder:
    enabled: bool=True

    def fit_transform(self,frame,y,context):
        self.base_=FocusedBuilder("spend_location",True).fit(frame,context)
        X=self.base_.transform(frame);self.target_encoder_=None
        if self.enabled:
            self.target_encoder_=SpatialTargetEncoder()
            encoded=self.target_encoder_.fit_transform(frame,y)
            for col in encoded:X[col]=encoded[col].to_numpy()
        return X

    def transform(self,frame):
        X=self.base_.transform(frame)
        if self.target_encoder_ is not None:
            encoded=self.target_encoder_.transform(frame)
            for col in encoded:X[col]=encoded[col].to_numpy()
        return X


def fit_member(profile,seed,frame,y,context):
    builder=SpatialBuilder(profile["spatial"])
    X=builder.fit_transform(frame,y,context)
    encoder=InputEncoder(profile["family"]).fit(X)
    if profile["family"]=="catboost":
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=750,depth=5,learning_rate=.04,l2_leaf_reg=10.,
            cat_features=encoder.cat_columns_,max_ctr_complexity=2,random_seed=seed,
            thread_count=4,verbose=False,allow_writing_files=False,loss_function="Logloss")
    else:
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=600,learning_rate=.035,num_leaves=15,
            min_child_samples=35,reg_lambda=5.,reg_alpha=.1,colsample_bytree=.9,
            subsample=.9,subsample_freq=1,random_state=seed,n_jobs=4,verbosity=-1,
            deterministic=True,force_col_wise=True)
    model.fit(encoder.transform(X),y)
    return dict(builder=builder,encoder=encoder,model=model)
