"""G-only specialists with public, leave-self-out co-traveler role features."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .competition_features import InputEncoder
from .focused_features import FocusedBuilder

PROFILES=[dict(family=family,roles=roles) for family in ["lightgbm","catboost"] for roles in [False,True]]
PARAMETERS=dict(lightgbm=dict(n_estimators=450,learning_rate=.035,num_leaves=7,
    min_child_samples=25,reg_lambda=12.,reg_alpha=.1,colsample_bytree=.9,
    subsample=.9,subsample_freq=1),catboost=dict(iterations=600,depth=4,
    learning_rate=.035,l2_leaf_reg=12.,max_ctr_complexity=1))
ROLE_COLUMNS=["OtherKnownCryoShare","OtherKnownCryoCount","OtherAdultMeanAge",
    "OtherAdultCount","OtherChildCount","AgeMinusOtherAdultMean","GroupAgeRole"]


def g_mask(frame):
    return frame.Deck.astype("string").eq("G").fillna(False).to_numpy(dtype=bool)


@dataclass
class GDeckBuilder:
    roles: bool=True

    def fit(self,frame,context=None):
        public=frame if context is None else context
        if not public.PassengerId.is_unique:raise ValueError("Public passenger IDs must be unique")
        self.base_=FocusedBuilder("location",True).fit(frame,public)
        # Never retain Transported or use outcomes in the context tables.
        own=pd.DataFrame(index=public.PassengerId.to_numpy())
        own["known"]=public.CryoSleep.notna().to_numpy(int)
        own["true"]=public.CryoSleep.eq(True).fillna(False).to_numpy(int)
        own["adult"]=public.Age.ge(18).to_numpy(int)
        own["adult_age"]=public.Age.where(public.Age.ge(18),0.).fillna(0.).to_numpy(float)
        own["child"]=public.Age.le(12).to_numpy(int)
        self.own_=own
        self.group_=own.groupby(public.GroupId.to_numpy(),observed=True).sum()
        return self

    def role_features(self,frame):
        group=self.group_.reindex(frame.GroupId.to_numpy()).fillna(0.).reset_index(drop=True)
        own=self.own_.reindex(frame.PassengerId.to_numpy()).fillna(0.).reset_index(drop=True)
        other=group-own
        if (other[["known","adult","child"]].to_numpy()<-1e-9).any():
            raise ValueError("Passenger/group context mismatch")
        share=(other.true/other.known.replace(0,np.nan)).to_numpy(float)
        age=(other.adult_age/other.adult.replace(0,np.nan)).to_numpy(float)
        out=pd.DataFrame(dict(OtherKnownCryoShare=share,OtherKnownCryoCount=other.known.to_numpy(float),
            OtherAdultMeanAge=age,OtherAdultCount=other.adult.to_numpy(float),
            OtherChildCount=other.child.to_numpy(float)),index=frame.index)
        out["AgeMinusOtherAdultMean"]=frame.Age.to_numpy(float)-age
        out["GroupAgeRole"]=np.select([
            frame.Age.le(12).to_numpy(bool)&(other.adult.to_numpy()>0),
            frame.Age.ge(18).to_numpy(bool)&(other.child.to_numpy()>0),
            frame.Age.isna().to_numpy(bool)],
            ["child_with_adult","adult_with_children","unknown_age"],default="other")
        out["GroupAgeRole"]=out.GroupAgeRole.astype(object)
        return out

    def transform(self,frame):
        X=self.base_.transform(frame)
        X["CabinRelative"]=frame.CabinRelative.to_numpy(float)
        if self.roles:
            roles=self.role_features(frame)
            for col in roles:X[col]=roles[col].to_numpy()
        return X


def fit_member(profile,seed,frame,y,context):
    builder=GDeckBuilder(profile["roles"]).fit(frame,context)
    mask=g_mask(frame);y=np.asarray(y)
    if mask.sum()<50 or len(np.unique(y[mask]))<2:raise ValueError("Insufficient G-deck training data")
    X=builder.transform(frame.iloc[np.flatnonzero(mask)])
    encoder=InputEncoder(profile["family"]).fit(X)
    if profile["family"]=="catboost":
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(**PARAMETERS["catboost"],cat_features=encoder.cat_columns_,
            random_seed=seed,thread_count=4,loss_function="Logloss",verbose=False,allow_writing_files=False)
    else:
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(**PARAMETERS["lightgbm"],random_state=seed,n_jobs=4,
            verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(encoder.transform(X),y[mask])
    return dict(builder=builder,encoder=encoder,model=model)


def predict_specialist(member,frame):
    mask=g_mask(frame);p=np.full(len(frame),np.nan)
    if mask.any():
        part=frame.iloc[np.flatnonzero(mask)]
        X=member["encoder"].transform(member["builder"].transform(part))
        p[mask]=member["model"].predict_proba(X)[:,1]
    return p


def blend_g(reference,specialist,frame,weight):
    reference=np.asarray(reference,dtype=float);specialist=np.asarray(specialist,dtype=float)
    if reference.shape!=(len(frame),) or specialist.shape!=reference.shape or not 0<=weight<=1:
        raise ValueError("Invalid G-only blend shape or weight")
    mask=g_mask(frame)
    if not np.isfinite(reference).all() or not np.isfinite(specialist[mask]).all():
        raise ValueError("Non-finite inference probability")
    if ((reference<0)|(reference>1)).any() or ((specialist[mask]<0)|(specialist[mask]>1)).any():
        raise ValueError("Probability outside [0,1]")
    result=reference.copy()
    result[mask]=(1-weight)*reference[mask]+weight*specialist[mask]
    return result


@dataclass
class GOnlyFoldModel:
    reference_members: list
    specialist_members: list
    specialist_weight: float

    def predict_proba(self,frame):
        if not np.isclose(sum(m["weight"] for m in self.reference_members),1.):
            raise ValueError("Reference fold weights must sum to one")
        reference=np.zeros(len(frame))
        for m in self.reference_members:
            X=m["encoder"].transform(m["builder"].transform(frame))
            reference+=m["weight"]*m["model"].predict_proba(X)[:,1]
        if not self.specialist_members:raise ValueError("Missing G specialist")
        specialist=np.mean([predict_specialist(m,frame) for m in self.specialist_members],axis=0)
        p=blend_g(reference,specialist,frame,self.specialist_weight)
        return np.column_stack([1-p,p])
