"""只从训练部分的留出预测学习概率校准，支持原集成预测接口。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


def calibration_frame(features,probability,mode):
    probability=np.asarray(probability,dtype=float)
    if len(probability)!=len(features) or not np.isfinite(probability).all():
        raise ValueError("预测概率与乘客数据不对齐或含非有限数值")
    p=np.clip(probability,1e-6,1-1e-6)
    output=pd.DataFrame({"logit":np.log(p/(1-p))},index=features.index)
    if mode=="global":return output
    if mode not in ("context","slope_context"):raise ValueError("未知校准方式")
    for col in ["HomePlanet","Deck","Side","CryoSleep"]:
        output[col]=features[col].astype(object).where(features[col].notna(),"__missing__").map(str).astype(object)
    output["log_group_size"]=np.log1p(features.GroupSize.to_numpy(dtype=float))
    if mode=="slope_context":
        output["DeckSide"]=(output.Deck+"_"+output.Side).astype(object)
        for planet in ["Earth","Europa","Mars"]:
            output["logit_"+planet]=output.logit*(output.HomePlanet==planet)
        output["logit_sleep"]=output.logit*(output.CryoSleep=="True")
    return output


@dataclass
class ProbabilityCalibrator:
    mode: str="global"
    regularization: float=1.

    def fit(self,features,probability,y):
        X=calibration_frame(features,probability,self.mode)
        categorical=[c for c in X if X[c].dtype==object]
        numerical=[c for c in X if c not in categorical]
        transform=ColumnTransformer([
            ("numeric",StandardScaler(),numerical),
            ("category",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical)],sparse_threshold=0)
        self.pipeline_=make_pipeline(transform,LogisticRegression(C=self.regularization,max_iter=1000,random_state=42))
        self.pipeline_.fit(X,np.asarray(y,dtype=int))
        return self

    def predict(self,features,probability):
        return self.pipeline_.predict_proba(calibration_frame(features,probability,self.mode))[:,1]


class IdentityTransform:
    def transform(self,frame):return frame


@dataclass
class CalibratedFoldModel:
    members: list
    calibrator: ProbabilityCalibrator
    calibration_weight: float=1.

    def predict_proba(self,features):
        if not np.isclose(sum(m["weight"] for m in self.members),1.):
            raise ValueError("折内基模型权重必须归一化")
        raw=np.zeros(len(features))
        for member in self.members:
            X=member["encoder"].transform(member["builder"].transform(features))
            raw+=member["weight"]*member["model"].predict_proba(X)[:,1]
        calibrated=self.calibrator.predict(features,raw)
        p=(1-self.calibration_weight)*raw+self.calibration_weight*calibrated
        return np.column_stack([1-p,p])
