"""使用折内教师产生的高置信软标签；测试集真实标签永不读取。"""
import numpy as np
import pandas as pd

from .competition_features import FeatureBuilder,InputEncoder
from .optimization import estimator


def augment_training(frame,y,test,probability,confidence,pseudo_weight=.25):
    y=np.asarray(y,dtype=float);p=np.asarray(probability,dtype=float)
    if len(y)!=len(frame) or len(p)!=len(test) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("训练标签或教师概率形状、取值不合法")
    if not .5<confidence<1 or not 0<pseudo_weight<=1:raise ValueError("置信度或权重不合法")
    if set(frame.GroupId)&set(test.GroupId):raise ValueError("真实训练组与待预测组重叠")
    mask=(p>=confidence)|(p<=1-confidence)
    train=frame.drop(columns="Transported",errors="ignore")
    extra=test.loc[mask].drop(columns="Transported",errors="ignore")
    augmented=pd.concat([train,extra],ignore_index=True)
    targets=np.r_[y,p[mask]]
    weights=np.r_[np.ones(len(y)),np.full(mask.sum(),pseudo_weight)]
    audit=dict(real_rows=len(y),pseudo_rows=int(mask.sum()),pseudo_positive_rows=int((p[mask]>=.5).sum()),
        confidence=confidence,pseudo_weight=pseudo_weight,
        pseudo_effective_weight=float(mask.sum()*pseudo_weight),target_in_features=False,
        real_test_group_overlap=0,targets="true training labels followed by teacher probabilities, not hard labels")
    return augmented,targets,weights,audit


def fit_student(config,seed,frame,y,test,probability):
    augmented,target,weights,audit=augment_training(frame,y,test,probability,config["confidence"],config["pseudo_weight"])
    builder=FeatureBuilder(config["feature_set"],"native").fit(augmented)
    X=builder.transform(augmented);encoder=InputEncoder("catboost").fit(X)
    learner=estimator(dict(config["teacher_config"],seed=seed),encoder.cat_columns_,threads=4)
    learner.set_params(loss_function="CrossEntropy",eval_metric="CrossEntropy")
    learner.fit(encoder.transform(X),target,sample_weight=weights)
    return dict(builder=builder,encoder=encoder,model=learner,pseudo_audit=audit)
