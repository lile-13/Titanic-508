"""小型神经网络对照；停止轮数仅由训练部分内部的分组验证确定。"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from .diversity_features import DiversityEncoder
from .focused_features import FocusedBuilder


PROFILES=[dict(family="mlp",mode="compact",variant=variant,resolve_planet=True)
          for variant in range(4)]
PROFILES += [dict(family="mlp",mode=mode,variant=4,resolve_planet=True)
             for mode in ["spend_location","physical"]]


def estimator(profile,seed,categorical):
    v=profile["variant"]
    return MLPClassifier(hidden_layer_sizes=(64,32),activation=["relu","relu","tanh","tanh","relu"][v],
        alpha=[.1,1.,.1,1.,.5][v],batch_size=128,learning_rate_init=.001,
        solver="adam",max_iter=300,early_stopping=False,tol=0.,n_iter_no_change=301,
        random_state=seed)


def fit_group_mlp(model,X,y,groups,max_epochs=300,patience=30):
    from sklearn.base import clone
    X=np.asarray(X);y=np.asarray(y);groups=np.asarray(groups)
    train,valid=next(GroupShuffleSplit(1,test_size=.18,random_state=model.random_state).split(X,y,groups))
    assert not set(groups[train])&set(groups[valid])
    probe=clone(model);best_loss=float("inf");best_epoch=1;stale=0
    with threadpool_limits(limits=1):
        for epoch in range(1,max_epochs+1):
            probe.partial_fit(X[train],y[train],classes=[0,1])
            loss=float(log_loss(y[valid],probe.predict_proba(X[valid])[:,1],labels=[0,1]))
            if loss<best_loss-1e-5:
                best_loss=loss;best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=patience:break
        final=clone(model).set_params(max_iter=best_epoch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",ConvergenceWarning)
            final.fit(X,y)
    final.epoch_selection_=dict(selected_epoch=best_epoch,max_epochs=max_epochs,patience=patience,
         inner_best_loss=best_loss,inner_train_rows=len(train),inner_valid_rows=len(valid),
         inner_group_overlap=0,validation_uses_outer_training_rows_only=True)
    return final


def fit_member(profile,seed,frame,y,context):
    builder=FocusedBuilder(profile["mode"],profile["resolve_planet"]).fit(frame,context)
    X=builder.transform(frame);encoder=DiversityEncoder().fit(X)
    model=fit_group_mlp(estimator(profile,seed,[]),encoder.transform(X),y,frame.GroupId)
    return dict(builder=builder,encoder=encoder,model=model)
