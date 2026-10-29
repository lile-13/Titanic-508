"""Learn shared relations on outer training rows, with G upweighted; route only G."""
import numpy as np

from .competition_features import InputEncoder
from .g_deck_models import GDeckBuilder,g_mask,PROFILES

PARAMETERS=dict(lightgbm=dict(n_estimators=500,learning_rate=.035,num_leaves=15,
    min_child_samples=35,reg_lambda=12.,reg_alpha=.1,colsample_bytree=.9,
    subsample=.9,subsample_freq=1),catboost=dict(iterations=650,depth=5,
    learning_rate=.035,l2_leaf_reg=12.,max_ctr_complexity=1),g_training_weight=2.)


def fit_member(profile,seed,frame,y,context):
    builder=GDeckBuilder(profile["roles"]).fit(frame,context)
    mask=g_mask(frame);y=np.asarray(y)
    if mask.sum()<50 or len(np.unique(y[mask]))<2:raise ValueError("Insufficient G-deck training data")
    X=builder.transform(frame)
    encoder=InputEncoder(profile["family"]).fit(X)
    if profile["family"]=="catboost":
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(**PARAMETERS["catboost"],cat_features=encoder.cat_columns_,
            random_seed=seed,thread_count=4,loss_function="Logloss",verbose=False,allow_writing_files=False)
    else:
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(**PARAMETERS["lightgbm"],random_state=seed,n_jobs=4,
            verbosity=-1,deterministic=True,force_col_wise=True)
    weight=np.where(mask,PARAMETERS["g_training_weight"],1.)
    model.fit(encoder.transform(X),y,sample_weight=weight)
    return dict(builder=builder,encoder=encoder,model=model)
