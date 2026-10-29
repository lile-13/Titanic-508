"""EBM with group-disjoint internal validation bags and frozen feature profiles."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .competition_features import InputEncoder
from .focused_features import FocusedBuilder


PROFILES=[dict(mode="compact",interactions=0),dict(mode="compact",interactions=12),
          dict(mode="physical",interactions=12),dict(mode="location",interactions=18)]
PARAMETERS=dict(max_bins=128,max_interaction_bins=32,outer_bags=6,inner_bags=0,
    learning_rate=.035,max_rounds=4000,early_stopping_rounds=100,
    early_stopping_tolerance=1e-5,min_samples_leaf=10,reg_lambda=2.,
    smoothing_rounds=200,interaction_smoothing_rounds=100,n_jobs=4)


def group_bags(groups,seed,outer_bags=6):
    groups=np.asarray(groups)
    if groups.ndim!=1 or len(np.unique(groups))<2:
        raise ValueError("Internal validation requires at least two groups")
    bags=np.ones((len(groups),outer_bags),dtype=np.int8)
    splitter=GroupShuffleSplit(outer_bags,test_size=.15,random_state=seed)
    for bag,(train,valid) in enumerate(splitter.split(groups,groups=groups)):
        assert not set(groups[train])&set(groups[valid])
        bags[valid,bag]=-1
    return bags


def fit_member(profile,seed,frame,y,context):
    # Optional import: historical model inference does not require InterpretML.
    from interpret.glassbox import ExplainableBoostingClassifier
    builder=FocusedBuilder(profile["mode"],True).fit(frame,context)
    X=builder.transform(frame)
    encoder=InputEncoder("catboost").fit(X)
    types=["nominal" if c in encoder.cat_columns_ else "continuous" for c in X]
    bags=group_bags(frame.GroupId,seed,PARAMETERS["outer_bags"])
    model=ExplainableBoostingClassifier(**PARAMETERS,interactions=profile["interactions"],
        feature_types=types,random_state=seed).fit(encoder.transform(X),y,bags=bags)
    model.group_validation_=dict(bags=PARAMETERS["outer_bags"],validation_group_fraction=.15,
        seed=seed,group_overlap=0,validation_rows_per_bag=(bags<0).sum(axis=0).tolist(),
        training_groups=len(np.unique(frame.GroupId)),outer_training_rows_only=True)
    return dict(builder=builder,encoder=encoder,model=model)
