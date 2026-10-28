"""局部甲板专家与消费方向约束的对照，不读取外层验证标签。"""
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost

from .focused_features import FocusedBuilder
from .competition_features import InputEncoder


@dataclass
class ConstrainedBuilder(FocusedBuilder):
    def transform(self, frame):
        # 保留原始消费，不加入随消费变化的汇总列，方便解释指定列约束。
        return super().transform(frame).drop(columns="TotalSpend",errors="ignore")


class DeckExperts:
    def __init__(self,family="lightgbm",expert_weight=.5,variant=0,seed=42,min_region_size=100):
        self.family=family;self.expert_weight=expert_weight;self.variant=variant
        self.seed=seed;self.min_region_size=min_region_size

    def get_params(self,deep=True):
        return dict(family=self.family,expert_weight=self.expert_weight,variant=self.variant,
                    seed=self.seed,min_region_size=self.min_region_size,regions=["G","E"],
                    iterations=[450,650][self.variant],learning_rate=.035,regularization=12.)

    def _learner(self, X):
        if self.family=="catboost":
            return CatBoostClassifier(iterations=[450,650][self.variant],depth=[4,5][self.variant],
                learning_rate=.035,l2_leaf_reg=12.,random_seed=self.seed,thread_count=4,
                cat_features=[c for c in X if X[c].dtype==object],verbose=False,
                allow_writing_files=False,loss_function="Logloss")
        return LGBMClassifier(n_estimators=[450,650][self.variant],learning_rate=.035,
            num_leaves=[7,15][self.variant],min_child_samples=25,reg_lambda=12.,reg_alpha=.1,
            colsample_bytree=.9,subsample=.9,subsample_freq=1,random_state=self.seed,n_jobs=4,
            verbosity=-1,deterministic=True,force_col_wise=True)

    def fit(self,X,y):
        y=np.asarray(y)
        self.global_=self._learner(X).fit(X,y);self.experts_={}
        for region in ["G","E"]:
            mask=X.Deck.astype("string").eq(region).fillna(False).to_numpy(dtype=bool)
            if mask.sum()>=self.min_region_size and len(np.unique(y[mask]))==2:
                self.experts_[region]=self._learner(X).fit(X.loc[mask],y[mask])
        self.classes_=np.array([0,1])
        return self

    def predict_proba(self,X):
        probability=self.global_.predict_proba(X)[:,1].copy()
        for region,model in self.experts_.items():
            mask=X.Deck.astype("string").eq(region).fillna(False).to_numpy(dtype=bool)
            if mask.any():
                probability[mask]=(1-self.expert_weight)*probability[mask]+self.expert_weight*model.predict_proba(X.loc[mask])[:,1]
        return np.column_stack([1-probability,probability])


PROFILES=[dict(family="monotone",mode="compact",variant=variant,positive=positive,resolve_planet=True)
          for variant in range(4) for positive in [False,True]]
PROFILES += [dict(family="expert_lgb",mode="location",variant=variant,expert_weight=weight,resolve_planet=True)
             for variant in [0,1] for weight in [.5,1.]]
PROFILES += [dict(family="expert_cat",mode="location",variant=variant,expert_weight=.75,resolve_planet=True)
             for variant in [0,1]]


def estimator(profile,seed,categorical):
    if profile["family"]=="monotone":
        variant=profile["variant"]
        constraints=dict(RoomService=-1,Spa=-1,VRDeck=-1)
        if profile["positive"]:constraints.update(FoodCourt=1,ShoppingMall=1)
        return xgboost.XGBClassifier(n_estimators=[700,900,650,850][variant],learning_rate=.035,
            max_depth=[3,3,5,5][variant],min_child_weight=[3,12,3,12][variant],reg_lambda=8.,
            reg_alpha=.1,subsample=.85,colsample_bytree=.9,max_bin=512,tree_method="hist",
            monotone_constraints=constraints,enable_categorical=True,random_state=seed,n_jobs=4,eval_metric="logloss")
    return DeckExperts("catboost" if profile["family"]=="expert_cat" else "lightgbm",
                       profile["expert_weight"],profile["variant"],seed)


def fit_member(profile,seed,frame,y,context):
    cls=ConstrainedBuilder if profile["family"]=="monotone" else FocusedBuilder
    builder=cls(profile["mode"],profile["resolve_planet"]).fit(frame,context)
    X=builder.transform(frame)
    encoder=InputEncoder("catboost" if profile["family"]=="expert_cat" else "lightgbm").fit(X)
    model=estimator(profile,seed,encoder.cat_columns_).fit(encoder.transform(X),y)
    return dict(builder=builder,encoder=encoder,model=model)
