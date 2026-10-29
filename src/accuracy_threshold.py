"""Learn a shrunk accuracy cutoff from group-held-out training predictions only."""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit,logit


@dataclass
class AccuracyThreshold:
    shrinkage: float=.5
    resamples: int=150
    seed: int=20261031

    def fit(self,features,probability,y):
        p=np.asarray(probability,dtype=float);y=np.asarray(y,dtype=int)
        if p.ndim!=1 or len(p)!=len(features) or len(y)!=len(p) or len(p)==0:
            raise ValueError("Misaligned calibration rows")
        if not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or np.any((p<0)|(p>1)):
            raise ValueError("Invalid probability or binary target")
        if not 0<=self.shrinkage<=1 or self.resamples<1:
            raise ValueError("Invalid shrinkage or resampling count")
        grid=np.round(np.linspace(.35,.65,61),3)
        grid=np.array(sorted(grid,key=lambda t:(abs(t-.5),t)))
        groups,inverse=np.unique(features.GroupId.to_numpy(),return_inverse=True)
        rng=np.random.default_rng(self.seed)
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=self.resamples)
        correct=((p[:,None]>=grid[None,:])==y[:,None]).astype(float)
        score=weights[:,inverse]@correct
        chosen=grid[np.argmax(score,axis=1)]
        self.raw_threshold_=float(np.median(chosen))
        self.effective_threshold_=float(expit(self.shrinkage*logit(self.raw_threshold_)))
        self.audit_=dict(group_resamples=self.resamples,groups=len(groups),seed=self.seed,
            raw_threshold=self.raw_threshold_,effective_threshold=self.effective_threshold_,
            shrinkage=self.shrinkage,grid_min=.35,grid_max=.65,grid_step=.005,
            source="Group-held-out predictions from outer training only; no outer validation labels")
        return self

    def predict(self,features,probability):
        p=np.asarray(probability,dtype=float)
        if len(p)!=len(features) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError("Invalid inference probability")
        return expit(logit(np.clip(p,1e-6,1-1e-6))-logit(self.effective_threshold_))
