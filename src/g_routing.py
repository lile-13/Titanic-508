"""Fixed, disjoint G cohort routing; all untouched passengers retain the reference."""
from dataclasses import dataclass

import numpy as np

from .g_deck_models import g_mask,predict_specialist


def cohort_masks(frame):
    g=g_mask(frame)
    cold=g&frame.CryoSleep.eq(True).fillna(False).to_numpy(bool)
    child=g&frame.CryoSleep.eq(False).fillna(False).to_numpy(bool)&frame.Age.le(12).fillna(False).to_numpy(bool)
    assert not (cold&child).any()
    return cold,child


def route_probability(frame,reference,cold_probability,child_probability,cold_weight=.25,child_weight=.25):
    reference=np.asarray(reference,dtype=float)
    cold_probability=np.asarray(cold_probability,dtype=float);child_probability=np.asarray(child_probability,dtype=float)
    if reference.shape!=(len(frame),) or cold_probability.shape!=reference.shape or child_probability.shape!=reference.shape:
        raise ValueError("Invalid routing probability shape")
    if not 0<=cold_weight<=1 or not 0<=child_weight<=1:raise ValueError("Invalid routing weight")
    cold,child=cohort_masks(frame)
    for p in [reference,cold_probability[cold],child_probability[child]]:
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError("Invalid routed probability")
    result=reference.copy()
    result[cold]=(1-cold_weight)*reference[cold]+cold_weight*cold_probability[cold]
    result[child]=(1-child_weight)*reference[child]+child_weight*child_probability[child]
    return result


@dataclass
class GRoutedFoldModel:
    reference_members: list
    cold_calibrator: object
    child_members: list
    cold_weight: float=.25
    child_weight: float=.25

    def predict_proba(self,frame):
        if not np.isclose(sum(m["weight"] for m in self.reference_members),1.):raise ValueError("Invalid reference weights")
        reference=np.zeros(len(frame))
        for m in self.reference_members:
            X=m["encoder"].transform(m["builder"].transform(frame))
            reference+=m["weight"]*m["model"].predict_proba(X)[:,1]
        cold,child=cohort_masks(frame)
        pc=np.full(len(frame),np.nan);pk=pc.copy()
        if cold.any():pc[cold]=self.cold_calibrator.predict(frame.iloc[np.flatnonzero(cold)],reference[cold])
        if child.any():
            if not self.child_members:raise ValueError("Missing child specialist")
            part=frame.iloc[np.flatnonzero(child)]
            pk[child]=np.mean([predict_specialist(m,part) for m in self.child_members],axis=0)
        p=route_probability(frame,reference,pc,pk,self.cold_weight,self.child_weight)
        return np.column_stack([1-p,p])
