"""Penalized logistic offsets on group-held-out teacher probabilities, G only."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit

from .g_deck_models import GDeckBuilder,g_mask

PROFILES=[dict(roles=roles,ridge=ridge) for ridge in [.01,.1] for roles in [False,True]]


def offset_objective(beta,X,offset,y,ridge):
    z=offset+X@beta
    loss=float(np.mean(np.logaddexp(0,z)-y*z)+.5*ridge*np.dot(beta,beta))
    gradient=X.T@(expit(z)-y)/len(y)+ridge*beta
    return loss,gradient


@dataclass
class GOffsetCalibrator:
    roles: bool=True
    ridge: float=.01

    def design(self,frame):
        true=frame.CryoSleep.eq(True).fillna(False).to_numpy(float)
        false=frame.CryoSleep.eq(False).fillna(False).to_numpy(float)
        side=frame.Side.astype("string").eq("S").fillna(False).to_numpy(float)
        base=np.column_stack([np.ones(len(frame)),true,false,side])
        if not self.roles:return base
        r=self.builder_.role_features(frame)
        # Fixed scales/defaults, no validation-fitted normalization or imputation.
        numeric=np.column_stack([
            r.OtherKnownCryoShare.fillna(.5).to_numpy()-.5,
            (r.OtherAdultMeanAge.fillna(35).to_numpy()-35)/20,
            r.OtherChildCount.to_numpy()/2,
            r.AgeMinusOtherAdultMean.fillna(0).to_numpy()/20,
            r.OtherKnownCryoCount.to_numpy()/8,r.OtherAdultCount.to_numpy()/5,
            r.OtherKnownCryoShare.isna().to_numpy(float),r.OtherAdultMeanAge.isna().to_numpy(float)])
        return np.column_stack([base,numeric,numeric*true[:,None],numeric*false[:,None]])

    def fit(self,frame,probability,y,context):
        p=np.asarray(probability,dtype=float);y=np.asarray(y,dtype=float)
        if p.shape!=(len(frame),) or y.shape!=p.shape or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
            raise ValueError("Invalid held-out teacher rows")
        if self.ridge<=0 or np.any((p<0)|(p>1)):raise ValueError("Invalid ridge or teacher probability")
        self.builder_=GDeckBuilder(self.roles).fit(frame,context)
        mask=g_mask(frame)
        if mask.sum()<20:raise ValueError("Insufficient held-out G rows")
        X=self.design(frame.iloc[np.flatnonzero(mask)])
        offset=logit(np.clip(p[mask],1e-6,1-1e-6))
        result=minimize(offset_objective,np.zeros(X.shape[1]),args=(X,offset,y[mask],self.ridge),
            jac=True,method="L-BFGS-B",options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
        if not result.success or not np.isfinite(result.x).all():raise RuntimeError("G offset optimization failed")
        self.coef_=result.x;self.audit_=dict(g_calibration_rows=int(mask.sum()),
            calibration_groups=int(frame.GroupId.iloc[np.flatnonzero(mask)].nunique()),ridge=self.ridge,
            role_features=self.roles,coefficients=self.coef_.tolist(),converged=bool(result.success),
            method="Ridge logistic correction with teacher logit coefficient fixed at 1; labels from training holdout only")
        return self

    def predict(self,frame,probability):
        p=np.asarray(probability,dtype=float).copy()
        if p.shape!=(len(frame),) or not np.isfinite(p).all():raise ValueError("Invalid inference rows")
        mask=g_mask(frame)
        if mask.any():p[mask]=expit(logit(np.clip(p[mask],1e-6,1-1e-6))+self.design(frame.iloc[np.flatnonzero(mask)])@self.coef_)
        return p


def corrected_probability(calibrator,frame,reference,weight):
    if not 0<=weight<=1:raise ValueError("Invalid correction weight")
    result=np.asarray(reference,dtype=float).copy();mask=g_mask(frame)
    adjusted=calibrator.predict(frame,reference)
    result[mask]=(1-weight)*result[mask]+weight*adjusted[mask]
    return result


@dataclass
class GOffsetFoldModel:
    reference_members: list
    calibrator: GOffsetCalibrator
    correction_weight: float

    def predict_proba(self,frame):
        if not np.isclose(sum(m["weight"] for m in self.reference_members),1.):raise ValueError("Invalid reference weights")
        p=np.zeros(len(frame))
        for m in self.reference_members:
            X=m["encoder"].transform(m["builder"].transform(frame))
            p+=m["weight"]*m["model"].predict_proba(X)[:,1]
        p=corrected_probability(self.calibrator,frame,p,self.correction_weight)
        return np.column_stack([1-p,p])
