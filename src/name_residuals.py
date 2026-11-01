"""Regularized name effects learned from training-only cross-fit residuals."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit,logit


def objective(beta,design,y,offset,penalty):
    z=offset+design@beta
    loss=float(np.sum(np.logaddexp(0,z)-y*z)+.5*penalty*np.dot(beta,beta))
    gradient=np.asarray(design.T@(expit(z)-y)).ravel()+penalty*beta
    return loss,gradient


@dataclass
class NameResidual:
    include_names:bool=True
    penalty:float=4.
    strength:float=.5

    def design(self,frame):
        rows=list(range(len(frame)))
        cols=[0]*len(frame)
        for column,mapping in self.maps_.items():
            for i,value in enumerate(frame[column]):
                if pd.notna(value) and str(value) in mapping:
                    rows.append(i);cols.append(mapping[str(value)])
        return csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(frame),self.n_parameters_))

    def fit(self,frame,y,teacher_probability):
        y=np.asarray(y,dtype=float)
        p=np.asarray(teacher_probability,dtype=float)
        if y.shape!=(len(frame),) or p.shape!=y.shape or len(y)==0 or not np.isin(y,[0,1]).all():
            raise ValueError("Invalid training targets")
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("Invalid teacher predictions")
        self.maps_={}
        self.n_parameters_=1
        if self.include_names:
            for column in ["FirstName","Surname"]:
                values=sorted(frame[column].dropna().map(str).unique().tolist())
                self.maps_[column]={v:self.n_parameters_+i for i,v in enumerate(values)}
                self.n_parameters_+=len(values)
        design=self.design(frame)
        result=minimize(objective,np.zeros(self.n_parameters_),args=(design,y,logit(np.clip(p,1e-6,1-1e-6)),self.penalty),
            method="L-BFGS-B",jac=True,options=dict(maxiter=200,ftol=1e-11,gtol=1e-6))
        if not result.success or not np.isfinite(result.x).all():raise ValueError(f"Name optimization failed: {result.message}")
        self.parameters_=result.x
        self.optimization_=dict(iterations=int(result.nit),objective=float(result.fun),
            max_gradient=float(np.max(np.abs(result.jac))),parameters=self.n_parameters_)
        return self

    def predict(self,frame,probability):
        p=np.asarray(probability,dtype=float)
        if p.shape!=(len(frame),) or not np.isfinite(p).all():raise ValueError("Invalid query probability")
        return expit(logit(np.clip(p,1e-6,1-1e-6))+self.strength*(self.design(frame)@self.parameters_))
