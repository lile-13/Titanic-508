"""Two-parameter, regularized probability combination on training-held-out groups."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def objective(beta, design, offset, y, ridge):
    z = offset + design @ beta
    center = np.array([0., .5])
    difference = beta - center
    loss = np.mean(np.logaddexp(0, z) - y*z) + .5*ridge*np.dot(difference, difference)
    gradient = design.T @ (expit(z)-y) / len(y) + ridge*difference
    return float(loss), gradient


@dataclass
class LogitStacker:
    ridge: float = .01
    tfm_enabled: bool = True

    def fit(self, reference, tfm, y):
        reference, tfm, y = [np.asarray(v, dtype=float) for v in [reference, tfm, y]]
        if reference.ndim != 1 or reference.shape != tfm.shape or reference.shape != y.shape or len(y) < 20:
            raise ValueError("Invalid held-out stacking rows")
        if not np.isfinite(np.r_[reference, tfm, y]).all() or not np.isin(y, [0, 1]).all() or self.ridge <= 0:
            raise ValueError("Invalid held-out stacking values")
        if np.any((reference < 0) | (reference > 1) | (tfm < 0) | (tfm > 1)):
            raise ValueError("Invalid probabilities")
        old = logit(np.clip(reference, 1e-6, 1-1e-6))
        new = logit(np.clip(tfm, 1e-6, 1-1e-6))
        design = np.column_stack([np.ones(len(y)), new-old])
        bounds = [(None, None), (0., 1.) if self.tfm_enabled else (0., 0.)]
        result = minimize(objective, [0., .5 if self.tfm_enabled else 0.],
                          args=(design, old, y, self.ridge), jac=True, bounds=bounds,
                          method="L-BFGS-B", options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8))
        if not result.success:
            raise RuntimeError("Stacking optimizer did not converge")
        self.intercept_, self.weight_ = map(float, result.x)
        self.audit_ = dict(intercept=self.intercept_, tfm_weight=self.weight_, ridge=self.ridge,
                           rows=len(y), converged=True, tfm_enabled=self.tfm_enabled)
        return self

    def predict(self, reference, tfm):
        old = logit(np.clip(np.asarray(reference, dtype=float), 1e-6, 1-1e-6))
        new = logit(np.clip(np.asarray(tfm, dtype=float), 1e-6, 1-1e-6))
        return expit(self.intercept_ + (1-self.weight_)*old + self.weight_*new)


@dataclass
class StackedTabICLFold:
    reference_members: list
    tfm_member: object
    calibrator: LogitStacker

    def predict_proba(self, frame):
        if not np.isclose(sum(m["weight"] for m in self.reference_members), 1.):
            raise ValueError("Reference fold weights must sum to one")
        ref = np.zeros(len(frame))
        for m in self.reference_members:
            X = m["encoder"].transform(m["builder"].transform(frame))
            ref += m["weight"]*m["model"].predict_proba(X)[:, 1]
        p = self.calibrator.predict(ref, self.tfm_member.predict_proba(frame)[:, 1])
        return np.column_stack([1-p, p])
