"""Bounded weights from genuinely out-of-fold training predictions."""
import numpy as np

from .competition_features import FeatureBuilder, InputEncoder
from .optimization import estimator

ARMS = ["control", "hard", "g_control", "hard_g"]


def training_weights(y, probability, is_g, arm):
    y, probability, is_g = np.asarray(y), np.asarray(probability, float), np.asarray(is_g, bool)
    if arm not in ARMS or y.shape != probability.shape or y.shape != is_g.shape or y.ndim != 1:
        raise ValueError("Invalid weighting inputs")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("Invalid training labels/probabilities")
    true_probability = np.where(y == 1, probability, 1-probability)
    weights = np.ones(len(y))
    if arm in ("hard", "hard_g"):
        weights *= 2-true_probability
    if arm in ("g_control", "hard_g"):
        weights *= np.where(is_g, 1.5, 1.)
    return weights/weights.mean()


def fit_member(frame, y, config, weights=None):
    builder = FeatureBuilder(config["feature_set"], config["strategy"]).fit(frame)
    X = builder.transform(frame)
    encoder = InputEncoder("catboost").fit(X)
    model = estimator(config, encoder.cat_columns_, threads=4)
    model.fit(encoder.transform(X), y, sample_weight=weights)
    return dict(builder=builder, encoder=encoder, model=model)


def predict(member, frame):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(frame)))[:, 1]
