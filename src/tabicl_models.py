"""Portable TabICL training contexts; optional deep-learning imports stay lazy."""
from dataclasses import dataclass, field
import hashlib

import numpy as np

from .focused_features import FocusedBuilder
from .paths import PROJECT_ROOT

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
PARAMS = dict(n_estimators=8, batch_size=2, device="cuda", use_amp=True, use_fa3=False,
              random_state=42, n_jobs=4, kv_cache=False, verbose=False,
              allow_auto_download=False, checkpoint_version=CHECKPOINT)


def make_model():
    from tabicl import TabICLClassifier
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen TabICL study requires a CUDA GPU")
    path = PROJECT_ROOT / "models/tabicl_pretrained" / CHECKPOINT
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Author checkpoint does not match the frozen SHA256")
    return TabICLClassifier(**PARAMS, model_path=str(path))


@dataclass
class TabICLMember:
    builder: object
    training_X: object
    training_y: np.ndarray
    _model: object = field(default=None, repr=False)

    def predict_proba(self, frame):
        if self._model is None:
            self._model = make_model().fit(self.training_X, self.training_y)
        return self._model.predict_proba(self.builder.transform(frame))

    def __getstate__(self):
        # Share one verified checkpoint; serialized contexts have no machine-specific paths.
        return {**self.__dict__, "_model": None}


def fit_member(frame, y, context):
    y = np.asarray(y, dtype=int)
    if y.shape != (len(frame),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid training context labels")
    builder = FocusedBuilder("compact", True).fit(frame, context)
    X = builder.transform(frame)
    if "Transported" in X:
        raise ValueError("Target entered the feature context")
    return TabICLMember(builder, X.copy(deep=True), y.copy())
