"""Explicit, distinct small TabPFN V2 checkpoints; Built with PriorLabs-TabPFN."""
from dataclasses import dataclass, field
import hashlib

import numpy as np

from .paths import PROJECT_ROOT as P
from .tabpfn_models import PARAMS

CHECKPOINTS = {
    "gn2p4bpt": "de01a095da7a083b25a74d1f7d85de524e8bd6137936092feab9790673ad9215",
    "vutqq28w": "c060b5f61480e8a6ec8cbf909a0fc236563eb4628df75b7b8c2d1f8fc1b9cf4e",
    "znskzxi4": "fadb46928ddd70a53c6e1b205772e78784bf7bea5132ba6adbd48f201d8e9076",
}


@dataclass
class TabPFNDiverseMember:
    builder: object
    training_X: object
    training_y: np.ndarray
    checkpoint_key: str
    _model: object = field(default=None, repr=False)

    def predict_proba(self, frame):
        if self._model is None:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            path = P / "models/tabpfn_pretrained" / f"tabpfn-v2-classifier-{self.checkpoint_key}.ckpt"
            if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINTS[self.checkpoint_key]:
                raise ValueError("Distinct TabPFN checkpoint differs from pinned bytes")
            self._model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, model_path=path, **PARAMS).fit(self.training_X, self.training_y)
        return self._model.predict_proba(self.builder.transform(frame))

    def __getstate__(self):
        return {**self.__dict__, "_model": None}


def from_context(original, key):
    if key not in CHECKPOINTS:
        raise ValueError("Unregistered checkpoint")
    return TabPFNDiverseMember(original.builder, original.training_X.copy(deep=True), original.training_y.copy(), key)
