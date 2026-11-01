"""Portable V2 contexts; the optional TabPFN runtime is never pickled."""
from dataclasses import dataclass, field
import hashlib

import numpy as np

from .paths import PROJECT_ROOT as P

CHECKPOINT="tabpfn-v2-classifier-v2_default.ckpt"
CHECKPOINT_SHA256="cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
PARAMS=dict(n_estimators=4,device="cuda",random_state=42,n_preprocessing_jobs=1,
    fit_mode="low_memory",softmax_temperature=.9,average_before_softmax=False,
    inference_precision="autocast",memory_saving_mode="auto",show_progress_bar=False)


def make_model():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    path=P/"models/tabpfn_pretrained"/CHECKPOINT
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
        raise ValueError("TabPFN author checkpoint differs from pinned SHA256")
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2,model_path=path,**PARAMS)


@dataclass
class TabPFNMember:
    builder: object
    training_X: object
    training_y: np.ndarray
    _model: object=field(default=None,repr=False)

    def predict_proba(self,frame):
        if self._model is None:
            self._model=make_model().fit(self.training_X,self.training_y)
        return self._model.predict_proba(self.builder.transform(frame))

    def __getstate__(self):
        return {**self.__dict__,"_model":None}


def from_context(original):
    return TabPFNMember(original.builder,original.training_X.copy(deep=True),original.training_y.copy())
