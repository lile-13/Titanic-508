"""Expand the fixed arithmetic TabICL ensemble without changing its context."""
import copy

from .tabicl_models import TabICLMember, make_model


class Views32Member(TabICLMember):
    def predict_proba(self, frame):
        if self._model is None:
            model = make_model()
            model.set_params(average_logits=False, n_estimators=32)
            self._model = model.fit(self.training_X, self.training_y)
        return self._model.predict_proba(self.builder.transform(frame))


def from_context(member):
    return Views32Member(copy.deepcopy(member.builder), member.training_X.copy(deep=True), member.training_y.copy())
