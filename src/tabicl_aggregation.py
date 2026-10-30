"""Retain all fixed TabICL views and use their arithmetic probability mean."""
from .tabicl_models import TabICLMember, make_model, fit_member


class ProbabilityMeanMember(TabICLMember):
    def predict_proba(self, frame):
        if self._model is None:
            model = make_model()
            model.set_params(average_logits=False)
            self._model = model.fit(self.training_X, self.training_y)
        return self._model.predict_proba(self.builder.transform(frame))


def from_context(member):
    return ProbabilityMeanMember(member.builder, member.training_X.copy(deep=True), member.training_y.copy())


def fit_probability_mean(frame, y, context):
    return from_context(fit_member(frame, y, context))
