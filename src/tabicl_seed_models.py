"""Keep eight probability-averaged views per independently seeded context."""
from dataclasses import dataclass

from .tabicl_models import TabICLMember,make_model


@dataclass
class SeedMember(TabICLMember):
    seed:int=42

    def predict_proba(self,frame):
        if self._model is None:
            self._model=make_model().set_params(random_state=self.seed,average_logits=False).fit(self.training_X,self.training_y)
        return self._model.predict_proba(self.builder.transform(frame))


def from_context(original,seed):
    return SeedMember(original.builder,original.training_X.copy(deep=True),original.training_y.copy(),seed=seed)
