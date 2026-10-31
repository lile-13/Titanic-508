"""Optional fine-tuning helpers, isolated from historical inference modules."""
from tabicl import FinetunedTabICLClassifier

from .tabicl_models import make_model


PARAMS = dict(epochs=30, learning_rate=1e-5, weight_decay=.01, grad_clip=1.,
    max_data_size=1024, finetune_ctx_query_ratio=.2,
    n_estimators_finetune=1, n_estimators_validation=8, n_estimators_inference=8,
    freeze_col=True, freeze_row=True, freeze_icl=False, amp=False,
    average_logits=False, eval_metric="log_loss", random_state=42, device="cuda",
    patience=5, min_delta=1e-4, early_stopping=True, save_interval=0,
    allow_auto_download=False, wandb_kwargs=None, verbose=False,
    extra_classifier_kwargs=dict(batch_size=2, use_amp=True, use_fa3=False, n_jobs=4))


class ObservedFinetune(FinetunedTabICLClassifier):
    """Record inner scores without changing the author's update or selection code."""
    def _validate_current_model(self, *args, **kwargs):
        result = super()._validate_current_model(*args, **kwargs)
        if not hasattr(self, "validation_history_"):
            self.validation_history_ = []
        self.validation_history_.append(dict(primary=float(result.primary), **result.secondary))
        return result


def predictor(checkpoint, context):
    model = make_model().set_params(model_path=str(checkpoint), average_logits=False)
    return model.fit(context["encoded_X"], context["y"])
