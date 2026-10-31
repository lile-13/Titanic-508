"""One synthetic CUDA epoch: feasibility only, no competition rows or labels."""
import hashlib
import importlib.metadata
import logging
from pathlib import Path
import time

import numpy as np
import torch
from tabicl import FinetunedTabICLClassifier, TabICLClassifier

from src.paths import PROJECT_ROOT
from src.tabicl_models import CHECKPOINT, CHECKPOINT_SHA256
from src.train import _write_json


def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    np.random.seed(42)
    original = PROJECT_ROOT / "models/tabicl_pretrained" / CHECKPOINT
    assert hashlib.sha256(original.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    folder = PROJECT_ROOT / "outputs/finetune_feasibility_20261031"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(1088, 18))
    y = (X[:, 0] + .5 * X[:, 1] * X[:, 2] + rng.normal(scale=.5, size=len(X)) > 0).astype(int)
    params = dict(epochs=1, learning_rate=1e-5, max_data_size=512,
        n_estimators_finetune=1, n_estimators_validation=2, n_estimators_inference=8,
        freeze_col=True, freeze_row=True, freeze_icl=False, amp=False,
        average_logits=False, eval_metric="log_loss", random_state=42, device="cuda",
        model_path=str(original), allow_auto_download=False,
        extra_classifier_kwargs=dict(batch_size=2, use_amp=True, use_fa3=False, n_jobs=4),
        wandb_kwargs=None, verbose=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = FinetunedTabICLClassifier(**params)
    model.fit(X[:768], y[:768], X_val=X[768:1024], y_val=y[768:1024], output_dir=folder)
    p = model.predict_proba(X[1024:])
    elapsed = time.perf_counter()-start
    assert p.shape == (64, 2) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    before = torch.load(original, map_location="cpu", weights_only=False)["state_dict"]
    after = torch.load(folder / "epoch1.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed, "Training performed no weight updates"
    assert all(torch.isfinite(v).all() for v in after.values())
    # Compare the chosen best checkpoint to the fitted wrapper, not epoch1:
    # early stopping may legitimately retain the original weights.
    reloaded = TabICLClassifier(model_path=str(folder / "best.ckpt"), allow_auto_download=False,
        n_estimators=8, average_logits=False, random_state=42, device="cuda", batch_size=2,
        use_amp=True, use_fa3=False, n_jobs=4)
    reloaded.fit(model._X_encoder_.transform(X[:768]), y[:768])
    q = reloaded.predict_proba(model._X_encoder_.transform(X[1024:]))
    np.testing.assert_allclose(p, q, rtol=0, atol=2e-6)
    result = dict(passed=True, dataset="synthetic only; no Spaceship Titanic rows used",
        train_rows=768, validation_rows=256, query_rows=64, features=18,
        fit_and_query_seconds=elapsed, peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
        changed_tensor_count=len(changed), changed_modules=sorted({k.split('.')[0] for k in changed}),
        max_reload_error=float(np.abs(p-q).max()),
        device=torch.cuda.get_device_name(), parameters={**params, "model_path": "shared author checkpoint"},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=CHECKPOINT_SHA256,
        environment={k:importlib.metadata.version(k) for k in ["tabicl", "torch", "transformers", "numpy", "scikit-learn"]},
        caveat="Hardware/API smoke test only. This is not a competition CV result or a claim of accuracy improvement.")
    _write_json(PROJECT_ROOT / "reports/checks/finetune_feasibility_20261031.json", result)
    print(result)


if __name__ == "__main__":
    main()
