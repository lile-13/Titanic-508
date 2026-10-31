"""Optional author TabM model with training-only preprocessing and group stopping."""
from dataclasses import dataclass, field
import random

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

from . import preprocess as pp
from .tabicl_inputs import TabICLInputBuilder


PARAMS = dict(k=32, d_block=512, n_blocks=2, dropout=.1, d_embedding=16,
              n_bins=16, lr=.002, weight_decay=.0003, batch_size=256,
              max_epochs=200, patience=20, seed=42, inner_seed=312026)


class TabMEncoder:
    def numeric(self, X):
        d = X[self.numeric_columns_].astype(float).replace([np.inf, -np.inf], np.nan).copy()
        for c in [*pp.SPEND_COLS, "TotalSpend"]:
            if c in d:
                d[c] = np.log1p(d[c].clip(lower=0))
        return d

    def fit(self, X):
        if pp.TARGET in X:
            raise ValueError("Target entered TabM inputs")
        self.columns_ = list(X)
        self.categorical_ = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
        self.numeric_columns_ = [c for c in X if c not in self.categorical_]
        d = self.numeric(X)
        self.medians_ = d.median().fillna(0.)
        d = d.fillna(self.medians_)
        self.active_ = [c for c in d if d[c].nunique() > 1]
        self.scales_ = (d.quantile(.75) - d.quantile(.25)).replace(0, 1.)
        self.categories_ = {c: {v: i+1 for i, v in enumerate(sorted(X[c].fillna("__missing__").map(str).unique()))}
                            for c in self.categorical_}
        self.cardinalities_ = [len(self.categories_[c])+1 for c in self.categorical_]
        return self

    def transform(self, X):
        X = X.loc[:, self.columns_]
        d = ((self.numeric(X).fillna(self.medians_) - self.medians_) / self.scales_)[self.active_]
        num = np.clip(d.to_numpy(dtype=np.float32), -8, 8)
        cat = (np.column_stack([X[c].fillna("__missing__").map(str).map(self.categories_[c]).fillna(0).to_numpy(dtype=np.int64)
                                for c in self.categorical_]) if self.categorical_ else np.empty((len(X), 0), dtype=np.int64))
        if not np.isfinite(num).all():
            raise ValueError("Nonfinite TabM input")
        return num, cat


def setup(seed):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This fixed study requires CUDA")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")


def make_model(encoder, bins):
    import torch
    from tabm import TabM
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings
    embeddings = PiecewiseLinearEmbeddings([torch.as_tensor(b) for b in bins],
                                           d_embedding=PARAMS["d_embedding"], activation=False, version="B")
    return TabM.make(n_num_features=len(encoder.active_), cat_cardinalities=encoder.cardinalities_,
                     d_out=1, num_embeddings=embeddings, k=PARAMS["k"], d_block=PARAMS["d_block"],
                     n_blocks=PARAMS["n_blocks"], dropout=PARAMS["dropout"])


def probabilities(model, num, cat):
    import torch
    model.eval()
    with torch.no_grad():
        parts = [model(num[i:i+512], cat[i:i+512]).squeeze(-1).sigmoid().mean(1).cpu().numpy()
                 for i in range(0, len(num), 512)]
    return np.concatenate(parts)


def train_network(encoder, X, y, seed, validation=None, epochs=None):
    import torch
    from rtdl_num_embeddings import compute_bins
    setup(seed)
    num, cat = encoder.transform(X)
    bins = [b.numpy().copy() for b in compute_bins(torch.from_numpy(num), n_bins=min(PARAMS["n_bins"], len(X)-1))]
    model = make_model(encoder, bins).cuda()
    num = torch.as_tensor(num, device="cuda")
    cat = torch.as_tensor(cat, device="cuda")
    target = torch.as_tensor(np.asarray(y), dtype=torch.float32, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
    if validation is not None:
        vx, vy = validation
        vn, vc = encoder.transform(vx)
        vn, vc = torch.as_tensor(vn, device="cuda"), torch.as_tensor(vc, device="cuda")
    best_loss, best_epoch, stale, losses = float("inf"), 1, 0, []
    for epoch in range(1, (PARAMS["max_epochs"] if epochs is None else epochs)+1):
        model.train()
        for idx in torch.randperm(len(X), device="cuda").split(PARAMS["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(num[idx], cat[idx]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[idx, None].expand_as(logits))
            loss.backward()
            optimizer.step()
        if validation is not None:
            value = float(log_loss(vy, probabilities(model, vn, vc), labels=[0, 1]))
            losses.append(value)
            if value < best_loss-1e-5:
                best_loss, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if epoch % 20 == 0:
                print(f"  inner epoch {epoch}, loss {value:.6f}, best {best_epoch}", flush=True)
            if stale >= PARAMS["patience"]:
                break
        elif epoch % 50 == 0:
            print(f"  refit epoch {epoch}/{epochs}", flush=True)
    state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    del model, optimizer
    torch.cuda.empty_cache()
    return bins, state, dict(selected_epoch=best_epoch, best_inner_log_loss=best_loss,
                             inner_losses=losses, stopped_epoch=epoch)


@dataclass
class TabMMember:
    builder: object
    encoder: object
    bins: object
    state: object
    training_ids: list
    selection: dict
    _model: object = field(default=None, repr=False)

    def predict_proba(self, frame):
        import torch
        if self._model is None:
            setup(PARAMS["seed"])
            self._model = make_model(self.encoder, self.bins)
            self._model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.state.items()})
            self._model.cuda().eval()
        num, cat = self.encoder.transform(self.builder.transform(frame))
        p = probabilities(self._model, torch.as_tensor(num, device="cuda"), torch.as_tensor(cat, device="cuda"))
        return np.column_stack([1-p, p])

    def __getstate__(self):
        return {**self.__dict__, "_model": None}


def fit_member(frame, y, public):
    import torch
    y = np.asarray(y, dtype=int)
    if y.shape != (len(frame),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid TabM training labels")
    ti, vi = next(GroupShuffleSplit(1, test_size=.18, random_state=PARAMS["inner_seed"]).split(frame, y, frame.GroupId))
    assert not set(frame.GroupId.iloc[ti]) & set(frame.GroupId.iloc[vi])
    builder = TabICLInputBuilder("compact").fit(frame.iloc[ti], public)
    X, V = builder.transform(frame.iloc[ti]), builder.transform(frame.iloc[vi])
    encoder = TabMEncoder().fit(X)
    _, _, selection = train_network(encoder, X, y[ti], PARAMS["seed"], validation=(V, y[vi]))
    selection.update(inner_train_ids=frame.PassengerId.iloc[ti].tolist(), inner_validation_ids=frame.PassengerId.iloc[vi].tolist(),
                     inner_group_overlap=0, preprocessing_fitted_on_inner_training_only=True)
    builder = TabICLInputBuilder("compact").fit(frame, public)
    X = builder.transform(frame)
    encoder = TabMEncoder().fit(X)
    bins, state, _ = train_network(encoder, X, y, PARAMS["seed"], epochs=selection["selected_epoch"])
    torch.cuda.empty_cache()
    return TabMMember(builder, encoder, bins, state, frame.PassengerId.tolist(), selection)
