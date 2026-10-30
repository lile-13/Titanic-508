"""A small factorial comparison of input richness and booking-group sequence."""
from dataclasses import dataclass

import numpy as np

from .competition_features import GROUP_EXTRAS, InputEncoder
from .focused_features import FocusedBuilder, unique_value_map
from .tabicl_models import TabICLMember
from . import preprocess as pp

VIEWS = ["compact", "compact_no_sequence", "expanded", "expanded_no_sequence"]
EXTRA = ["GroupPos", "GroupSurnameCount", "CabinRelative", *GROUP_EXTRAS]


@dataclass
class TabICLInputBuilder:
    view: str = "compact"

    def fit(self, frame, context):
        if self.view not in VIEWS:
            raise ValueError("Unknown TabICL input view")
        self.base_ = FocusedBuilder("compact", True).fit(frame, context)
        self.group_side_ = unique_value_map(context, "GroupId", "Side")
        return self

    def transform(self, frame):
        d = frame.copy()
        d["Side"] = d.Side.astype(object).fillna(d.GroupId.map(self.group_side_))
        X = self.base_.transform(d)
        if self.view.startswith("expanded"):
            for column in EXTRA:
                X[column] = frame[column].to_numpy(dtype=float)
            for column in pp.SPEND_COLS:
                X["log_" + column] = np.log1p(X[column])
        if self.view.endswith("no_sequence"):
            X = X.drop(columns="GroupIdNum")
        return X.replace([np.inf, -np.inf], np.nan)


def fit_tfm(view, frame, y, context):
    y = np.asarray(y, dtype=int)
    if y.shape != (len(frame),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid in-context training labels")
    builder = TabICLInputBuilder(view).fit(frame, context)
    X = builder.transform(frame)
    if pp.TARGET in X:
        raise ValueError("Target entered the input matrix")
    return TabICLMember(builder, X.copy(deep=True), y.copy())


def fit_cat(view, frame, y, context, params):
    from catboost import CatBoostClassifier
    builder = TabICLInputBuilder(view).fit(frame, context)
    X = builder.transform(frame)
    encoder = InputEncoder("catboost").fit(X)
    model = CatBoostClassifier(**params, cat_features=encoder.cat_columns_, random_seed=42,
                               thread_count=4, loss_function="Logloss", verbose=False,
                               allow_writing_files=False)
    model.fit(encoder.transform(X), y)
    return dict(builder=builder, encoder=encoder, model=model)
