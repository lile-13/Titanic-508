"""Auditable original-ensemble fits for previously unused grouped partitions."""
import hashlib
import json

import joblib
import numpy as np

from .competition_features import FeatureBuilder, InputEncoder
from .optimization import estimator
from .train import _write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_reference(runner, plan, training, validation, tag, output, models, report):
    assert not set(runner.features.GroupId.iloc[training]) & set(runner.features.GroupId.iloc[validation])
    total = np.zeros(len(validation))
    member_names = []
    for index, (weight, component) in enumerate(zip(plan["weights"], plan["members"])):
        cfg = component["config"]
        builder = FeatureBuilder(cfg["feature_set"], cfg["strategy"]).fit(runner.features.iloc[training])
        X = builder.transform(runner.features.iloc[training])
        V = builder.transform(runner.features.iloc[validation])
        encoder = InputEncoder(cfg["family"]).fit(X)
        X, V = encoder.transform(X), encoder.transform(V)
        for seed in component["seeds"]:
            name = f"reference_{tag}_component_{index}_seed_{seed}"
            cache, modelpath, recordpath = output/f"{name}.npz", models/f"{name}.joblib", report/f"{name}.json"
            if cache.exists() and modelpath.exists() and recordpath.exists():
                saved = np.load(cache)
                np.testing.assert_array_equal(saved["training"], training)
                np.testing.assert_array_equal(saved["validation"], validation)
                assert sha(modelpath) == json.loads(recordpath.read_text(encoding="utf-8"))["model_sha256"]
                p = saved["probability"]
            else:
                model = estimator(dict(cfg, seed=seed), encoder.cat_columns_, runner.threads)
                model.fit(X, runner.y[training])
                p = model.predict_proba(V)[:, 1]
                joblib.dump(dict(builder=builder, encoder=encoder, model=model), modelpath, compress=3)
                np.savez_compressed(cache, training=training, validation=validation, probability=p)
                _write_json(recordpath, dict(model_sha256=sha(modelpath), config=dict(cfg, seed=seed), component=index, tag=tag))
                print(f"Frozen original {name} ready", flush=True)
            total += weight/len(component["seeds"])*p
            member_names.append(name)
    _write_json(report/f"reference_{tag}.json", dict(members=member_names))
    return total
