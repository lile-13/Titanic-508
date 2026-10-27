"""可恢复的开发折搜索、冻结方案及保留折确认。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from . import preprocess as pp
from .competition_features import FeatureBuilder, InputEncoder, PROFILES, build_pool, development_splits
from .paths import EXPERIMENTS_DIR, FEATURES_DIR, PROJECT_ROOT
from .train import SHANGHAI, _data_hashes, _write_csv, _write_json, make_submission, validate_fold_map


def estimator(config, categorical, threads=4):
    params = config["params"].copy()
    iterations = params.pop("iterations")
    seed = config.get("seed", 42)
    if config["family"] == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=iterations, **params, cat_features=categorical,
                                  random_seed=seed, thread_count=threads, loss_function="Logloss",
                                  eval_metric="Logloss", verbose=False, allow_writing_files=False)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=iterations, **params, random_state=seed, n_jobs=threads,
                          verbosity=-1, deterministic=True, force_col_wise=True)


def fit_model(model, family, X, y, valid=None):
    if valid is None:
        model.fit(X, y)
    elif family == "catboost":
        model.fit(X, y, eval_set=valid, early_stopping_rounds=100, use_best_model=True, verbose=False)
    else:
        import lightgbm as lgb
        model.fit(X, y, eval_set=[valid], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return model


def config_id(config, stage, folds):
    text = json.dumps([config, stage, folds], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:14]


def baseline_config(family):
    params = (dict(iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=5.0)
              if family == "catboost" else
              dict(iterations=300, learning_rate=0.05, num_leaves=15))
    return dict(family=family, feature_set="base", strategy="rule_group", params=params, seed=42)


def screen_params(family):
    if family == "catboost":
        return dict(iterations=1200, depth=6, learning_rate=0.05, l2_leaf_reg=7.0,
                    max_ctr_complexity=1)
    return dict(iterations=1800, learning_rate=0.04, num_leaves=15,
                min_child_samples=30, reg_alpha=0.1, reg_lambda=3.0)


class SearchRunner:
    def __init__(self, name="optimization_20261025", threads=4):
        self.name, self.threads = name, threads
        self.report = EXPERIMENTS_DIR / name
        self.output = PROJECT_ROOT / "outputs" / name
        self.report.mkdir(parents=True, exist_ok=True)
        (self.output / "search").mkdir(parents=True, exist_ok=True)
        self.raw, self.raw_test = pp.load_raw()
        self.features, self.test_features = build_pool(self.raw, self.raw_test)
        self.y = self.raw.Transported.astype(int).to_numpy()
        folds = pd.read_csv(FEATURES_DIR / "folds.csv", dtype={"PassengerId": str, "GroupId": str})
        self.fold_of = validate_fold_map(self.features, folds)
        self.splits = development_splits(self.fold_of)
        self.cache = {}
        protocol = dict(name=name, development_folds=[1, 2, 3, 4], confirmation_fold=5,
                        cohort="public train+test unlabeled features", data_hashes=_data_hashes(pp.find_data_dir()),
                        holdout_history="Fold 5 appeared in earlier baselines; excluded from this round of selection",
                        promotion_rule="dev gain >=0.003 and frozen confirmation >= baseline confirmation minus 0.003",
                        sources={str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), PROJECT_ROOT / "src/competition_features.py"]})
        protocol_path = self.report / "protocol.json"
        if protocol_path.exists():
            old = json.loads(protocol_path.read_text(encoding="utf-8"))
            if old["data_hashes"] != protocol["data_hashes"] or old["sources"] != protocol["sources"]:
                raise ValueError("搜索代码或数据已变化，请使用新的 --name，不能混合旧试验")
        else:
            _write_json(protocol_path, protocol)

    def prepared(self, family, feature_set, strategy, fold):
        key = (family, feature_set, strategy, fold)
        if key not in self.cache:
            ti, vi = self.splits[fold - 1]
            builder = FeatureBuilder(feature_set, strategy).fit(self.features.iloc[ti])
            Xtr = builder.transform(self.features.iloc[ti])
            Xva = builder.transform(self.features.iloc[vi])
            encoder = InputEncoder(family).fit(Xtr)
            self.cache[key] = (encoder.transform(Xtr), encoder.transform(Xva), encoder.cat_columns_, ti, vi)
        return self.cache[key]

    def evaluate(self, config, stage, folds=(1, 2, 3, 4), early=True):
        trial = config_id(config, stage, list(folds))
        record_file = self.report / f"trial_{trial}.json"
        prob_file = self.output / "search" / f"{trial}.npz"
        if record_file.exists() and prob_file.exists():
            return json.loads(record_file.read_text(encoding="utf-8"))
        started = time.perf_counter()
        probability = np.full(len(self.y), np.nan)
        scores, losses, iterations = [], [], []
        for fold in folds:
            X, V, cats, ti, vi = self.prepared(config["family"], config["feature_set"], config["strategy"], fold)
            assert not (self.fold_of[ti] == 5).any() and not (self.fold_of[vi] == 5).any()
            model = estimator(config, cats, self.threads)
            fit_model(model, config["family"], X, self.y[ti], (V, self.y[vi]) if early else None)
            pred = model.predict_proba(V)[:, 1]
            probability[vi] = pred
            scores.append(float(accuracy_score(self.y[vi], pred >= 0.5)))
            losses.append(float(log_loss(self.y[vi], pred)))
            iterations.append(int(model.tree_count_ if config["family"] == "catboost" else
                                  (model.best_iteration_ or config["params"]["iterations"])))
        record = dict(trial=trial, stage=stage, config=config, folds=list(folds),
                      accuracy=float(np.mean(scores)), std=float(np.std(scores, ddof=1)),
                      log_loss=float(np.mean(losses)), fold_scores=scores, iterations=iterations,
                      seconds=time.perf_counter()-started, early_stopping=early)
        np.savez_compressed(prob_file, probability=probability)
        _write_json(record_file, record)
        print(f"{stage} {config['family']} {config['feature_set']}/{config['strategy']} "
              f"acc={record['accuracy']:.4f} loss={record['log_loss']:.4f} "
              f"iter={int(np.median(iterations))} [{record['seconds']:.1f}s]", flush=True)
        return record

    def records(self, family=None, stages=None):
        records = [json.loads(p.read_text(encoding="utf-8")) for p in self.report.glob("trial_*.json")]
        return [r for r in records if (family is None or r["config"]["family"] == family) and
                (stages is None or r["stage"] in stages)]

    def probability(self, record):
        return np.load(self.output / "search" / f"{record['trial']}.npz")["probability"]

    def search(self, family, trials=32):
        self.evaluate(baseline_config(family), "baseline", early=False)
        screens = []
        for feature_set, strategy in PROFILES:
            c = dict(family=family, feature_set=feature_set, strategy=strategy,
                     params=screen_params(family), seed=42)
            screens.append(self.evaluate(c, "screen", folds=(1, 2)))
        screens.sort(key=lambda r: (-r["accuracy"], r["log_loss"]))
        full = [self.evaluate(r["config"], "profile") for r in screens[:3]]
        full.sort(key=lambda r: (-r["accuracy"], r["log_loss"]))
        rng = np.random.default_rng(20261025 if family == "catboost" else 20261026)
        for i in range(trials):
            source = full[int(rng.choice(len(full), p=[0.5, 0.3, 0.2][:len(full)]))]["config"]
            if family == "catboost":
                params = dict(iterations=2200, depth=int(rng.choice([4, 5, 6, 7])),
                              learning_rate=float(np.exp(rng.uniform(np.log(.02), np.log(.08)))),
                              l2_leaf_reg=float(np.exp(rng.uniform(np.log(1.5), np.log(20)))),
                              random_strength=float(rng.choice([0., .5, 1., 2.])),
                              bagging_temperature=float(rng.uniform(0, 2)), rsm=float(rng.uniform(.8, 1)),
                              max_ctr_complexity=int(rng.choice([1, 2])),
                              one_hot_max_size=int(rng.choice([2, 4, 10])),
                              border_count=int(rng.choice([128, 254])))
            else:
                params = dict(iterations=2500, learning_rate=float(np.exp(rng.uniform(np.log(.015), np.log(.08)))),
                              num_leaves=int(rng.choice([7, 11, 15, 23, 31])),
                              min_child_samples=int(rng.choice([15, 25, 40, 60, 100])),
                              reg_alpha=float(np.exp(rng.uniform(np.log(.001), np.log(8)))),
                              reg_lambda=float(np.exp(rng.uniform(np.log(.1), np.log(20)))),
                              colsample_bytree=float(rng.choice([.65, .8, 1.])),
                              subsample=float(rng.choice([.7, .85, 1.])), subsample_freq=1,
                              max_bin=int(rng.choice([63, 127, 255])), extra_trees=bool(rng.integers(0, 2)))
            c = dict(family=family, feature_set=source["feature_set"], strategy=source["strategy"], params=params, seed=42)
            print(f"Trial {i+1}/{trials}", flush=True)
            self.evaluate(c, "tune")
        self.export_table(family)

    def export_table(self, family=None):
        rows = []
        for r in self.records(family):
            rows.append(dict(trial=r["trial"], stage=r["stage"], family=r["config"]["family"],
                             features=r["config"]["feature_set"], strategy=r["config"]["strategy"],
                             accuracy=r["accuracy"], std=r["std"], log_loss=r["log_loss"],
                             median_iteration=int(np.median(r["iterations"])), seconds=r["seconds"]))
        _write_csv(pd.DataFrame(rows).sort_values(["stage", "accuracy"], ascending=[True, False]),
                   self.report / f"trials_{family or 'all'}.csv")

    def freeze(self):
        path = self.report / "frozen_plan.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        members, probabilities = [], []
        for family, count in [("catboost", 2), ("lightgbm", 1)]:
            ranked = sorted(self.records(family, ["tune", "profile", "baseline"]),
                            key=lambda r: (-r["accuracy"], r["log_loss"]))
            if not ranked:
                raise ValueError(f"先完成{family}搜索")
            choices = [ranked[0]]
            if count == 2:
                other = next((r for r in ranked[1:] if r["config"]["feature_set"] != ranked[0]["config"]["feature_set"]
                              or r["config"]["strategy"] != ranked[0]["config"]["strategy"]), ranked[1])
                choices.append(other)
            for chosen in choices:
                c = json.loads(json.dumps(chosen["config"]))
                c["params"]["iterations"] = max(100, int(np.median(chosen["iterations"])))
                predictions = []
                for seed in [42, 2026, 103]:
                    c["seed"] = seed
                    r = self.evaluate(c.copy(), "confirmation", early=False)
                    predictions.append(self.probability(r))
                members.append(dict(config=c.copy(), seeds=[42, 2026, 103]))
                probabilities.append(np.mean(predictions, axis=0))
        dev = self.fold_of != 5
        rows = []
        for a in range(5):
            for b in range(5-a):
                weights = np.array([a, b, 4-a-b])/4
                pred = np.sum([w*p[dev] for w, p in zip(weights, probabilities)], axis=0)
                for threshold in [.45, .475, .5, .525, .55]:
                    score = accuracy_score(self.y[dev], pred >= threshold)
                    rows.append(dict(weights=weights.tolist(), threshold=threshold, accuracy=float(score),
                                     log_loss=float(log_loss(self.y[dev], pred))))
        # 接近最优时偏向0.5阈值和更简单的组合，避免为一两个开发样本增加复杂度。
        best = max(r["accuracy"] for r in rows)
        near = [r for r in rows if r["accuracy"] >= best - 2/dev.sum()]
        near.sort(key=lambda r: (abs(r["threshold"]-.5), sum(w>0 for w in r["weights"]),
                                  -r["accuracy"], r["log_loss"]))
        selected = near[0]
        baseline = self.evaluate(baseline_config("catboost"), "baseline", early=False)
        plan = dict(name=self.name, locked_at=datetime.now(SHANGHAI).isoformat(), members=members,
                    weights=selected["weights"], threshold=selected["threshold"],
                    development_accuracy=selected["accuracy"], development_best_scanned=best,
                    development_baseline=baseline["accuracy"], confirmation_fold=5,
                    note="Frozen before this round reads confirmation-fold labels for model evaluation")
        _write_json(path, plan)
        _write_json(self.report / "development_mixture_search.json", rows)
        self.export_table()
        return plan

    def fit_partition(self, plan, ti, vi, *, save_test=True):
        validation = np.zeros(len(vi))
        test_pred = np.zeros(len(self.test_features))
        fitted = []
        for weight, member in zip(plan["weights"], plan["members"]):
            if weight == 0:
                continue
            c = member["config"]
            builder = FeatureBuilder(c["feature_set"], c["strategy"]).fit(self.features.iloc[ti])
            X = builder.transform(self.features.iloc[ti])
            V = builder.transform(self.features.iloc[vi])
            T = builder.transform(self.test_features) if save_test else None
            encoder = InputEncoder(c["family"]).fit(X)
            X, V = encoder.transform(X), encoder.transform(V)
            if save_test:
                T = encoder.transform(T)
            for seed in member["seeds"]:
                seeded = dict(c, seed=seed)
                model = estimator(seeded, encoder.cat_columns_, self.threads)
                fit_model(model, c["family"], X, self.y[ti])
                w = weight / len(member["seeds"])
                validation += w * model.predict_proba(V)[:, 1]
                if save_test:
                    test_pred += w * model.predict_proba(T)[:, 1]
                    fitted.append(dict(weight=w, builder=builder, encoder=encoder, model=model))
        return validation, test_pred, fitted

    def finalize(self):
        plan = self.freeze()
        result_path = self.report / "final_summary.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        ti, vi = np.flatnonzero(self.fold_of != 5), np.flatnonzero(self.fold_of == 5)
        p, tp, fitted = self.fit_partition(plan, ti, vi)
        holdout_score = float(accuracy_score(self.y[vi], p >= plan["threshold"]))
        baseline_oof = pd.read_csv(EXPERIMENTS_DIR / "training/catboost_baseline_20261018/oof_predictions.csv",
                                   dtype={"PassengerId": str}).set_index("PassengerId").loc[self.raw.PassengerId]
        np.testing.assert_array_equal(baseline_oof.fold, self.fold_of)
        baseline_score = float(accuracy_score(self.y[vi], baseline_oof.probability.to_numpy()[vi] >= .5))
        promoted = (plan["development_accuracy"] >= plan["development_baseline"]+.003 and
                    holdout_score >= baseline_score-.003)
        holdout = dict(accuracy=holdout_score, baseline_accuracy=baseline_score, n=len(vi),
                       promoted=promoted, threshold=plan["threshold"],
                       historical_caveat="Previously inspected in old baselines; unused for current search")
        _write_json(self.report / "confirmation_result.json", holdout)
        print(f"FROZEN CONFIRMATION: {holdout_score:.4f} vs baseline {baseline_score:.4f}; promoted={promoted}", flush=True)
        if not promoted:
            _write_json(result_path, dict(status="not_promoted", plan=plan, confirmation=holdout))
            return json.loads(result_path.read_text(encoding="utf-8"))
        probabilities = np.full(len(self.y), np.nan)
        probabilities[vi] = p
        test_probability = tp / 5
        saved_models = [dict(m, weight=m["weight"]/5) for m in fitted]
        metrics = [dict(fold=5, accuracy=holdout_score, log_loss=float(log_loss(self.y[vi], p)))]
        for fold in range(1, 5):
            tri, vai = np.flatnonzero(self.fold_of != fold), np.flatnonzero(self.fold_of == fold)
            pred, test, models = self.fit_partition(plan, tri, vai)
            probabilities[vai] = pred
            test_probability += test/5
            saved_models += [dict(m, weight=m["weight"]/5) for m in models]
            metrics.append(dict(fold=fold, accuracy=float(accuracy_score(self.y[vai], pred >= plan["threshold"])),
                                log_loss=float(log_loss(self.y[vai], pred))))
            print(f"Frozen five-fold refit {fold}/5: {metrics[-1]['accuracy']:.4f}", flush=True)
        assert np.isfinite(probabilities).all() and np.isfinite(test_probability).all()
        oof = pd.DataFrame(dict(PassengerId=self.raw.PassengerId, GroupId=self.features.GroupId,
                                fold=self.fold_of, Transported=self.y.astype(bool), probability=probabilities,
                                prediction=probabilities >= plan["threshold"]))
        _write_csv(oof, self.report / "oof_predictions.csv")
        _write_csv(pd.DataFrame(metrics).sort_values("fold"), self.report / "fold_metrics.csv")
        _write_csv(pd.DataFrame(dict(PassengerId=self.raw_test.PassengerId, probability=test_probability,
                                    prediction=test_probability >= plan["threshold"])), self.output / "test_predictions.csv")
        submission = make_submission(self.raw_test.PassengerId, test_probability,
                                     pp.load_submission_template(), plan["threshold"])
        _write_csv(submission, self.output / "submission.csv")
        model_dir = PROJECT_ROOT / "models" / self.name
        model_dir.mkdir(exist_ok=True)
        bundle = dict(members=saved_models, plan=plan, data_hashes=_data_hashes(pp.find_data_dir()))
        joblib.dump(bundle, model_dir / "ensemble.joblib", compress=3)
        scores = [r["accuracy"] for r in metrics]
        result = dict(status="promoted_candidate", name=self.name, plan=plan, confirmation=holdout,
                      selected_oof_mean=float(np.mean(scores)), selected_oof_std=float(np.std(scores,ddof=1)),
                      oof_accuracy=float(accuracy_score(self.y, probabilities>=plan["threshold"])),
                      oof_log_loss=float(log_loss(self.y, probabilities)),
                      models=len(saved_models), submitted_to_kaggle=False,
                      interpretation="Folds1-4 participated in selection; mean is post-selection, not an unbiased score",
                      bundle=str(model_dir / "ensemble.joblib"), submission=str(self.output / "submission.csv"))
        _write_json(result_path, result)
        return result


def predict_ensemble(bundle_path, data_dir=None):
    bundle = joblib.load(bundle_path)
    directory = pp.find_data_dir(data_dir)
    if _data_hashes(directory) != bundle["data_hashes"]:
        raise ValueError("公开数据池已变化，需要重新训练")
    raw, test = pp.load_raw(directory)
    _, features = build_pool(raw, test)
    pred = np.zeros(len(test))
    for member in bundle["members"]:
        X = member["encoder"].transform(member["builder"].transform(features))
        pred += member["weight"] * member["model"].predict_proba(X)[:,1]
    submission = make_submission(test.PassengerId, pred, pp.load_submission_template(directory),
                                 bundle["plan"]["threshold"])
    return pred, submission
