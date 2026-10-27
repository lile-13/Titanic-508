"""成员1训练流程：统一五折、逐折预处理、折外预测与最终全量训练。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import preprocess as pp
from .paths import EXPERIMENTS_DIR, FEATURES_DIR, PROJECT_ROOT

SHANGHAI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "logreg"
    strategy: str = "rule_group"
    seed: int = 42
    threshold: float = 0.5
    c: float = 1.0
    iterations: int = 600
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 5.0
    threads: int = 2

    def __post_init__(self):
        if self.model not in ("logreg", "catboost"):
            raise ValueError("model 必须是 logreg 或 catboost")
        if self.strategy not in ("simple", "indicator", "rule_group"):
            raise ValueError("正式训练只支持保留所有乘客的填补策略")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold 必须位于 (0, 1)")
        if self.c <= 0 or self.iterations < 1 or not 1 <= self.depth <= 16:
            raise ValueError("C、iterations 或 depth 不合法")
        if not 0 < self.learning_rate <= 1 or self.l2_leaf_reg < 0 or self.threads < 1:
            raise ValueError("learning_rate、l2_leaf_reg 或 threads 不合法")


def validate_fold_map(features: pd.DataFrame, fold_map: pd.DataFrame) -> np.ndarray:
    """按 PassengerId 对齐，并拒绝漏行、重复、额外乘客和同行组跨折。"""
    for name, frame in [("训练特征", features), ("折映射", fold_map)]:
        if "PassengerId" not in frame or frame.PassengerId.isna().any():
            raise ValueError(f"{name} 缺少有效 PassengerId")
        if not frame.PassengerId.is_unique:
            raise ValueError(f"{name} 包含重复 PassengerId")
    if "fold" not in fold_map or not fold_map.fold.isin(range(1, 6)).all():
        raise ValueError("fold 必须为 1~5")
    if set(features.PassengerId) != set(fold_map.PassengerId):
        raise ValueError("fold 映射必须恰好覆盖全部训练乘客")
    fold_of = features.PassengerId.map(fold_map.set_index("PassengerId").fold)
    if set(fold_of) != set(range(1, 6)):
        raise ValueError("五个 fold 必须都有样本")
    check = pd.DataFrame({"group": features.GroupId.to_numpy(), "fold": fold_of.to_numpy()})
    if (check.groupby("group").fold.nunique() > 1).any():
        raise ValueError("存在同行组跨 fold")
    return fold_of.to_numpy(dtype=int)


def make_inputs(clean: pd.DataFrame, model: str, columns=None) -> pd.DataFrame:
    """相同基础特征：LogReg 使用 one-hot，CatBoost 使用原生类别列。"""
    if model == "logreg":
        X = pp.make_model_matrix(clean)
        if columns is not None:
            X = X.reindex(columns=columns, fill_value=0)
    elif model == "catboost":
        indicators = [f"{c}_na" for c in pp.MISSING_INDICATOR_COLS if f"{c}_na" in clean]
        numeric = list(pp.BOOL_FEATURES) + list(pp.NUM_FEATURES) + indicators
        X = clean[numeric].apply(pd.to_numeric, errors="raise").astype(float)
        for col in pp.CAT_FEATURES:
            X[col] = clean[col].astype("object").fillna("__missing__").map(str).astype(object)
        if columns is not None:
            X = X.loc[:, columns]
    else:
        raise ValueError(f"未知模型：{model}")
    if X.isna().any().any() or not np.isfinite(X.select_dtypes("number").to_numpy()).all():
        raise ValueError("模型输入含缺失或无穷值")
    return X


def make_model(config: TrainingConfig):
    if config.model == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=config.c, max_iter=3000, random_state=config.seed),
        )
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("请先在项目环境安装 requirements.txt 中的 CatBoost") from exc
    # 固定迭代数，不用外层验证折选择 best_iteration，保持首次五折比较口径清晰。
    return CatBoostClassifier(
        iterations=config.iterations, depth=config.depth,
        learning_rate=config.learning_rate, l2_leaf_reg=config.l2_leaf_reg,
        loss_function="Logloss", random_seed=config.seed, thread_count=config.threads,
        cat_features=list(pp.CAT_FEATURES), allow_writing_files=False,
        verbose=False, task_type="CPU",
    )


def positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("模型必须包含标签 1 的预测概率")
    result = np.asarray(model.predict_proba(X), dtype=float)[:, positive[0]]
    if len(result) != len(X) or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("预测概率不合法")
    return result


def make_submission(test_ids, probability, template: pd.DataFrame, threshold=0.5):
    ids = list(test_ids)
    probability = np.asarray(probability, dtype=float)
    if list(template.columns) != ["PassengerId", "Transported"]:
        raise ValueError("提交模板列不符合预期")
    if ids != template.PassengerId.tolist() or len(set(ids)) != len(ids):
        raise ValueError("测试乘客顺序或数量与提交模板不一致")
    if probability.ndim != 1 or len(probability) != len(ids):
        raise ValueError("预测概率行数不匹配")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("提交概率不合法")
    return pd.DataFrame({"PassengerId": ids, "Transported": probability >= threshold})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_hashes(directory: Path):
    return {name: _sha256(directory / name) for name in [
        "train.csv", "test.csv", "sample_submission.csv",
    ]}


def _write_json(path: Path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def _git_head():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_training(config: TrainingConfig, run_name: str, *, data_dir=None,
                 folds_path=FEATURES_DIR / "folds.csv", fit_full=False,
                 report_root=EXPERIMENTS_DIR / "training",
                 output_root=PROJECT_ROOT / "outputs", model_root=PROJECT_ROOT / "models"):
    """执行一次有独立目录的实验。fit_full=True 额外保存模型及可提交文件。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run_name 使用字母、数字、下划线或短横线，长度不超过80")
    directory = pp.find_data_dir(data_dir)
    raw, raw_test = pp.load_raw(directory)
    features, test_features = pp.build_features_combined(raw, raw_test)
    folds_path = Path(folds_path).resolve()
    fold_map = pd.read_csv(folds_path, dtype={"PassengerId": str, "GroupId": str})
    fold_of = validate_fold_map(features, fold_map)
    report_dir, output_dir = Path(report_root) / run_name, Path(output_root) / run_name
    model_dir = Path(model_root) / run_name
    destinations = [report_dir] + ([output_dir, model_dir] if fit_full else [])
    if any(p.exists() for p in destinations):
        raise FileExistsError("实验目录已存在，请使用新的 --run-name，避免覆盖旧结果")
    make_model(config)  # 在创建输出前检查模型依赖与配置。
    for p in destinations:
        p.mkdir(parents=True, exist_ok=False)

    started_at, start = datetime.now(SHANGHAI).isoformat(), time.perf_counter()
    target = features.Transported.astype(int).to_numpy()
    oof = np.full(len(features), np.nan)
    coverage = np.zeros(len(features), dtype=int)
    metrics = []
    for fold in range(1, 6):
        ti, vi = np.flatnonzero(fold_of != fold), np.flatnonzero(fold_of == fold)
        train, valid = features.iloc[ti], features.iloc[vi]
        prep = pp.Preprocessor(config.strategy).fit(train)
        Xtr = make_inputs(prep.transform(train), config.model)
        Xva = make_inputs(prep.transform(valid), config.model, Xtr.columns)
        model = make_model(config)
        fold_start = time.perf_counter()
        model.fit(Xtr, target[ti])
        probability = positive_probability(model, Xva)
        train_probability = positive_probability(model, Xtr)
        oof[vi], coverage[vi] = probability, coverage[vi] + 1
        record = {
            "fold": fold, "n_train": len(ti), "n_valid": len(vi), "n_features": Xtr.shape[1],
            "train_accuracy": float(accuracy_score(target[ti], train_probability >= config.threshold)),
            "accuracy": float(accuracy_score(target[vi], probability >= config.threshold)),
            "log_loss": float(log_loss(target[vi], probability, labels=[0, 1])),
            "seconds": time.perf_counter() - fold_start,
        }
        metrics.append(record)
        print(f"{run_name} fold {fold}/5: accuracy={record['accuracy']:.4f}, "
              f"train={record['train_accuracy']:.4f}, features={Xtr.shape[1]}", flush=True)
    if not (coverage == 1).all() or not np.isfinite(oof).all():
        raise RuntimeError("每位训练乘客必须恰好获得一次折外预测")
    fold_metrics = pd.DataFrame(metrics)
    oof_frame = pd.DataFrame({
        "PassengerId": features.PassengerId.to_numpy(), "GroupId": features.GroupId.to_numpy(),
        "fold": fold_of, "Transported": target.astype(bool),
        "probability": oof, "prediction": oof >= config.threshold,
    })
    # 折外预测是团队共享的实验交付，放在 reports 下；测试预测与模型仍留在忽略目录。
    _write_csv(oof_frame, report_dir / "oof_predictions.csv")
    _write_csv(fold_metrics, report_dir / "fold_metrics.csv")

    packages = ["numpy", "pandas", "scikit-learn", "joblib"]
    if config.model == "catboost":
        packages.append("catboost")
    summary = {
        "run_name": run_name, "started_at": started_at, "config": asdict(config),
        "encoding": "native_categorical" if config.model == "catboost" else "one_hot",
        "accuracy_mean": float(fold_metrics.accuracy.mean()),
        "accuracy_std": float(fold_metrics.accuracy.std(ddof=1)),
        "oof_accuracy": float(accuracy_score(target, oof >= config.threshold)),
        "oof_log_loss": float(log_loss(target, oof, labels=[0, 1])),
        "n_train": len(features), "n_test": len(test_features), "fit_full": fit_full,
        "data_hashes": _data_hashes(directory), "folds_sha256": _sha256(folds_path),
        "source_hashes": {name: _sha256(PROJECT_ROOT / name) for name in ["src/train.py", "src/preprocess.py"]},
        "versions": {p: version(p) for p in packages}, "git_head": _git_head(),
        "feature_policy": "train+test unlabeled cohort counts; fold-local imputation/scaling; threshold fixed",
        "oof_path": str((report_dir / "oof_predictions.csv").resolve()),
        "output_dir": str(output_dir.resolve()) if fit_full else None,
    }

    if fit_full:
        prep = pp.Preprocessor(config.strategy).fit(features)
        X = make_inputs(prep.transform(features), config.model)
        Xt = make_inputs(prep.transform(test_features), config.model, X.columns)
        model = make_model(config)
        model.fit(X, target)
        probability = positive_probability(model, Xt)
        template = pp.load_submission_template(directory)
        submission = make_submission(raw_test.PassengerId, probability, template, config.threshold)
        _write_csv(pd.DataFrame({"PassengerId": raw_test.PassengerId, "probability": probability,
                                "prediction": probability >= config.threshold}),
                   output_dir / "test_predictions.csv")
        _write_csv(submission, output_dir / "submission.csv")
        bundle_path = model_dir / "bundle.joblib"
        joblib.dump({"model": model, "preprocessor": prep, "columns": list(X.columns),
                     "config": asdict(config), "data_hashes": summary["data_hashes"],
                     "run_name": run_name}, bundle_path)
        summary["model_bundle"] = str(bundle_path.resolve())
        summary["full_n_features"] = X.shape[1]
        if config.model == "catboost":
            importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
            _write_csv(importance.sort_values("importance", ascending=False), report_dir / "feature_importance.csv")

    summary["seconds"] = time.perf_counter() - start
    summary["status"] = "completed"
    _write_json(report_dir / "summary.json", summary)
    print(f"Completed: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}; "
          f"OOF={summary['oof_accuracy']:.4f}; report={report_dir}", flush=True)
    return summary


def predict_saved_model(bundle_path, *, data_dir=None):
    """在相同公开特征池上复现最终模型预测，拒绝悄悄切换计数口径。"""
    bundle = joblib.load(bundle_path)
    directory = pp.find_data_dir(data_dir)
    if _data_hashes(directory) != bundle["data_hashes"]:
        raise ValueError("数据与模型训练时的公开特征池不一致，需重新训练并明确计数口径")
    raw, raw_test = pp.load_raw(directory)
    _, test_features = pp.build_features_combined(raw, raw_test)
    config = TrainingConfig(**bundle["config"])
    X = make_inputs(bundle["preprocessor"].transform(test_features), config.model, bundle["columns"])
    probability = positive_probability(bundle["model"], X)
    predictions = pd.DataFrame({"PassengerId": raw_test.PassengerId, "probability": probability,
                                "prediction": probability >= config.threshold})
    submission = make_submission(raw_test.PassengerId, probability,
                                 pp.load_submission_template(directory), config.threshold)
    return predictions, submission
