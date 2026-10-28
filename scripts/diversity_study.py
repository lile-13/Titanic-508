"""有限的异构模型实验；开发期排除第5折，冻结后才重训五折。"""
import hashlib
import json
import os
import time

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "4")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

from src.competition_features import FeatureBuilder, InputEncoder
from src.diversity_features import DiversityEncoder
from src.optimization import SearchRunner, config_id, predict_ensemble
from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR
from src.train import _write_json, _write_csv, make_submission
from src import preprocess as pp

NAME = "diversity_study_20261028"
PROFILES = [
    dict(family="svm", features="base", C=1., gamma=.02),
    dict(family="svm", features="base", C=4., gamma=.02),
    dict(family="svm", features="base", C=12., gamma=.02),
    dict(family="svm", features="base", C=4., gamma=.06),
    dict(family="svm", features="spend", C=4., gamma=.02),
    dict(family="svm", features="spend", C=12., gamma=.02),
    dict(family="extra", features="spend", min_samples_leaf=3),
    dict(family="extra", features="spend", min_samples_leaf=8),
    dict(family="hist", features="spend", max_leaf_nodes=15, l2_regularization=5.),
    dict(family="hist", features="spend", max_leaf_nodes=31, l2_regularization=10.),
    dict(family="ordered", features="spend", depth=5, l2_leaf_reg=6.),
    dict(family="ordered", features="rich_names", depth=6, l2_leaf_reg=12.),
    dict(family="lossguide", features="spend", max_leaves=15, l2_leaf_reg=6.),
    dict(family="lossguide", features="rich_names", max_leaves=31, l2_leaf_reg=12.),
]


def fit_member(profile, seed, features, y, groups):
    native = profile["family"] in ("ordered", "lossguide")
    builder = FeatureBuilder(profile["features"], "native" if native else "rule_group").fit(features)
    X = builder.transform(features)
    encoder = (InputEncoder("catboost") if native else DiversityEncoder()).fit(X)
    X = encoder.transform(X)
    params = {k: v for k, v in profile.items() if k not in ("family", "features")}
    family = profile["family"]
    if family == "svm":
        folds = list(GroupKFold(3, shuffle=True, random_state=seed).split(X, y, groups))
        model = CalibratedClassifierCV(SVC(kernel="rbf", cache_size=512, **params),
                                      method="sigmoid", cv=folds, n_jobs=1, ensemble=False)
    elif family == "extra":
        model = ExtraTreesClassifier(n_estimators=600, max_features=.8, n_jobs=4,
                                     random_state=seed, **params)
    elif family == "hist":
        model = HistGradientBoostingClassifier(max_iter=450, learning_rate=.045,
                 min_samples_leaf=30, early_stopping=False, random_state=seed, **params)
    else:
        special = dict(boosting_type="Ordered", max_ctr_complexity=2) if family == "ordered" else dict(
            grow_policy="Lossguide", depth=8, min_data_in_leaf=20, max_ctr_complexity=1)
        model = CatBoostClassifier(iterations=800, learning_rate=.035, random_seed=seed,
                    thread_count=4, loss_function="Logloss", cat_features=encoder.cat_columns_,
                    verbose=False, allow_writing_files=False, **special, **params)
    model.fit(X, y)
    return dict(builder=builder, encoder=encoder, model=model)


def probability(member, features):
    return member["model"].predict_proba(member["encoder"].transform(member["builder"].transform(features)))[:, 1]


def main():
    old = SearchRunner(threads=4)
    report = EXPERIMENTS_DIR / NAME
    output = PROJECT_ROOT / "outputs" / NAME
    report.mkdir(exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    protocol = dict(name=NAME, profiles=PROFILES, development_folds=[1,2,3,4], excluded_fold=5,
        seeds=[42,2026,103], mixture_weights=[.25,.5,.75,1.], threshold=.5,
        selection="Top two configurations repeated with three seeds; require accuracy gain >= .0015 and logloss deterioration <= .002 over frozen incumbent",
        fixed_model_parameters=dict(catboost_iterations=800, catboost_lr=.035, extra_trees=600,
                                    hist_iterations=450, hist_lr=.045, svm_calibration="3-fold GroupKFold, sigmoid"),
        caveat="Additional development on historically inspected folds, not a fresh blind holdout",
        sources={p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in
                 ["scripts/diversity_study.py","src/diversity_features.py","src/competition_features.py"]},
        data_hashes=json.loads((old.report/"protocol.json").read_text(encoding="utf-8"))["data_hashes"])
    path=report/"protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=protocol:raise ValueError("源码或协议改变，不允许混用缓存")
    else:_write_json(path,protocol)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    dev=old.fold_of!=5;incumbent=np.zeros(len(old.y))
    for weight, member in zip(plan["weights"],plan["members"]):
        for seed in member["seeds"]:
            trial=config_id(dict(member["config"],seed=seed),"confirmation",[1,2,3,4])
            pred=np.load(old.output/"search"/f"{trial}.npz")["probability"]
            assert np.isnan(pred[~dev]).all()
            incumbent[dev]+=weight/len(member["seeds"])*pred[dev]
    base_acc=float(accuracy_score(old.y[dev],incumbent[dev]>=.5))
    base_loss=float(log_loss(old.y[dev],incumbent[dev]))
    np.testing.assert_allclose(base_acc,plan["development_accuracy"],atol=1e-14)

    def evaluate(index,seed):
        cache=output/f"profile_{index}_seed_{seed}.npz";record=report/f"profile_{index}_seed_{seed}.json"
        if cache.exists() and record.exists():return np.load(cache)["probability"]
        started=time.perf_counter();pred=np.full(len(old.y),np.nan);scores=[]
        for fold,(ti,vi) in enumerate(old.splits,1):
            member=fit_member(PROFILES[index],seed,old.features.iloc[ti],old.y[ti],old.features.GroupId.iloc[ti])
            pred[vi]=probability(member,old.features.iloc[vi]);scores.append(float(accuracy_score(old.y[vi],pred[vi]>=.5)))
            print(f"Diversity {index} {PROFILES[index]['family']} seed {seed} fold {fold}: {scores[-1]:.4f}",flush=True)
        assert np.isnan(pred[~dev]).all()
        np.savez_compressed(cache,probability=pred)
        _write_json(record,dict(profile=index,config=PROFILES[index],seed=seed,fold_scores=scores,
             accuracy=float(accuracy_score(old.y[dev],pred[dev]>=.5)),log_loss=float(log_loss(old.y[dev],pred[dev])),
             error_correlation=float(np.corrcoef((pred[dev]>=.5)!=old.y[dev],(incumbent[dev]>=.5)!=old.y[dev])[0,1]),
             seconds=time.perf_counter()-started))
        return pred

    def choices(predictions):
        rows=[]
        for index,pred in predictions.items():
            for weight in protocol["mixture_weights"]:
                mixed=(1-weight)*incumbent[dev]+weight*pred[dev]
                rows.append(dict(profile=index,new_weight=weight,accuracy=float(accuracy_score(old.y[dev],mixed>=.5)),
                                 log_loss=float(log_loss(old.y[dev],mixed))))
        return sorted(rows,key=lambda r:(-r["accuracy"],r["log_loss"],r["new_weight"]))

    first={i:evaluate(i,42) for i in range(len(PROFILES))}
    initial=choices(first);top=[]
    for row in initial:
        if row["profile"] not in top:top.append(row["profile"])
        if len(top)==2:break
    seeded={i:np.mean([first[i],evaluate(i,2026),evaluate(i,103)],axis=0) for i in top}
    scans=choices(seeded);winner=scans[0]
    _write_csv(pd.DataFrame(initial),report/"single_seed_mixtures.csv")
    _write_csv(pd.DataFrame(scans),report/"three_seed_mixtures.csv")
    passed=winner["accuracy"]>=base_acc+.0015 and winner["log_loss"]<=base_loss+.002
    frozen=dict(profile=winner["profile"],config=PROFILES[winner["profile"]],weight=winner["new_weight"],
                threshold=.5,seeds=protocol["seeds"],development=winner,incumbent_accuracy=base_acc,
                incumbent_log_loss=base_loss,generate_candidate=bool(passed))
    _write_json(report/"frozen_choice.json",frozen)
    if not passed:
        _write_json(report/"summary.json",dict(status="no_candidate",frozen=frozen,kaggle_score=None,
                   reason="Predeclared gate not met; preserve observed Kaggle best 0.81108"))
        print("No new candidate; incumbent retained",flush=True);return
    w=frozen["weight"]
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=[dict(m,weight=m["weight"]*(1-w)) for m in bundle["members"] if w<1]
    canonical=pd.read_csv(old.report/"oof_predictions.csv",dtype={"PassengerId":str})
    old_test=pd.read_csv(old.output/"test_predictions.csv",dtype={"PassengerId":str})
    np.testing.assert_array_equal(canonical.PassengerId,old.raw.PassengerId)
    np.testing.assert_array_equal(old_test.PassengerId,old.raw_test.PassengerId)
    oof=(1-w)*canonical.probability.to_numpy();tp=(1-w)*old_test.probability.to_numpy()
    for fold in range(1,6):
        ti=np.flatnonzero(old.fold_of!=fold);vi=np.flatnonzero(old.fold_of==fold)
        for seed in frozen["seeds"]:
            member=fit_member(frozen["config"],seed,old.features.iloc[ti],old.y[ti],old.features.GroupId.iloc[ti])
            oof[vi]+=w/3*probability(member,old.features.iloc[vi]);tp+=w/15*probability(member,old.test_features)
            members.append(dict(member,weight=w/15))
        print(f"Diversity frozen refit {fold}/5",flush=True)
    modeldir=PROJECT_ROOT/"models"/NAME;modeldir.mkdir(exist_ok=True)
    modelpath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(threshold=.5,name=NAME,frozen_choice=frozen),data_hashes=bundle["data_hashes"]),modelpath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,tp,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_diversity4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=tp,prediction=tp>=.5)),output/"test_predictions.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw.PassengerId,fold=old.fold_of,Transported=old.y,probability=oof,prediction=oof>=.5)),report/"oof_predictions.csv")
    reloaded,check=predict_ensemble(modelpath)
    np.testing.assert_allclose(reloaded,tp,atol=1e-12,rtol=0)
    pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="candidate_generated",frozen=frozen,
        canonical_oof=float(accuracy_score(old.y,oof>=.5)),canonical_log_loss=float(log_loss(old.y,oof)),
        changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=old_test.prediction.to_numpy())),
        models=len(members),reload_max_error=float(np.max(np.abs(reloaded-tp))),kaggle_score=None,
        interpretation="Post-selection OOF is descriptive; a Kaggle improvement is not established until user submits"))


if __name__=="__main__":main()
