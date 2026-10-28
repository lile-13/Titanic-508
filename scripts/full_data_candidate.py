"""固定参数的训练样本量对照：全量训练与原五折集成等权融合。"""
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.competition_features import FeatureBuilder, InputEncoder
from src.optimization import SearchRunner, estimator, predict_ensemble
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv,make_submission
from src import preprocess as pp

NAME="full_data_candidate_20261028"


def main():
    old=SearchRunner(threads=4)
    report=EXPERIMENTS_DIR/NAME;output=PROJECT_ROOT/"outputs"/NAME;modeldir=PROJECT_ROOT/"models"/NAME
    for path in [report,output,modeldir]:
        if path.exists():raise FileExistsError(f"实验目录已存在，不覆盖：{path}")
    for path in [report,output,modeldir]:path.mkdir(parents=True)
    protocol=dict(name=NAME,source_experiment=old.name,full_data_weight=.5,fold_bag_weight=.5,
        threshold=.5,selection="No parameter, weight or threshold search; an unvalidated training-strategy comparison candidate",
        caveat="Full-data predictions have no out-of-fold validation. Historical OOF belongs to the fold-bag incumbent only. This is not a measured score improvement.",
        source_hash=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        frozen_plan_hash=hashlib.sha256((old.report/"frozen_plan.json").read_bytes()).hexdigest())
    _write_json(report/"protocol.json",protocol)
    plan=json.loads((old.report/"frozen_plan.json").read_text(encoding="utf-8"))
    incumbent=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    members=[dict(m,weight=m["weight"]*.5) for m in incumbent["members"]]
    original=pd.read_csv(old.output/"test_predictions.csv",dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(original.PassengerId,old.raw_test.PassengerId)
    full=np.zeros(len(old.raw_test));started=time.perf_counter()
    for weight,choice in zip(plan["weights"],plan["members"]):
        config=choice["config"]
        builder=FeatureBuilder(config["feature_set"],config["strategy"]).fit(old.features)
        X=builder.transform(old.features);T=builder.transform(old.test_features)
        encoder=InputEncoder(config["family"]).fit(X);X,T=encoder.transform(X),encoder.transform(T)
        for seed in choice["seeds"]:
            model=estimator(dict(config,seed=seed),encoder.cat_columns_,threads=4)
            model.fit(X,old.y)
            member_weight=weight/len(choice["seeds"])
            full+=member_weight*model.predict_proba(T)[:,1]
            members.append(dict(weight=.5*member_weight,builder=builder,encoder=encoder,model=model))
            print(f"Full-data {config['family']} {config['feature_set']} seed {seed} completed",flush=True)
    mixed=.5*full+.5*original.probability.to_numpy()
    bundlepath=modeldir/"ensemble.joblib"
    joblib.dump(dict(members=members,plan=dict(name=NAME,threshold=.5,protocol=protocol),
                     data_hashes=incumbent["data_hashes"]),bundlepath,compress=3)
    submission=make_submission(old.raw_test.PassengerId,mixed,pp.load_submission_template(),.5)
    _write_csv(submission,output/"submission_fullblend4690.csv")
    _write_csv(pd.DataFrame(dict(PassengerId=old.raw_test.PassengerId,probability=mixed,
                    full_probability=full,bag_probability=original.probability,prediction=mixed>=.5)),output/"test_predictions.csv")
    reloaded,check=predict_ensemble(bundlepath)
    np.testing.assert_allclose(reloaded,mixed,atol=1e-12,rtol=0)
    pd.testing.assert_frame_equal(check,submission)
    _write_json(report/"summary.json",dict(status="training_strategy_comparison_only",kaggle_score=None,
        canonical_oof=None,changed_test_labels=int(np.sum(submission.Transported.to_numpy()!=original.prediction.to_numpy())),
        pure_full_changed_labels=int(np.sum((full>=.5)!=original.prediction.to_numpy())),
        rows=len(submission),members=len(members),new_full_models=9,threshold=.5,seconds=time.perf_counter()-started,
        probability_max_reload_error=float(np.max(np.abs(reloaded-mixed))),
        submission_sha256=hashlib.sha256((output/"submission_fullblend4690.csv").read_bytes()).hexdigest(),
        recommendation="Optional single controlled Kaggle comparison. Preserve observed 0.81108 incumbent until a higher platform score is observed."))
    print("Full-data comparison file generated; Kaggle score remains unknown",flush=True)


if __name__=="__main__":main()
