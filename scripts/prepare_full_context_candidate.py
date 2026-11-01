"""A separate full-training-context strategy comparison, with no invented CV score."""
import gc

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.audit_unlabeled_views import sha,read
from src import preprocess as pp
from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P,EXPERIMENTS_DIR
from src.tabicl_aggregation import fit_probability_mean
from src.train import _write_json,_write_csv,make_submission

NAME="full_context_strategy_20261101"


def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    old=SearchRunner()
    report,out,models=EXPERIMENTS_DIR/NAME,P/"outputs"/NAME,P/"models"/NAME
    for d in [report,out,models]:d.mkdir(parents=True,exist_ok=True)
    source=P/"outputs/manual_submission_batch_20261031/test_components.csv"
    best=read(EXPERIMENTS_DIR/"kaggle_submissions.json")["current_platform_best"]
    assert best["score"]==.81318
    protocol=dict(name=NAME,observed_platform_best=best,threshold=.5,
        method="Keep tree25%; use fold-context TabICL37.5% and full-training-context TabICL37.5%. Same original checkpoint, seed42, eight views, arithmetic averaging. No neural parameter updates.",
        hypothesis="Each fold context sees about80% of labels. A single context using all8693 training passengers may add information; average it with the existing five-context estimate to retain some context diversity.",
        selection="Fixed half replacement of the incumbent's75% TFM component. Requested exploratory submission; no training-size/weight/threshold search.",
        validation="No new OOF accuracy exists for this final training-size comparison. Cannot score an all-training-context model on its training rows or borrow the old fold OOF score as its own.",
        source_components_sha256=sha(source),
        sources={f:sha(P/f) for f in ["scripts/prepare_full_context_candidate.py","src/tabicl_aggregation.py","src/tabicl_models.py","src/focused_features.py","src/competition_features.py","src/preprocess.py"]},
        data_hashes=read(old.report/"protocol.json")["data_hashes"])
    if (report/"protocol.json").exists():assert read(report/"protocol.json")==protocol
    else:_write_json(report/"protocol.json",protocol)
    mp=models/"full_context.joblib"
    cache=out/"full_context_test.npz"
    if (report/"model.json").exists():
        assert sha(mp)==read(report/"model.json")["model_sha256"]
        probability=np.load(cache)["probability"]
    else:
        public=pd.concat([old.features.drop(columns=pp.TARGET),old.test_features],ignore_index=True)
        model=fit_probability_mean(old.features,old.y,public)
        assert len(model.training_y)==8693 and pp.TARGET not in model.training_X
        probability=model.predict_proba(old.test_features)[:,1]
        joblib.dump(model,mp,compress=3)
        np.savez_compressed(cache,probability=probability)
        _write_json(report/"model.json",dict(model_sha256=sha(mp),training_rows=8693,views=8,seed=42))
        del model;gc.collect();torch.cuda.empty_cache()
    components=pd.read_csv(source,dtype={"PassengerId":str},float_precision="round_trip")
    np.testing.assert_array_equal(components.PassengerId,old.raw_test.PassengerId)
    q=.25*components.tree.to_numpy()+.375*components.tfm.to_numpy()+.375*probability
    ready=P/"outputs/kaggle_ready_4690/next_batch_20261101"
    ready.mkdir(parents=True,exist_ok=True)
    path=ready/"submission_fullcontext_4690.csv"
    frame=make_submission(old.raw_test.PassengerId,q,pp.load_submission_template(),.5)
    if path.exists():pd.testing.assert_frame_equal(frame,pd.read_csv(path,dtype={"PassengerId":str}))
    else:_write_csv(frame,path)
    original=pd.read_csv(P/best["submission"],dtype={"PassengerId":str})
    candidate=dict(id=NAME,model_run=NAME,status="prepared_not_submitted",prepared_on="2026-11-01",
        file=path.relative_to(P).as_posix(),sha256=sha(path),rows=4277,score=None,kaggle_score=None,
        description="25% trees + 37.5% five-fold TabICL + 37.5% full-training-context TabICL; threshold0.5",
        changes_vs_current_best=int((frame.Transported!=original.Transported).sum()),suggested_order=2,
        exploratory=True,formal_promotion=False,local_cv_accuracy=None,
        source_protocol_sha256=sha(report/"protocol.json"),audit=f"reports/checks/{NAME}.json")
    assert candidate["changes_vs_current_best"]>0
    _write_json(report/"summary.json",dict(status="exploratory_candidate_prepared",candidate=candidate,
        caveat="A final context-size comparison, with no leakage-free OOF result for this exact training-size change. Actual Kaggle result unknown."))
    _write_json(ready/"full_context_manifest.json",candidate)
    print(candidate)


if __name__=="__main__":main()
