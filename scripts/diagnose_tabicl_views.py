"""Inspect existing per-view logits without changing any deployed aggregation."""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import MethodType

import joblib
import numpy as np
import torch

from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_json,_write_csv
from src.tabicl_models import make_model
import pandas as pd


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=SearchRunner();report=EXPERIMENTS_DIR/"tabicl_failure_diagnosis_20261030"
    output=PROJECT_ROOT/"outputs/tabicl_failure_diagnosis_20261030"
    z=np.load(PROJECT_ROOT/"outputs/fixed_weight_confirmation_20261029/repeat_2_fold_3.npz");vi=z["validation"]
    frame=old.features.iloc[vi];cold=frame.Deck.eq("G").fillna(False).to_numpy(bool)&frame.CryoSleep.eq(True).fillna(False).to_numpy(bool)
    sources={"full_context":PROJECT_ROOT/"models/tabicl_paired_20261030/repeat_2_fold_3.joblib",
             "half_seed_5101":PROJECT_ROOT/"models/tabicl_failure_diagnosis_20261030/seed_5101_fraction_0.5.joblib"}
    all_rows=[];summaries=[]
    for name,path in sources.items():
        member=joblib.load(path);member._model=make_model().fit(member.training_X,member.training_y)
        clf=member._model;captured=[];forward=clf._batch_forward
        def recording(self,Xs,ys,feature_shuffles=None):
            output=forward(Xs,ys,feature_shuffles);captured.append(output.copy());return output
        clf._batch_forward=MethodType(recording,clf)
        p=member.predict_proba(frame)[:,1]
        clf._batch_forward=forward
        cache=PROJECT_ROOT/"outputs"/path.parent.name/(path.stem+".npz")
        historical=np.load(cache)
        np.testing.assert_array_equal(historical["validation"],vi)
        np.testing.assert_allclose(p,historical["probability"],atol=2e-6,rtol=0)
        outputs=np.concatenate(captured,axis=0);aligned=[];normalizers=[]
        for norm,shuffles in clf.ensemble_generator_.class_shuffles_.items():
            for shuffle in shuffles:
                aligned.append(outputs[len(aligned)][...,shuffle]);normalizers.append(norm)
        logits=np.stack(aligned)
        reproduced=clf.softmax(logits.mean(axis=0),axis=-1,temperature=clf.softmax_temperature)[:,1]
        np.testing.assert_allclose(reproduced,p,atol=1e-6,rtol=0)
        probabilities=clf.softmax(logits,axis=-1,temperature=clf.softmax_temperature)[:,:,1]
        np.savez_compressed(output/f"views_{name}.npz",validation=vi,logits=logits,
                            probabilities=probabilities,logit_mean_probability=p)
        for i,q in enumerate(probabilities):
            all_rows.append(dict(context=name,view=i,normalizer=str(normalizers[i]),
                                 cold_G_rows=int(cold.sum()),cold_G_mean_probability=float(q[cold].mean()),
                                 cold_G_accuracy=float(((q[cold]>=.5)==old.y[vi][cold]).mean()),
                                 cold_G_positive_prediction_share=float((q[cold]>=.5).mean())))
        source=Path(importlib.metadata.distribution("tabicl").locate_file("tabicl/_sklearn/classifier.py"))
        summaries.append(dict(context=name,model_sha256=sha(path),views=len(aligned),
                              cold_G_mean_probability=float(p[cold].mean()),cold_G_mean_view_std=float(probabilities[:,cold].std(axis=0).mean()),
                              cold_G_observed_rate=float(old.y[vi][cold].mean()),
                              arithmetic_probability_mean_cold_G=float(probabilities[:,cold].mean()),
                              aggregation_reproduced=True,historical_prediction_reproduced=True,installed_classifier_sha256=sha(source),
                              caveat="Existing view inspection on a selected failure fold; arithmetic mean shown descriptively only, not evaluated/promoted as a new submission"))
        del member,clf;gc.collect();torch.cuda.empty_cache()
    _write_csv(pd.DataFrame(all_rows),report/"per_view_cold_G.csv")
    _write_json(report/"per_view_summary.json",summaries)
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":main()
