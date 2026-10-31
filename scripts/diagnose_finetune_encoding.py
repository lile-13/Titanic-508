"""Input-only control reconstruction after the failed fine-tuning screen."""
import gc
import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch
from tabicl._sklearn.preprocessing import TransformToNumerical

from src.optimization import SearchRunner
from src.paths import PROJECT_ROOT as P
from src.tabicl_models import make_model, CHECKPOINT
from src.tabicl_finetune_probe import predictor
from src.train import _write_json


def main():
    old=SearchRunner()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    rows=[]
    for fold in range(1,6):
        stem=f"fresh_repeat_1_fold_{fold}"
        original=joblib.load(P/f"models/tabicl_probability_mean_20261030/{stem}.joblib")
        cache=np.load(P/f"outputs/tabicl_probability_mean_20261030/{stem}.npz")
        query=original.builder.transform(old.features.iloc[cache["validation"]])
        outer=TransformToNumerical().fit(original.training_X)
        context=joblib.load(P/f"models/tabicl_finetune_20261031/paired_repeat_1_fold_{fold}/context.joblib")
        a,b=outer.transform(original.training_X),context["encoded_X"]
        changed=[dict(column=i,changed_rows=int(np.count_nonzero(a[:,i]!=b[:,i])),
                      max_difference=float(np.max(np.abs(a[:,i]-b[:,i]))))
                 for i in range(a.shape[1]) if not np.array_equal(a[:,i],b[:,i])]
        direct=make_model().set_params(average_logits=False).fit(original.training_X,original.training_y)
        p=direct.predict_proba(query)[:,1]
        encoded=predictor(P/"models/tabicl_pretrained"/CHECKPOINT,dict(encoded_X=a,y=original.training_y))
        q=encoded.predict_proba(outer.transform(query))[:,1]
        np.testing.assert_allclose(p,cache["probability"],rtol=0,atol=2e-6)
        np.testing.assert_allclose(q,p,rtol=0,atol=2e-6)
        rows.append(dict(fold=fold,encoded_training_differences=changed,
            direct_vs_saved_max_error=float(np.abs(p-cache["probability"]).max()),
            full_outer_encoding_vs_direct_max_error=float(np.abs(q-p).max())))
        print(f"Encoding control fold {fold} reproduced",flush=True)
        del original,context,direct,encoded
        gc.collect()
        torch.cuda.empty_cache()
    _write_json(P/"reports/experiments/tabicl_finetune_20261031/encoding_diagnostic.json",dict(
        passed=True,folds=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat="Post-screen input-only control reconstruction; no target values used for scoring or model selection. No fine-tuned predictions with changed encoding were evaluated. This does not establish accuracy improvement."))


if __name__=="__main__":
    main()
