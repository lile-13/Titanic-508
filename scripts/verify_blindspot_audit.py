"""Independently recompute diagnostic counts and archive the exploratory parity check."""
import hashlib
import json

import numpy as np
import pandas as pd

from scripts.audit_model_blindspots import descriptors
from src import preprocess as pp
from src.competition_features import build_pool
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_csv,_write_json


def main():
    report=EXPERIMENTS_DIR/"error_audit_20261029"
    raw,test=pp.load_raw();d,t=build_pool(raw,test)
    public=pd.concat([d.drop(columns=pp.TARGET),t],ignore_index=True)
    desc=descriptors(public);a=pd.read_csv(report/"passenger_diagnostics.csv",dtype={"PassengerId":str,"GroupId":str},float_precision="round_trip")
    np.testing.assert_array_equal(a.PassengerId,raw.PassengerId)
    y=raw.Transported.to_numpy(int);p=a.probability.to_numpy();wrong=(p>=.5)!=y
    np.testing.assert_array_equal(wrong,a.error)
    assert int(wrong.sum())==1602
    slices=pd.read_csv(report/"slice_metrics.csv",float_precision="round_trip")
    keys={}
    for row in slices.itertuples():
        if row.columns not in keys:
            keys[row.columns]=desc[row.columns.split(" + ")].astype("string").fillna("Missing").agg(" | ".join,axis=1)
        key=keys[row.columns];mask=key.iloc[:len(d)].eq(row.value).to_numpy()
        assert int(mask.sum())==row.n and int(wrong[mask].sum())==row.errors
        assert int(key.iloc[len(d):].eq(row.value).sum())==row.test_n
        np.testing.assert_allclose(wrong[mask].mean(),row.error_rate,atol=1e-14)
    errors=pd.read_csv(report/"error_cases.csv",dtype={"PassengerId":str})
    assert set(errors.PassengerId)==set(raw.PassengerId[wrong])
    ps=[p]
    for repeat in [1,2]:
        source=pd.read_csv(EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029"/f"repeat_{repeat}_oof.csv",dtype={"PassengerId":str},float_precision="round_trip")
        np.testing.assert_array_equal(source.PassengerId,raw.PassengerId);ps.append(source.reference.to_numpy())
    probability=np.column_stack(ps);persistent=((probability>=.5)!=y[:,None]).all(axis=1)
    hard=np.where(y[:,None]==1,probability,1-probability).max(axis=1)<=.1
    assert persistent.sum()==1364 and hard.sum()==74
    saved=pd.read_csv(report/"confident_persistent_errors.csv",dtype={"PassengerId":str})
    assert set(saved.PassengerId)==set(raw.PassengerId[hard])
    cohort=json.loads((report/"mechanisms/cohort_metrics.json").read_text(encoding="utf-8"))
    assert cohort[1]["n"]==2498 and cohort[1]["errors"]==719
    assert cohort[3]["n"]==258 and cohort[3]["errors"]==99
    # Excluding self makes these aggregations contextual rather than personal aliases.
    sample=np.random.default_rng(20261029).choice(len(public),256,replace=False)
    members=public.groupby("GroupId",observed=True).indices
    known=public.CryoSleep.notna().astype(int);true=public.CryoSleep.eq(True).fillna(False).astype(int)
    count=known.groupby(public.GroupId).transform("sum")-known
    total=true.groupby(public.GroupId).transform("sum")-true
    share=total/count.replace(0,np.nan)
    adult=public.Age.ge(18).astype(int);age=public.Age.where(adult.eq(1),0.).fillna(0.)
    age_total=age.groupby(public.GroupId).transform("sum")-age
    adult_count=adult.groupby(public.GroupId).transform("sum")-adult
    mean_age=age_total/adult_count.replace(0,np.nan)
    child=public.Age.le(12).astype(int)
    other_children=child.groupby(public.GroupId).transform("sum")-child
    for i in sample:
        group_indices=members[public.GroupId.iloc[i]]
        other=public.iloc[group_indices[group_indices!=i]]
        expected_share=other.CryoSleep.dropna().astype(float).mean()
        expected_age=other.Age[other.Age.ge(18)].mean()
        np.testing.assert_allclose(share.iloc[i],expected_share,equal_nan=True,atol=1e-14)
        np.testing.assert_allclose(mean_age.iloc[i],expected_age,equal_nan=True,atol=1e-12)
        assert other_children.iloc[i]==other.Age.le(12).sum()
    # This small follow-up scan was chosen after examining the original slices.
    # It is descriptive; no modulo feature or corrected label is fed to a model.
    modulo=[]
    base=(d.HomePlanet.eq("Earth")&d.Deck.eq("G")).fillna(False)
    for cohort_name,mask in [("Earth G",base),("Earth G cryo true",base&d.CryoSleep.eq(True)),("Earth G cryo false",base&d.CryoSleep.eq(False))]:
        for divisor in [2,3,4,5]:
            for remainder in range(divisor):
                part=(mask&d.CabinNum.mod(divisor).eq(remainder)).fillna(False).to_numpy(bool)
                if part.any():modulo.append(dict(cohort=cohort_name,divisor=divisor,remainder=remainder,n=int(part.sum()),
                    positive_rate=float(y[part].mean()),error_rate=float(wrong[part].mean())))
    _write_csv(pd.DataFrame(modulo),report/"mechanisms/cabin_modulo_exploration.csv")
    _write_json(PROJECT_ROOT/"reports/checks/blindspot_independent_verification_20261029.json",dict(passed=True,
        source_sha256=hashlib.sha256((PROJECT_ROOT/"scripts/verify_blindspot_audit.py").read_bytes()).hexdigest(),
        slice_rows_checked=len(slices),canonical_errors=1602,persistent_errors=1364,confident_persistent_errors=74,
        group_context_brute_force_cases_checked=256,candidate_context_self_exclusion_matches=True,
        modulo_exploration_rows=len(modulo),new_model_trained=False,new_submission_generated=False))
    print(f"PASS: {len(slices)} slice rows, 1602 errors, 74 persistent high-confidence cases, 256 contextual feature checks")


if __name__=="__main__":main()
