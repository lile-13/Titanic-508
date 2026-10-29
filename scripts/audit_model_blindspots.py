"""Describe frozen-model OOF errors and feature gaps; never fit or promote a model."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from threadpoolctl import threadpool_limits

from src import preprocess as pp
from src.competition_features import build_pool
from src.paths import PROJECT_ROOT,EXPERIMENTS_DIR
from src.train import _write_csv,_write_json

NAME="error_audit_20261029"
CATEGORICAL=["HomePlanet","Deck","Side","CryoSleep","Destination","age_band",
    "spend_state","spend_band","spend_pattern","group_size_band","missing_band",
    "group_cryo_mix","group_spend_mix","group_cabin_mix","cabin_third","group_position"]
INTERACTIONS=[("HomePlanet","Deck"),("HomePlanet","CryoSleep"),
    ("HomePlanet","spend_state"),("Deck","CryoSleep"),("Deck","spend_state"),
    ("CryoSleep","spend_state"),("HomePlanet","spend_pattern"),
    ("CryoSleep","group_spend_mix"),("spend_state","group_size_band"),
    ("spend_state","group_cryo_mix"),("Deck","cabin_third")]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptors(frame):
    out=pd.DataFrame(index=frame.index)
    for c in ["HomePlanet","Deck","Side","CryoSleep","Destination"]:
        out[c]=frame[c].astype("string").fillna("Missing")
    out["age_band"]=pd.cut(frame.Age,[-np.inf,12,17,25,40,60,np.inf],
        labels=["0-12","13-17","18-25","26-40","41-60","61+"]).astype("string").fillna("Missing")
    spend=frame[pp.SPEND_COLS];complete=spend.notna().all(axis=1)
    positive=spend.fillna(0).gt(0).any(axis=1)
    out["spend_state"]=np.select([positive,complete],["Positive observed","Confirmed zero"],default="Zero/unknown")
    total=spend.sum(axis=1,min_count=1)
    out["spend_band"]=pd.cut(total,[-np.inf,0,100,500,1500,5000,np.inf],
        labels=["0","1-100","101-500","501-1500","1501-5000","5001+"]).astype("string").fillna("Missing")
    personal=spend[["RoomService","Spa","VRDeck"]].fillna(0).gt(0).any(axis=1)
    social=spend[["FoodCourt","ShoppingMall"]].fillna(0).gt(0).any(axis=1)
    out["spend_pattern"]=np.select([~complete,personal&social,personal,social],
        ["Incomplete","Both","Personal only","Social only"],default="Zero")
    out["group_size_band"]=pd.cut(frame.GroupSize,[0,1,2,4,np.inf],labels=["1","2","3-4","5+"]).astype("string")
    missing=frame[pp.MISSING_INDICATOR_COLS].isna().sum(axis=1)
    out["missing_band"]=pd.cut(missing,[-1,0,1,np.inf],labels=["0","1","2+"]).astype("string")
    # All group descriptors use the same public unlabeled pool, never group outcomes.
    known_true=frame.CryoSleep.eq(True).fillna(False)
    known_false=frame.CryoSleep.eq(False).fillna(False)
    has_true=known_true.groupby(frame.GroupId).transform("any")
    has_false=known_false.groupby(frame.GroupId).transform("any")
    out["group_cryo_mix"]=np.select([has_true&has_false,has_true,has_false],
        ["Mixed known states","Only true observed","Only false observed"],default="Unknown")
    has_positive=positive.groupby(frame.GroupId).transform("any")
    has_zero=(complete&~positive).groupby(frame.GroupId).transform("any")
    out["group_spend_mix"]=np.select([has_positive&has_zero,has_positive,has_zero],
        ["Positive and zero","Positive without known zero","Zero without positive"],default="Unknown")
    cabin=frame.Deck.astype("string")+"/"+frame.CabinNum.astype("string")+"/"+frame.Side.astype("string")
    cabin=cabin.where(frame[["Deck","CabinNum","Side"]].notna().all(axis=1))
    mixed=cabin.groupby(frame.GroupId).transform("nunique")>1
    out["group_cabin_mix"]=np.where(mixed,"Multiple known cabins","At most one known cabin")
    out["cabin_third"]=pd.cut(frame.CabinRelative,[-np.inf,1/3,2/3,np.inf],labels=["First","Middle","Last"]).astype("string").fillna("Missing")
    out["group_position"]=pd.cut(frame.GroupIdNum,[-np.inf,2000,4000,6000,8000,np.inf],
        labels=["0-2000","2001-4000","4001-6000","6001-8000","8001+"]).astype("string")
    return out


def main():
    report=EXPERIMENTS_DIR/NAME;report.mkdir(exist_ok=True)
    raw,test=pp.load_raw();features,test_features=build_pool(raw,test)
    public=pd.concat([features.drop(columns=pp.TARGET),test_features],ignore_index=True)
    description=descriptors(public);desc=description.iloc[:len(raw)].reset_index(drop=True)
    test_desc=description.iloc[len(raw):].reset_index(drop=True)
    original=EXPERIMENTS_DIR/"optimization_20261025"
    canonical=pd.read_csv(original/"oof_predictions.csv",dtype={"PassengerId":str,"GroupId":str},float_precision="round_trip")
    np.testing.assert_array_equal(canonical.PassengerId,raw.PassengerId)
    np.testing.assert_array_equal(canonical.GroupId,features.GroupId)
    y=raw.Transported.to_numpy(int);fold=canonical.fold.to_numpy()
    np.testing.assert_array_equal(canonical.Transported.to_numpy(int),y)
    ledger=EXPERIMENTS_DIR/"kaggle_submissions.json";ledger_hash=sha(ledger)
    submission=PROJECT_ROOT/"outputs/kaggle_ready_4690/submission_ensemble4690.csv"
    submission_hash=sha(submission)
    protocol=dict(name=NAME,purpose="Diagnostic only; no fitting, threshold choice or candidate promotion",
        categorical_slices=CATEGORICAL,interaction_slices=INTERACTIONS,minimum_slice_rows=200,
        ranking="Error count above overall rate; prioritize >=5 percentage points higher error in all three historical OOF variants. Overlapping slices are not additive.",
        bootstrap="1000 GroupId-cluster multinomial resamples, seed 20261029; conditional fixed-prediction intervals for top 12. Exploratory and not adjusted for slice selection or multiple comparisons.",
        caveat="Canonical OOF includes selection folds; repartitioned OOF reuses the same historical rows. These are stability descriptions, not independent confirmation or causal evidence.",
        sources={p:sha(PROJECT_ROOT/p) for p in ["scripts/audit_model_blindspots.py","src/competition_features.py","src/preprocess.py","features/folds.csv"]},
        model_sha256=sha(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib"),
        input_hashes={str(p.relative_to(PROJECT_ROOT)):sha(p) for p in [original/"oof_predictions.csv",original/"frozen_plan.json",ledger]},
        data_hashes={n:sha(pp.find_data_dir()/n) for n in ["train.csv","test.csv"]})
    path=report/"protocol.json"
    if path.exists():assert json.loads(path.read_text(encoding="utf-8"))==json.loads(json.dumps(protocol))
    else:_write_json(path,protocol)
    probability=[canonical.probability.to_numpy()]
    for repeat in [1,2]:
        file=EXPERIMENTS_DIR/"fixed_weight_confirmation_20261029"/f"repeat_{repeat}_oof.csv"
        data=pd.read_csv(file,dtype={"PassengerId":str},float_precision="round_trip")
        np.testing.assert_array_equal(data.PassengerId,raw.PassengerId)
        np.testing.assert_array_equal(data.Transported.to_numpy(int),y)
        probability.append(data.reference.to_numpy());protocol["input_hashes"][str(file.relative_to(PROJECT_ROOT))]=sha(file)
    # Persist complete provenance before reading any slice's outcome summary.
    if len(protocol["input_hashes"])>3:
        _write_json(report/"input_hashes.json",protocol["input_hashes"])
    probability=np.column_stack(probability);errors=(probability>=.5)!=y[:,None]
    bundle=joblib.load(PROJECT_ROOT/"models/optimization_20261025/ensemble.joblib")
    plan=bundle["plan"];per_fold=sum(len(m["seeds"]) for m in plan["members"])
    component=np.full((len(raw),3),np.nan);seed_probability=np.full((len(raw),9),np.nan);coverage=[]
    for block,f in enumerate([5,1,2,3,4]):
        vi=np.flatnonzero(fold==f);ti=np.flatnonzero(fold!=f)
        assert not set(features.GroupId.iloc[ti])&set(features.GroupId.iloc[vi])
        members=bundle["members"][block*per_fold:(block+1)*per_fold]
        assert len(members)==9
        for index,member in enumerate(members):
            X=member["builder"].transform(features.iloc[vi])
            assert pp.TARGET not in X and "PassengerId" not in X
            p=member["model"].predict_proba(member["encoder"].transform(X))[:,1]
            seed_probability[vi,index]=p
            if block==0 and index%3==0:
                coverage.append(dict(component=index//3,weight=plan["weights"][index//3],columns=list(X)))
        for c in range(3):component[vi,c]=seed_probability[vi,c*3:c*3+3].mean(axis=1)
        print(f"Blindspot audit: frozen fold {f} components reproduced",flush=True)
    np.testing.assert_allclose(component@np.array(plan["weights"]),probability[:,0],atol=1e-12,rtol=0)
    component_error=(component>=.5)!=y[:,None]
    component_vote=(component>=.5).sum(axis=1)
    frame=pd.DataFrame(dict(PassengerId=raw.PassengerId,GroupId=features.GroupId,fold=fold,Transported=y,
        probability=probability[:,0],error=errors[:,0],error_variants=errors.sum(axis=1),
        component_positive_votes=component_vote,all_components_wrong=component_error.all(axis=1),
        cat_spend_probability=component[:,0],cat_names_probability=component[:,1],lgb_probability=component[:,2]))
    frame=pd.concat([frame,desc],axis=1)
    _write_csv(frame,report/"passenger_diagnostics.csv")
    wrong=errors[:,0];baseline_error=errors.mean(axis=0);slices=[];masks=[]
    for columns in [(c,) for c in CATEGORICAL]+INTERACTIONS:
        name=" + ".join(columns)
        key=desc[list(columns)].astype("string").fillna("Missing").agg(" | ".join,axis=1)
        test_key=test_desc[list(columns)].astype("string").fillna("Missing").agg(" | ".join,axis=1)
        for value in sorted(key.unique()):
            mask=key.eq(value).to_numpy();n=int(mask.sum())
            rate=errors[mask].mean(axis=0);model_accuracy=1-component_error[mask].mean(axis=0)
            row=dict(slice_id=len(slices),columns=name,value=value,n=n,groups=int(features.GroupId[mask].nunique()),
                test_n=int(test_key.eq(value).sum()),errors=int(wrong[mask].sum()),error_rate=float(rate[0]),
                error_share=float(wrong[mask].sum()/wrong.sum()),excess_errors=float(n*(rate[0]-baseline_error[0])),
                repeat_1_error_rate=float(rate[1]),repeat_2_error_rate=float(rate[2]),
                excess_canonical=float(rate[0]-baseline_error[0]),excess_repeat_1=float(rate[1]-baseline_error[1]),
                excess_repeat_2=float(rate[2]-baseline_error[2]),persistent_wrong=int(errors[mask].all(axis=1).sum()),
                cat_spend_accuracy=float(model_accuracy[0]),cat_names_accuracy=float(model_accuracy[1]),
                lgb_accuracy=float(model_accuracy[2]),ensemble_accuracy=float(1-rate[0]),
                mean_probability=float(probability[mask,0].mean()),positive_rate=float(y[mask].mean()))
            slices.append(row);masks.append(mask)
    table=pd.DataFrame(slices)
    table["stable_high_risk"]=(table.n>=200)&(table[["excess_canonical","excess_repeat_1","excess_repeat_2"]]>=.05).all(axis=1)
    table=table.sort_values(["stable_high_risk","excess_errors"],ascending=False)
    _write_csv(table,report/"slice_metrics.csv")
    selected=table[table.stable_high_risk].head(12)
    _,inverse=np.unique(features.GroupId,return_inverse=True);g=inverse.max()+1
    counts=np.zeros((g,len(selected)*2+2))
    counts[:,0]=np.bincount(inverse,minlength=g);counts[:,1]=np.bincount(inverse,weights=wrong,minlength=g)
    for index,row in enumerate(selected.itertuples()):
        mask=masks[row.slice_id]
        counts[:,2+index*2]=np.bincount(inverse,weights=mask,minlength=g)
        counts[:,3+index*2]=np.bincount(inverse,weights=mask&wrong,minlength=g)
    with threadpool_limits(limits=4):
        boot=np.random.default_rng(20261029).multinomial(g,np.full(g,1/g),size=1000)@counts
    intervals=[]
    for index,row in enumerate(selected.itertuples()):
        n,e=boot[:,2+index*2],boot[:,3+index*2]
        rate=np.divide(e,n,out=np.full_like(e,np.nan),where=n>0)
        other_n=boot[:,0]-n;other_e=boot[:,1]-e
        other=np.divide(other_e,other_n,out=np.full_like(e,np.nan),where=other_n>0)
        lo,hi=np.nanquantile(rate,[.025,.975]);dlo,dhi=np.nanquantile(rate-other,[.025,.975])
        intervals.append(dict(slice_id=row.slice_id,columns=row.columns,value=row.value,error_rate_lower=float(lo),
            error_rate_upper=float(hi),gap_to_complement_lower=float(dlo),gap_to_complement_upper=float(dhi)))
    _write_json(report/"conditional_cluster_intervals.json",intervals)
    buckets=pd.cut(np.abs(probability[:,0]-.5),[-1,.05,.15,.3,.5],labels=["0-.05",".05-.15",".15-.30",".30-.50"])
    confidence=pd.DataFrame(dict(margin=buckets,error=wrong)).groupby("margin",observed=True).agg(n=("error","size"),errors=("error","sum"))
    confidence["error_rate"]=confidence.errors/confidence.n
    _write_csv(confidence.reset_index(),report/"confidence_errors.csv")
    label_probability=np.where(y[:,None]==1,probability,1-probability)
    hard=(label_probability<=.1).all(axis=1)
    _write_csv(frame.loc[wrong].sort_values(["error_variants","probability"],ascending=[False,False]),report/"error_cases.csv")
    _write_csv(frame.loc[hard],report/"confident_persistent_errors.csv")
    complementarity=dict(ensemble_wrong=int(wrong.sum()),wrong_with_some_component_correct=int((wrong&~component_error.all(axis=1)).sum()),
        wrong_with_all_components_wrong=int((wrong&component_error.all(axis=1)).sum()),
        component_disagreement_rows=int(((component_vote>0)&(component_vote<3)).sum()),
        component_disagreement_errors=int((wrong&(component_vote>0)&(component_vote<3)).sum()),
        component_accuracy=[float(accuracy_score(y,component[:,c]>=.5)) for c in range(3)],
        mean_within_component_seed_std=[float(seed_probability[:,c*3:c*3+3].std(axis=1).mean()) for c in range(3)],
        caveat="Choosing a component using these validation labels would be an oracle, not a deployable improvement")
    repeated=dict(accuracies=[float(accuracy_score(y,probability[:,c]>=.5)) for c in range(3)],
        errors_by_number_of_variants={str(k):int((errors.sum(axis=1)==k).sum()) for k in range(4)},
        canonical_wrong_persistent_all_three=int(errors.all(axis=1).sum()),confident_wrong_all_three=int(hard.sum()))
    _write_json(report/"feature_coverage.json",coverage)
    summary=dict(name=NAME,status="diagnostic_only",rows=len(raw),test_rows=len(test),
        canonical_accuracy=float(accuracy_score(y,probability[:,0]>=.5)),canonical_log_loss=float(log_loss(y,probability[:,0])),
        repeated=repeated,complementarity=complementarity,
        stable_high_risk_slices=int(table.stable_high_risk.sum()),top_slices=selected.to_dict("records"),
        unchanged_submission_sha256=submission_hash,observed_kaggle_best=.81108,
        no_new_model_fit=True,no_new_submission=True,limitations=protocol["caveat"])
    _write_json(report/"summary.json",summary)
    assert sha(ledger)==ledger_hash and sha(submission)==submission_hash
    assert np.isfinite(component).all() and len(frame)==8693 and not frame.PassengerId.duplicated().any()
    _write_json(PROJECT_ROOT/"reports/checks/error_audit_20261029.json",dict(passed=True,rows=8693,
        canonical_oof_reproduced_from_45_models=True,complete_group_separation=True,
        slice_aggregates_recomputed=True,platform_ledger_unchanged=True,submission_sha256=submission_hash,
        analysis_only=True))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
