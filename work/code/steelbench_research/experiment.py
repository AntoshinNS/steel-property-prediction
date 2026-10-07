"""Offline, group-disjoint attribution audit. See README_RU.md before real use.

The permutation estimator implements the empirical interventional Shapley game;
it is not the SHAP library or TreeSHAP. Every path is additive, but finite-path
attributions are Monte Carlo approximations. No model tuning uses test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT = {
    "columns": {"sample": "sample_id", "grade": "grade", "family": "family",
                "source": "source", "unit": "heat_id", "target": "tensile_strength"},
    "features": ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Cu", "Al", "austenitize_T", "temper_T"],
    "grade_aliases": {},
    "unit_scope": "global",
    "duplicate_policy": "error",
    "protocols": ["random", "grade", "family", "source"],
    "seed": 2026, "outer_folds": 5, "inner_folds": 3,
    "bootstrap_models": 10, "background_size": 32, "permutation_pairs": 16,
    "background_repeats": 5, "mc_repeats": 3, "neighbors": 5,
    "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
    "min_samples_leaf": 3, "ridge_alpha": 1.0, "diagnostic_alpha": 1.0,
    "bootstrap_ci": 2000, "retention_grid": [round(x, 2) for x in np.linspace(.5, 1., 11)],
    "target_unit": "MPa", "dataset_label": "SteelBench — release to be specified",
    "provenance_reviewed": False, "units_reviewed": False,
}


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_config(c):
    for key in DEFAULT:
        if key not in c:
            raise ValueError(f"Missing config field: {key}")
    for key in ("bootstrap_models", "inner_folds", "outer_folds"):
        if c[key] < 2:
            raise ValueError(f"{key} must be >= 2")
    for key in ("background_size", "permutation_pairs", "neighbors", "bootstrap_ci", "background_repeats", "mc_repeats"):
        if c[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    q=np.array(c["retention_grid"])
    if len(q)<2 or np.any(np.diff(q)<=0) or q[0] != .5 or q[-1] != 1.:
        raise ValueError("retention_grid must increase from 0.5 to 1.0")
    if not set(c["protocols"]) <= {"random", "grade", "family", "source"}:
        raise ValueError("Unsupported protocol")
    if c["unit_scope"] not in {"global", "source"} or c["duplicate_policy"] not in {"error", "same_unit", "allow_reviewed"}:
        raise ValueError("Invalid unit_scope or duplicate_policy")
    if len(set(c["features"])) != len(c["features"]) or not c["features"]:
        raise ValueError("Features must be unique and nonempty")
    if set(c["features"]) & set(c["columns"].values()):
        raise ValueError("Features cannot contain target or identifier columns")


def load_data(path, c):
    """Explicit schema; no silent grade inference, unit conversion or target fill."""
    validate_config(c)
    raw=pd.read_csv(path, dtype={c["columns"][k]:"string" for k in ("sample","grade","family","source","unit")})
    needed=set(c["features"]) | set(c["columns"].values())
    missing=needed-set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Edit columns/features in config; do not invent labels.")
    d=pd.DataFrame(index=raw.index)
    for role in ("sample", "grade", "family", "source", "unit"):
        v=raw[c["columns"][role]].astype("string").str.strip()
        if v.isna().any() or v.eq("").any():
            raise ValueError(f"Missing {role} identifiers")
        d[role]=v.astype(str)
    d["grade"]=d.grade.str.upper().replace({str(k).strip().upper():str(v).strip().upper() for k,v in c["grade_aliases"].items()})
    if d["sample"].duplicated().any():
        raise ValueError("sample identifiers must be unique")
    if c["unit_scope"] == "source":
        d["unit"]=[json.dumps([s,u]) for s,u in zip(d.source,d.unit)]
    for col in c["features"]+[c["columns"]["target"]]:
        values=pd.to_numeric(raw[col], errors="raise").astype(float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite value in {col}")
        d["y" if col==c["columns"]["target"] else col]=values
    excluded=d.loc[d.y.isna(),["sample"]].assign(reason="missing_reported_target")
    d=d.loc[d.y.notna()].reset_index(drop=True)
    if len(d)<8 or d.y.std()==0:
        raise ValueError("Too few usable observations or constant target")
    if d.groupby("grade").family.nunique().max()>1:
        raise ValueError("A normalized grade spans multiple families; resolve taxonomy first")
    if d.groupby("unit").grade.nunique().max()>1:
        raise ValueError("An independent unit spans normalized grades; resolve duplicate/alias provenance")
    # Exact input duplicates are conservative flags, not automatic data deletions.
    sig=pd.util.hash_pandas_object(d[c["features"]],index=False)
    dup=d.loc[sig.duplicated(keep=False)].copy()
    if len(dup):
        check=pd.DataFrame({"signature":sig,"unit":d.unit}).groupby("signature").unit.nunique()
        if c["duplicate_policy"]=="error" or (c["duplicate_policy"]=="same_unit" and check.max()>1):
            raise ValueError("Exact feature duplicates detected. Resolve provenance or explicitly permit same_unit duplicates.")
    audit={"input_rows":len(raw),"usable_rows":len(d),"excluded_missing_target":len(excluded),
           "grades":int(d.grade.nunique()),"families":int(d.family.nunique()),
           "sources":int(d.source.nunique()),"independent_units":int(d.unit.nunique()),
           "exact_duplicate_rows":len(dup),"feature_missing":{k:int(d[k].isna().sum()) for k in c["features"]},
           "source_file_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    return d, audit, excluded


def partition(d, protocol, n_splits, seed, inner=False):
    group_col={"random":"unit","grade":"grade","family":"family","source":"source"}[protocol]
    groups=d[group_col].to_numpy()
    n=len(np.unique(groups))
    if n<2:
        raise ValueError(f"At least two {group_col} groups are required")
    splitter=(LeaveOneGroupOut() if protocol in {"family","source"} and not inner
              else GroupKFold(n_splits=min(n_splits,n),shuffle=True,random_state=seed))
    for tr,te in splitter.split(d,groups=groups):
        a,b=d.iloc[tr],d.iloc[te]
        for field in {"unit",group_col}:
            if set(a[field]) & set(b[field]):
                raise ValueError(f"Leakage: {field} overlaps in {protocol} split")
        yield tr,te


def base_model(c, seed):
    return make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
        GradientBoostingRegressor(n_estimators=c["n_estimators"],max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],min_samples_leaf=c["min_samples_leaf"],random_state=seed))


def permutation_attributions(predict, x, background, pairs=16, seed=0):
    """Unbiased sampled-permutation Shapley estimates for a finite background.

    v_x(S) = mean_z f(x_S,z_notS). Antithetic paths share all permutations across
    models. Evaluation is vectorized over observations and backgrounds.
    """
    x=np.asarray(x,float);background=np.asarray(background,float)
    n,p=x.shape; m=len(background)
    if not m or pairs<1:
        raise ValueError("Nonempty background and positive pairs required")
    rng=np.random.default_rng(seed)
    phi=np.zeros((n,p)); expected=float(np.mean(predict(background)))
    # Keep batches bounded for moderate benchmark sizes.
    orders=[]
    for _ in range(pairs):
        order=rng.permutation(p);orders.extend([order,order[::-1]])
    for start in range(0,n,32):
        xx=x[start:start+32]; nn=len(xx)
        for order in orders:
            hybrid=np.broadcast_to(background,(nn,m,p)).copy()
            prev=np.full(nn,expected)
            for j in order:
                hybrid[:,:,j]=xx[:,None,j]
                now=predict(hybrid.reshape(-1,p)).reshape(nn,m).mean(axis=1)
                phi[start:start+nn,j]+=(now-prev)/len(orders)
                prev=now
    pred=predict(x)
    if not np.allclose(phi.sum(axis=1)+expected,pred,rtol=1e-7,atol=1e-6):
        raise AssertionError("Attribution additivity failed")
    return phi, expected


def background_rows(d,size,seed):
    # Random row per independent unit, then uniform units.
    pick=d.sample(frac=1,random_state=seed).drop_duplicates("unit")
    return pick.sample(n=min(size,len(pick)),random_state=seed)


def pair_stability(phi):
    n,p=phi.shape[1:]; rank=np.full(n,np.nan); overlap=np.zeros(n)
    for i in range(n):
        rho=[]; jac=[]
        for a,b in itertools.combinations(range(len(phi)),2):
            va,vb=np.abs(phi[a,i]),np.abs(phi[b,i])
            if np.ptp(va)>0 and np.ptp(vb)>0:
                rho.append(float(spearmanr(va,vb).statistic))
            # Feature index supplies deterministic tie-breaking.
            aa=set(np.argsort(-va,kind="stable")[:min(3,p)])
            bb=set(np.argsort(-vb,kind="stable")[:min(3,p)])
            jac.append(len(aa&bb)/len(aa|bb))
        rank[i]=np.mean(rho) if rho else np.nan
        overlap[i]=np.mean(jac)
    return rank,overlap


def ensemble_diagnostics(train, test, c, seed, store_dir=None):
    feats=c["features"];x=train[feats].to_numpy();xt=test[feats].to_numpy()
    sy=float(train.y.std(ddof=1))
    if not sy>0:
        raise ValueError("Constant training target")
    bg=background_rows(train,c["background_size"],seed)
    rng=np.random.default_rng(seed);groups=train.grade.unique()
    models=[]; predictions=[]; phis=[]; expected=[]; fits=[]
    for b in range(c["bootstrap_models"]):
        chosen=rng.choice(groups,len(groups),replace=True)
        ix=np.concatenate([np.flatnonzero(train.grade.to_numpy()==g) for g in chosen])
        model=base_model(c,seed+b).fit(x[ix],train.y.to_numpy()[ix])
        phi,e=permutation_attributions(model.predict,xt,bg[feats].to_numpy(),c["permutation_pairs"],seed+123)
        models.append(model);predictions.append(model.predict(xt));phis.append(phi);expected.append(e)
        fits.append({"member":b,"sample_ids":train.iloc[ix]["sample"].tolist()})
    pred=np.array(predictions);phi=np.array(phis)
    # One representative per independent unit for distance, train-only preprocessing.
    prep=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler()).fit(x)
    reps=train.drop_duplicates("unit")[feats].to_numpy()
    nearest=NearestNeighbors(n_neighbors=min(c["neighbors"],len(reps))).fit(prep.transform(reps))
    dist=nearest.kneighbors(prep.transform(xt))[0].mean(axis=1)
    sx=phi.std(axis=0,ddof=1).mean(axis=1)
    out=pd.DataFrame({"sample":test["sample"].to_numpy(),"prediction":pred.mean(axis=0),
        "distance":dist,"prediction_sd":pred.std(axis=0,ddof=1),
        "dispersion":pred.std(axis=0,ddof=1)/sy,"attribution":sx/sy,"attribution_sd":sx})
    if store_dir is not None:
        store_dir=Path(store_dir);store_dir.mkdir(parents=True,exist_ok=True)
        rank,jac=pair_stability(phi);out["rank_agreement"]=rank;out["top3_jaccard"]=jac
        # Model fixed: change background only, with common permutation random draws.
        bgphis=[]; bgrecords=[]
        for r in range(c["background_repeats"]):
            br=background_rows(train,c["background_size"],seed+1000+r)
            q,e=permutation_attributions(models[0].predict,xt,br[feats].to_numpy(),c["permutation_pairs"],seed+123)
            bgphis.append(q);bgrecords.append({"sample_ids":br["sample"].tolist(),"expected":e})
        # Model and background fixed: change permutations only, measuring MC error.
        mcphis=[]
        for r in range(c["mc_repeats"]):
            q,_=permutation_attributions(models[0].predict,xt,bg[feats].to_numpy(),c["permutation_pairs"],seed+2000+r)
            mcphis.append(q)
        out["background_sd"]=(np.std(bgphis,axis=0,ddof=1).mean(axis=1) if len(bgphis)>1 else np.nan)
        out["mc_sd"]=(np.std(mcphis,axis=0,ddof=1).mean(axis=1) if len(mcphis)>1 else np.nan)
        np.savez_compressed(store_dir/"attributions.npz",phi=phi,expected=np.array(expected),
            predictions=pred,sample_ids=test["sample"].to_numpy(dtype=str),features=np.array(feats),
            background_phi=np.array(bgphis),mc_phi=np.array(mcphis))
        json_write(store_dir/"fit_manifest.json",{"training_samples":train["sample"].tolist(),"background_samples":bg["sample"].tolist(),
            "bootstrap_members":fits,"background_variants":bgrecords,"seed":seed,"permutation_seed":seed+123,"target_sd":sy})
        joblib.dump(models,store_dir/"ensemble.joblib")
    return out


def risk_curve(y,pred,score,ids,grid):
    y=np.asarray(y);pred=np.asarray(pred);score=np.asarray(score)
    if not np.isfinite(score).all(): raise ValueError("Nonfinite risk score")
    # Lexicographic ID order prevents input row order from becoming a hidden tie rule.
    order=np.lexsort((np.asarray(ids,dtype=str),score));n=len(y)
    rows=[]
    for q in grid:
        k=max(1,int(np.ceil(q*n-1e-12)));sel=order[:k]
        rows.append({"requested_retention":q,"accepted":k,"actual_retention":k/n,
                     "mae":float(np.mean(np.abs(y[sel]-pred[sel])))})
    aurc=float(np.trapezoid([r["mae"] for r in rows],grid)/(grid[-1]-grid[0]))
    return aurc,rows


def regression_metrics(y,pred):
    return {"mae":float(mean_absolute_error(y,pred)),"rmse":float(np.sqrt(mean_squared_error(y,pred))),
            "r2":float(r2_score(y,pred)) if len(y)>1 and np.var(y)>0 else np.nan}


def summarize(pred,c,out):
    curves=[];scores=[];predictive=[]
    for (protocol,family),part in pred.groupby(["protocol","family"],sort=True):
        for model in ("prediction","ridge_prediction","dummy_prediction"):
            predictive.append({"protocol":protocol,"family":family,"model":model,"n":len(part),
                               **regression_metrics(part.y,part[model])})
        # Cross-fold scores on grade/random data can have different rank scales;
        # within-family diagnostics are confirmatory only for single-model-domain LOFO.
        for method in ("distance","dispersion","g0","g1"):
            a,rr=risk_curve(part.y,part.prediction,part[method],part["sample"],c["retention_grid"])
            scores.append({"protocol":protocol,"family":family,"method":method,"n":len(part),"aurc":a})
            curves.extend({"protocol":protocol,"family":family,"method":method,**r} for r in rr)
    pd.DataFrame(predictive).to_csv(out/"metrics_by_family.csv",index=False)
    summary=[]
    for protocol,part in pred.groupby("protocol"):
        for model in ("prediction","ridge_prediction","dummy_prediction"):
            group=[r for r in predictive if r["protocol"]==protocol and r["model"]==model]
            summary.append({"protocol":protocol,"model":model,**regression_metrics(part.y,part[model]),
                            "macro_mae":float(np.mean([r["mae"] for r in group]))})
    pd.DataFrame(summary).to_csv(out/"metrics_overall.csv",index=False)
    curves=pd.DataFrame(curves);scores=pd.DataFrame(scores)
    curves.to_csv(out/"risk_retention.csv",index=False);scores.to_csv(out/"aurc_by_family.csv",index=False)
    primary=scores.loc[scores.protocol=="family"].pivot(index="family",columns="method",values="aurc")
    result={"status":"not_run","dataset_label":c["dataset_label"]}
    if len(primary):
        primary["delta_g0_minus_g1"]=primary.g0-primary.g1
        primary.to_csv(out/"primary_paired_effects.csv")
        delta=primary.delta_g0_minus_g1.to_numpy();rng=np.random.default_rng(c["seed"])
        boot=rng.choice(delta,(c["bootstrap_ci"],len(delta)),replace=True).mean(axis=1)
        result={"status":"computed","dataset_label":c["dataset_label"],"families":len(delta),
            "mean_delta_aurc":float(delta.mean()),"median_delta_aurc":float(np.median(delta)),
            "ci95_family_bootstrap":np.quantile(boot,[.025,.975]).tolist(),"families_improved":int((delta>0).sum()),
            "interpretation":"Positive delta favors g1. Conditional on fitted procedure and observed families; no automatic superiority claim."}
    json_write(out/"primary_result.json",result)
    # This is a descriptive mean over families, not a confidence band.
    lo=curves[curves.protocol=="family"]
    if len(lo):
        fig,ax=plt.subplots(figsize=(6,4))
        for method,g in lo.groupby("method"):
            avg=g.groupby("requested_retention").mae.mean()
            ax.plot(avg.index,avg.values,marker="o",ms=3,label=method)
        ax.set(xlabel="Retained fraction (requested)",ylabel=f"Macro MAE ({c['target_unit']})",title=c["dataset_label"])
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/"risk_retention.png",dpi=220);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,max(3,len(primary)*.25)))
        ax.barh(primary.index.astype(str),primary.delta_g0_minus_g1)
        ax.axvline(0,color="black",lw=.8)
        ax.set(xlabel=f"AURC(g0) − AURC(g1) ({c['target_unit']})",title=c["dataset_label"])
        fig.tight_layout();fig.savefig(out/"paired_effects.png",dpi=220);plt.close(fig)
    m=pd.DataFrame(summary)
    fig,ax=plt.subplots(figsize=(6,4))
    m.pivot(index="protocol",columns="model",values="macro_mae").plot.bar(ax=ax)
    ax.set(xlabel="Evaluation protocol",ylabel=f"Macro MAE ({c['target_unit']})",title=c["dataset_label"])
    ax.tick_params(axis="x",rotation=0);fig.tight_layout();fig.savefig(out/"generalization.png",dpi=220);plt.close(fig)
    return result


def run(path,c,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty; use a new run directory")
    out.mkdir(parents=True,exist_ok=True)
    json_write(out/"run_status.json",{"state":"running"})
    if not c["provenance_reviewed"] or not c["units_reviewed"]:
        raise ValueError("Review target provenance, group IDs and units; then explicitly set the two reviewed flags in config")
    start=perf_counter();d,audit,excluded=load_data(path,c)
    json_write(out/"config.json",c);json_write(out/"audit.json",audit)
    excluded.to_csv(out/"excluded_rows.csv",index=False)
    pd.crosstab(d.family,d.source).to_csv(out/"family_by_source.csv")
    versions={k:importlib.metadata.version(k) for k in ("numpy","pandas","scipy","scikit-learn","matplotlib","joblib")}
    json_write(out/"environment.json",{"python":platform.python_version(),"packages":versions,
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    allpred=[];splitlog=[]
    for protocol in c["protocols"]:
        for fold,(tr,te) in enumerate(partition(d,protocol,c["outer_folds"],c["seed"])):
            train,test=d.iloc[tr].reset_index(drop=True),d.iloc[te].reset_index(drop=True)
            seed=c["seed"]+10000*list(DEFAULT["protocols"]).index(protocol)+100*fold
            fd=out/f"{protocol}_{fold:03d}";fd.mkdir()
            print(f"{protocol} fold {fold}: train={len(train)} test={len(test)}",flush=True)
            inner=[]
            for k,(a,b) in enumerate(partition(train,protocol,c["inner_folds"],seed,inner=True)):
                aa,bb=train.iloc[a].reset_index(drop=True),train.iloc[b].reset_index(drop=True)
                diag=ensemble_diagnostics(aa,bb,c,seed+10+k)
                diag["error"]=np.abs(bb.y.to_numpy()-diag.prediction.to_numpy())
                diag["inner_fold"]=k;inner.append(diag)
                splitlog.extend({"protocol":protocol,"outer_fold":fold,"inner_fold":k,"role":role,"sample":sid}
                    for role,subset in (("inner_train",aa),("inner_validation",bb)) for sid in subset["sample"])
            inner=pd.concat(inner,ignore_index=True)
            if set(inner["sample"]) != set(train["sample"]) or inner["sample"].duplicated().any():
                raise AssertionError("Internal OOF coverage failed")
            inner.to_csv(fd/"diagnostic_oof.csv",index=False)
            # Fixed alpha avoids tuning diagnostics on correlated OOF pseudo-labels.
            gmodels={}
            for name,cols in (("g0",["distance","dispersion"]),("g1",["distance","dispersion","attribution"])):
                gmodels[name]=(make_pipeline(StandardScaler(),Ridge(alpha=c["diagnostic_alpha"]))
                    .fit(inner[cols],np.log1p(inner.error)),cols)
            outer=ensemble_diagnostics(train,test,c,seed,store_dir=fd)
            for name,(gm,cols) in gmodels.items():outer[name]=gm.predict(outer[cols])
            for name,est in (("ridge_prediction",make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=c["ridge_alpha"]))),
                             ("dummy_prediction",DummyRegressor())):
                est.fit(train[c["features"]].to_numpy(),train.y)
                outer[name]=est.predict(test[c["features"]].to_numpy())
            for col in ("y","family","grade","source","unit"):outer[col]=test[col].to_numpy()
            outer["protocol"]=protocol;outer["outer_fold"]=fold
            outer.to_csv(fd/"predictions.csv",index=False);joblib.dump(gmodels,fd/"diagnostic_models.joblib")
            allpred.append(outer)
            splitlog.extend({"protocol":protocol,"outer_fold":fold,"inner_fold":-1,"role":role,"sample":sid}
                for role,subset in (("outer_train",train),("outer_test",test)) for sid in subset["sample"])
    pred=pd.concat(allpred,ignore_index=True)
    if pred.duplicated(["protocol","sample"]).any():raise AssertionError("Duplicate outer prediction")
    pred.to_csv(out/"predictions.csv",index=False);pd.DataFrame(splitlog).to_csv(out/"split_manifest.csv",index=False)
    result=summarize(pred,c,out)
    json_write(out/"run_status.json",{"state":"complete","seconds":perf_counter()-start,"result":result})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def synthetic(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise ValueError("Synthetic destination must be empty")
    rng=np.random.default_rng(19);rows=[]
    for f in range(5):
        for g in range(3):
            for j in range(5):
                x=rng.normal(size=3)+f*.4
                y=500+80*x[0]+30*x[1]*x[2]+rng.normal(0,10)+(50 if f==4 else 0)
                rows.append({"sample_id":f"S{f}_{g}_{j}","grade":f"F{f}G{g}","family":f"F{f}",
                    "source":f"SRC{j%3}","heat_id":f"H{f}_{g}_{j}","tensile_strength":y,
                    "x1":x[0],"x2":x[1],"x3":x[2]})
    pd.DataFrame(rows).to_csv(folder/"SYNTHETIC_NOT_STEEL.csv",index=False)
    c=json.loads(json.dumps(DEFAULT));c.update(features=["x1","x2","x3"],protocols=["family"],
        bootstrap_models=3,background_size=4,permutation_pairs=2,background_repeats=2,mc_repeats=2,
        n_estimators=20,inner_folds=2,bootstrap_ci=200,provenance_reviewed=True,units_reviewed=True,
        dataset_label="SYNTHETIC SOFTWARE TEST — NOT STEELBENCH")
    json_write(folder/"smoke_config.json",c)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("make-smoke-data");s.add_argument("--out",required=True)
    s=sub.add_parser("config");s.add_argument("--out",required=True)
    for cmd in ("audit","run"):
        s=sub.add_parser(cmd);s.add_argument("--data",required=True);s.add_argument("--config",required=True);s.add_argument("--out",required=True)
    args=parser.parse_args()
    if args.command=="make-smoke-data":synthetic(args.out)
    elif args.command=="config":json_write(args.out,DEFAULT)
    elif args.command=="run":run(args.data,json.loads(Path(args.config).read_text()),args.out)
    else:
        c=json.loads(Path(args.config).read_text());d,audit,excluded=load_data(args.data,c)
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        json_write(out/"audit.json",audit);excluded.to_csv(out/"excluded_rows.csv",index=False)
        pd.crosstab(d.family,d.source).to_csv(out/"family_by_source.csv")
        print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
