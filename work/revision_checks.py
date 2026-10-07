"""Recheck the supplied analysis, boundary guarantees and post-review controls.

Run from the supplement root. No fitted models or original results are altered.
The checks concern supplied identifiers, not independently verified heat identity.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import itertools
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
OUT = WORK / 'revision' / 'verification'
OUT.mkdir(parents=True, exist_ok=True)
C = json.loads((WORK/'real_run/steelbench_config_real_fast.json').read_text())
F = C['features']
raw = pd.read_csv(WORK/'dataset/steelbench_core_open.csv')
a = pd.read_csv(WORK/'real_run/steelbench_core_open_analysis.csv')
pd.testing.assert_frame_equal(raw, a[raw.columns], check_exact=True)
assert raw.heat_id.is_unique and raw.tensile_strength.notna().all()
sig = pd.util.hash_pandas_object(raw[F].astype(float), index=False)
groups = pd.DataFrame({'grade': raw.grade_id, 'signature': sig})
n = groups.groupby(['grade', 'signature'])['signature'].transform('size')
reconstructed = raw.heat_id.copy()
reconstructed.loc[n>1] = 'repeat::' + raw.loc[n>1,'grade_id'] + '::' + sig[n>1].astype(str)
pd.testing.assert_series_equal(reconstructed, a.analysis_unit, check_names=False)

r = WORK/'real_run/results_uts_fast'
p = pd.read_csv(r/'predictions.csv')
m = pd.read_csv(r/'split_manifest.csv')
lookup = a.set_index('heat_id')
assert not p.duplicated(['protocol','sample']).any()
for protocol, z in p.groupby('protocol'):
    assert set(z['sample']) == set(raw.heat_id), protocol
boundaries = []
for key, z in m.groupby(['protocol','outer_fold','inner_fold']):
    tr = z.loc[z.role.str.endswith('train'),'sample']
    te = z.loc[z.role.str.endswith('test') | z.role.str.endswith('validation'),'sample']
    assert set(tr).isdisjoint(te), key
    assert set(lookup.loc[tr,'analysis_unit']).isdisjoint(lookup.loc[te,'analysis_unit']), key
    column = {'random':'analysis_unit','grade':'grade_id','family':'steel_family','source':'data_tier'}[key[0]]
    assert set(lookup.loc[tr,column]).isdisjoint(lookup.loc[te,column]), key
    boundaries.append({'protocol':key[0], 'outer_fold':int(key[1]), 'inner_fold':int(key[2]),
                       'training_rows':len(tr), 'held_out_rows':len(te), 'unit_disjoint':True, 'group_disjoint':True})
pd.DataFrame(boundaries).to_csv(OUT/'boundary_checks.csv',index=False)

by_id = pd.Series(sig.to_numpy(), index=raw.heat_id)
overlap = []
for (protocol, fold), z in p.groupby(['protocol','outer_fold']):
    test = z['sample']; train = raw.heat_id[~raw.heat_id.isin(test)]
    match = by_id.loc[test].isin(by_id.loc[train])
    overlap.append({'protocol':protocol, 'outer_fold':int(fold), 'test_rows':len(test),
                    'test_rows_matching_training_input':int(match.sum())})
op = pd.DataFrame(overlap)
op.to_csv(OUT/'feature_overlap_by_fold.csv',index=False)
op.groupby('protocol')[['test_rows','test_rows_matching_training_input']].sum().to_csv(OUT/'feature_overlap_by_protocol.csv')

# Recheck predictive summaries from the stored per-observation errors.
scores = []
for (protocol, model), row in pd.read_csv(r/'metrics_overall.csv').set_index(['protocol','model']).iterrows():
    z = p[p.protocol==protocol]; err = z.y-z[model]
    vals = {'mae':err.abs().mean(), 'rmse':np.sqrt((err**2).mean()),
            'r2':1-(err**2).sum()/((z.y-z.y.mean())**2).sum(),
            'macro_mae':err.abs().groupby(z.family).mean().mean()}
    for key,value in vals.items():
        np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-10)
    scores.append({'protocol':protocol, 'model':model, **vals})
pd.DataFrame(scores).to_csv(OUT/'recomputed_metrics.csv',index=False)

# Recompute primary retention curves, AURCs and the conditional bootstrap endpoint.
curve_rows=[]; areas=[]
for (protocol,family),z in p.groupby(['protocol','family'],sort=True):
    error=np.abs(z.y.to_numpy()-z.prediction.to_numpy())
    for method in ['distance','dispersion','g0','g1']:
        order=np.lexsort((z['sample'].to_numpy(dtype=str),z[method].to_numpy()))
        vals=[]
        for q in C['retention_grid']:
            k=max(1,int(np.ceil(q*len(z)-1e-12)))
            value=float(error[order[:k]].mean());vals.append(value)
            curve_rows.append({'protocol':protocol,'family':family,'method':method,
                'requested_retention':q,'accepted':k,'actual_retention':k/len(z),'mae':value})
        areas.append({'protocol':protocol,'family':family,'method':method,
                      'aurc':float(np.trapezoid(vals,C['retention_grid'])/.5)})
curves=pd.DataFrame(curve_rows)
pd.testing.assert_frame_equal(curves,pd.read_csv(r/'risk_retention.csv'),check_exact=False,rtol=1e-12,atol=1e-10)
area=pd.DataFrame(areas)
paired=area[area.protocol=='family'].pivot(index='family',columns='method',values='aurc')
paired['delta_g0_minus_g1']=paired.g0-paired.g1
saved_pair=pd.read_csv(r/'primary_paired_effects.csv').set_index('family')
pd.testing.assert_frame_equal(paired,saved_pair,check_exact=False,check_names=False,rtol=1e-12,atol=1e-10)
delta=paired.delta_g0_minus_g1.to_numpy()
bootstrap=np.random.default_rng(C['seed']).choice(delta,(C['bootstrap_ci'],len(delta)),replace=True).mean(axis=1)
primary={'mean_delta_aurc':float(delta.mean()),'median_delta_aurc':float(np.median(delta)),
         'ci95_family_bootstrap':np.quantile(bootstrap,[.025,.975]).tolist(),
         'families_improved':int((delta>0).sum())}
original=json.loads((r/'primary_result.json').read_text())
for key,value in primary.items():np.testing.assert_allclose(value,original[key],rtol=1e-12,atol=1e-10)
(OUT/'recomputed_primary.json').write_text(json.dumps(primary,indent=2))

# Verify the post-review summary without refitting (a separate rerun log is supplied).
ar = pd.read_csv(WORK/'revision/ablation_predictions.csv')
base = p[p.protocol.isin(['family','grade'])].copy();base['variant']='original_median'
ar = pd.concat([ar,base[['protocol','variant','sample','y','prediction','family']]])
cc_ids = set(a.loc[a[F].notna().all(axis=1),'heat_id'])
summary = []
for population in ['all','complete_common']:
    z = ar.copy()
    if population=='complete_common':
        z = z[z['sample'].isin(cc_ids)]
        eligible=ar[ar.variant=='complete_case'][['protocol','sample']].drop_duplicates()
        z=z.merge(eligible,on=['protocol','sample'],how='inner')
    for (protocol,variant), v in z.groupby(['protocol','variant']):
        e=(v.y-v.prediction).abs()
        summary.append(dict(protocol=protocol,population=population,variant=variant,n=len(v),
                            families=v.family.nunique(),mae=e.mean(),macro_mae=e.groupby(v.family).mean().mean()))
summary=pd.DataFrame(summary)
saved=pd.read_csv(WORK/'revision/ablation_summary.csv')
pd.testing.assert_frame_equal(summary,saved,check_exact=False,rtol=1e-12,atol=1e-10)

sg = pd.DataFrame({'signature':sig, 'grade':raw.grade_id, 'family':raw.steel_family, 'source':raw.data_tier})
signatures = sg.groupby('signature').agg(n=('grade','size'),grades=('grade','nunique'),families=('family','nunique'),sources=('source','nunique'))
fam = a.groupby('steel_family').agg(records=('heat_id','size'),grades=('grade_id','nunique'),sources=('data_tier','nunique'))
cc = a[a[F].notna().all(axis=1)]
trains = [set(raw.heat_id)-set(z['sample']) for _,z in p[p.protocol=='family'].groupby('outer_fold')]
jac=[len(x&y)/len(x|y) for x,y in itertools.combinations(trains,2)]
report = {
    'status':'PASS', 'records':len(a), 'grades':int(a.grade_id.nunique()), 'families':len(fam),
    'units':int(a.analysis_unit.nunique()), 'target_summary':{'min':float(a.tensile_strength.min()),
    'max':float(a.tensile_strength.max()),'median':float(a.tensile_strength.median()),'sample_sd':float(a.tensile_strength.std(ddof=1))},
    'raw_sha256':hashlib.sha256((WORK/'dataset/steelbench_core_open.csv').read_bytes()).hexdigest(),
    'analysis_sha256':hashlib.sha256((WORK/'real_run/steelbench_core_open_analysis.csv').read_bytes()).hexdigest(),
    'analysis_unit_reconstruction_exact':True,'raw_columns_unchanged':True,
    'boundaries_checked':len(boundaries),'outer_folds':int(p.groupby(['protocol','outer_fold']).ngroups),
    'exact_repeated_signatures':int((signatures.n>1).sum()),'rows_in_repeated_signatures':int(signatures.loc[signatures.n>1,'n'].sum()),
    'cross_grade_signatures':int((signatures.grades>1).sum()),'rows_in_cross_grade_signatures':int(signatures.loc[signatures.grades>1,'n'].sum()),
    'cross_family_or_source_signatures':int(((signatures.families>1)|(signatures.sources>1)).sum()),
    'rows_in_cross_family_or_source_signatures':int(signatures.loc[(signatures.families>1)|(signatures.sources>1),'n'].sum()),
    'complete_case_records':len(cc),'complete_case_grades':int(cc.grade_id.nunique()),
    'complete_case_families':cc.steel_family.value_counts().to_dict(),'complete_case_sources':cc.data_tier.value_counts().to_dict(),
    'families_lt10':int((fam.records<10).sum()),'families_lt20':int((fam.records<20).sum()),'singletons':int((fam.records==1).sum()),
    'single_source_families':int((fam.sources==1).sum()),
    'training_jaccard_min_median_max':[float(min(jac)),float(np.median(jac)),float(max(jac))],
    'environment':{'python':platform.python_version(),**{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib']}}
}
(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
print(op.groupby('protocol')[['test_rows','test_rows_matching_training_input']].sum().to_string())
