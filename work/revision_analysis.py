from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
sys.path.insert(0,str(Path('work/code/steelbench_research').resolve()))
import experiment as ex
O=Path('work/revision');O.mkdir(exist_ok=True)
C=json.loads(Path('work/real_run/steelbench_config_real_fast.json').read_text())
D,_,_=ex.load_data('work/real_run/steelbench_core_open_analysis.csv',C)
P=pd.read_csv('work/real_run/results_uts_fast/predictions.csv')
F=C['features'];complete=D[F].notna().all(axis=1)
stats=[];over=[];trainsets=[]
for protocol in C['protocols']:
 for fold,g in P[P.protocol==protocol].groupby('outer_fold'):
  te=D[D['sample'].isin(g['sample'])];tr=D[~D['sample'].isin(g['sample'])]
  for role,z in [('train',tr),('test',te)]:
   stats.append(dict(protocol=protocol,fold=int(fold),role=role,records=len(z),grades=z.grade.nunique(),families=z.family.nunique(),units=z.unit.nunique()))
  if protocol=='family':trainsets.append(set(tr['sample']))
S=pd.DataFrame(stats);S.to_csv(O/'split_statistics.csv',index=False)
for a,b in itertools.combinations(trainsets,2):over.append(len(a&b)/len(a|b))
family=D.groupby('family').agg(n=('y','size'),grades=('grade','nunique'),sources=('source','nunique'),uts_mean=('y','mean'),uts_sd=('y','std'))
for f in F:family[f+'_missing']=D.groupby('family')[f].apply(lambda x:x.isna().mean())
family.to_csv(O/'family_audit.csv')
pd.crosstab(D.family,D.source).to_csv(O/'family_source.csv')
audit=dict(complete_records=int(complete.sum()),complete_grades=int(D[complete].grade.nunique()),complete_families=int(D[complete].family.nunique()),complete_sources=D[complete].source.value_counts().to_dict(),families_lt10=int((family.n<10).sum()),families_lt20=int((family.n<20).sum()),single_source_families=int((family.sources==1).sum()),jaccard_min=min(over),jaccard_median=float(np.median(over)),jaccard_max=max(over))
(O/'audit.json').write_text(json.dumps(audit,indent=2))
print(audit,flush=True)
# Post-review predictive ablation, fixed original outer test sets; no attribution claims.
rows=[]
for protocol in ['family','grade']:
 for fold,g in P[P.protocol==protocol].groupby('outer_fold'):
  te=D[D['sample'].isin(g['sample'])];tr=D[~D['sample'].isin(g['sample'])]
  tecc=te[te[F].notna().all(axis=1)]
  for variant in ['median_indicator','drop_four','complete_case']:
   feats=[f for f in F if f not in ['Al','Cu','austenitize_T','temper_T']] if variant=='drop_four' else F
   train=tr[tr[F].notna().all(axis=1)] if variant=='complete_case' else tr
   test=tecc if variant=='complete_case' else te
   if len(test)==0 or len(train)<5 or train.grade.nunique()<2:continue
   seed=2026+(20000 if protocol=='family' else 10000)+100*int(fold)
   rng=np.random.default_rng(seed);pred=[];groups=train.grade.unique()
   for b in range(5):
    chosen=rng.choice(groups,len(groups),replace=True)
    ix=np.concatenate([np.flatnonzero(train.grade.to_numpy()==v) for v in chosen])
    m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=variant=='median_indicator'),GradientBoostingRegressor(n_estimators=120,max_depth=3,learning_rate=.05,min_samples_leaf=3,random_state=seed+b))
    m.fit(train.iloc[ix][feats],train.iloc[ix].y);pred.append(m.predict(test[feats]))
   for sid,y,yp,fam in zip(test['sample'],test.y,np.mean(pred,axis=0),test.family):rows.append(dict(protocol=protocol,variant=variant,sample=sid,y=y,prediction=yp,family=fam,fold=fold))
  print('fold',fold,'done',flush=True)
R=pd.DataFrame(rows);R.to_csv(O/'ablation_predictions.csv',index=False)
base=P[P.protocol.isin(['family','grade'])].copy();base['variant']='original_median';R=pd.concat([R,base[['protocol','variant','sample','y','prediction','family']]])
res=[]
for population in ['all','complete_common']:
 z=R if population=='all' else R[R['sample'].isin(D[complete]['sample'])]
 if population=='complete_common':
  eligible=R[R.variant=='complete_case'][['protocol','sample']].drop_duplicates()
  z=z.merge(eligible,on=['protocol','sample'],how='inner')
 for (protocol,variant),g in z.groupby(['protocol','variant']):
  e=(g.y-g.prediction).abs();res.append(dict(protocol=protocol,population=population,variant=variant,n=len(g),families=g.family.nunique(),mae=e.mean(),macro_mae=e.groupby(g.family).mean().mean()))
pd.DataFrame(res).to_csv(O/'ablation_summary.csv',index=False)
print(pd.DataFrame(res).to_string(index=False))
