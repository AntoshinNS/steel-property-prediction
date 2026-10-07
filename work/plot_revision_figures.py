from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'work/real_run/results_uts_fast'
O=ROOT/'work/revision/figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7,'font.family':'DejaVu Sans'})
# Fig. 2: fixed-prediction protocol comparison; one chart per figure.
m=pd.read_csv(R/'metrics_overall.csv').query("model=='prediction'").set_index('protocol').loc[['random','grade','family','source']]
fig,ax=plt.subplots(figsize=(3.4,2.12),layout='constrained')
x=np.arange(4); w=.36
ax.bar(x-w/2,m.mae,w,label='Sample-weighted MAE')
ax.bar(x+w/2,m.macro_mae,w,label='Macro-family MAE')
ax.set_xticks(x,['Random','Grade','Family','Source tier'])
ax.set_ylabel('Mean absolute error (MPa)');ax.set_ylim(0,254)
ax.legend(loc='upper left',frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'fig2_protocol_error.png',dpi=360);fig.savefig(O/'fig2_protocol_error.pdf');plt.close(fig)
# Fig. 3: averaging is across the same 17 family curves at requested retention.
r=pd.read_csv(R/'risk_retention.csv').query("protocol=='family'")
fig,ax=plt.subplots(figsize=(3.4,2.25),layout='constrained')
for method,label,marker in [('distance','Distance','o'),('dispersion','Dispersion','s'),('g0','g0: distance + dispersion','^'),('g1','g1: + attribution spread','D')]:
 z=r[r.method==method].groupby('requested_retention').mae.mean()
 ax.plot(z.index,z.values,label=label,marker=marker,markersize=2.4,linewidth=1.1)
ax.set_xlabel('Requested retained fraction q');ax.set_ylabel('Macro-family MAE (MPa)')
ax.set_xticks(np.arange(.5,1.01,.1));ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',bbox_to_anchor=(0,1.38),frameon=False,handlelength=1.5,ncol=2,columnspacing=.8)
fig.savefig(O/'fig3_retention.png',dpi=360);fig.savefig(O/'fig3_retention.pdf');plt.close(fig)
# Fig. 4: all families, not a favorable subset; zero remains an actual zero.
p=pd.read_csv(R/'primary_paired_effects.csv').sort_values('delta_g0_minus_g1',ascending=False,kind='stable')
fig,ax=plt.subplots(figsize=(3.4,3.25),layout='constrained')
y=np.arange(len(p));ax.barh(y,p.delta_g0_minus_g1,height=.72)
ax.set_yticks(y,[v.replace('_',' ') for v in p.family]);ax.invert_yaxis()
ax.axvline(0,linewidth=.65,linestyle=':')
ax.set_xlabel('AURC(g0) − AURC(g1) (MPa)');ax.set_xlim(-11,35)
ax.set_xticks([-10,0,10,20,30]);ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'fig4_family_effects.png',dpi=360);fig.savefig(O/'fig4_family_effects.pdf');plt.close(fig)
