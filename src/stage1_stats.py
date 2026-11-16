import pandas as pd, numpy as np
from sklearn.metrics import balanced_accuracy_score as BA
P='data/processed/'
pr=pd.read_csv(P+'test_predictions.csv'); panel=pd.read_csv(P+'daily_panel.csv')
A,B='time + previous mood','+ linear SSM recovery state'
C='+ last night/yesterday physiology'
# disruption flag: short night (sleep z < -1) or very active previous day (steps z > 1) within the last 3 days
pn=panel.sort_values(['id','day']).copy()
pn['event']=((pn.minutesAsleep_z< -1)|(pn.steps_z.groupby(pn.id).shift(1)>1)).astype(int)
pn['recent']=pn.groupby('id').event.transform(lambda x: x.rolling(3,min_periods=1).max())
pr=pr.merge(pn[['id','day','recent']],on=['id','day'],how='left')
pr['recent']=pr.recent.fillna(0).astype(int)
def stats(df):
    o={}
    for k,lab in [('carry','prev'),('base',A+'|pred'),('phys',C+'|pred'),('ssm',B+'|pred')]:
        o[k]=BA(df.mood,df[lab])
    o['ll_base']=-np.log(df[A+'|p_true']).mean(); o['ll_ssm']=-np.log(df[B+'|p_true']).mean()
    return o
print('reports by stratum:',pr.recent.value_counts().to_dict())
for name,df in [('all test reports',pr),('within 3 days of a disruption',pr[pr.recent==1]),('ordinary days',pr[pr.recent==0])]:
    o=stats(df); print(f'\n{name} (n={len(df)})'); print({k:round(v,3) for k,v in o.items()})
# participant bootstrap of SSM minus base
rng=np.random.default_rng(0); ids=pr.id.unique(); out=[]
for _ in range(2000):
    samp=pd.concat([pr[pr.id==i] for i in rng.choice(ids,len(ids))])
    o=stats(samp); out.append((o['ssm']-o['base'],o['ll_base']-o['ll_ssm'],o['ssm']-o['carry'],o['phys']-o['base']))
out=np.array(out)
for j,lab in enumerate(['balacc SSM minus time+prev','logloss improvement SSM vs time+prev','balacc SSM minus carry forward','balacc raw physiology minus time+prev']):
    lo,hi=np.percentile(out[:,j],[2.5,97.5]); print(f'{lab:42s} mean {out[:,j].mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')
