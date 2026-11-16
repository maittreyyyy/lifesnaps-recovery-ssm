import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
P='data/processed/'
panel=pd.read_csv(P+'daily_panel.csv'); r=pd.read_csv(P+'reports.csv'); s=pd.read_csv(P+'splits.csv')
lat=pd.read_csv(P+'latent_linear.csv')
r=r.merge(s,on='id'); r['test']=r.day>=r.test_from_day
r['t']=pd.to_datetime(r.date)+pd.to_timedelta(r.hour,unit='h')
r['prev']=r.groupby('id').mood.shift(); r['gap']=(r.t-r.groupby('id').t.shift()).dt.total_seconds()/3600
r=r[r.prev.notna()&(r.gap<=36)].copy()
r['night_day']=r.day.where(r.hour>=6,r.day-1)
NIGHT=['minutesAsleep_z','sleep_efficiency_z','rmssd_z']; DAY=['resting_hr_z','steps_z']
night=panel[['id','day']+NIGHT].rename(columns={'day':'night_day'})
dayv=panel[['id','day']+DAY].copy(); dayv['day']+=1
# accumulation control: 5 day exponential moving averages, causal (shifted to exclude the report day)
ew=panel.sort_values(['id','day']).copy()
for k in ['minutesAsleep_z','resting_hr_z','steps_z']:
    ew[k+'_ewm']=ew.groupby('id')[k].transform(lambda x: x.shift(1).ewm(halflife=5,ignore_na=True).mean())
ewc=[c for c in ew if c.endswith('_ewm')]
r=(r.merge(night,on=['id','night_day'],how='left').merge(dayv,on=['id','day'],how='left')
    .merge(ew[['id','day']+ewc],on=['id','day'],how='left')
    .merge(lat.rename(columns={'day':'night_day'}),on=['id','night_day'],how='left'))
def prep(cols):
    out=r[cols].copy()
    for c in cols: out[c+'_miss']=out[c].isna().astype(int); out[c]=out[c].fillna(0)
    return out
base=pd.concat([pd.get_dummies(r.slot,prefix='slot'),pd.get_dummies(r.prev,prefix='prev')],axis=1).astype(float)
sets={'time + previous mood':base,
      '+ last night/yesterday physiology':pd.concat([base,prep(NIGHT+DAY)],axis=1),
      '+ 5 day moving averages (control)':pd.concat([base,prep(ewc)],axis=1),
      '+ linear SSM recovery state':pd.concat([base,prep(['x_morning','x_var'])],axis=1)}
tr,te=~r.test.values,r.test.values
preds=r.loc[te,['id','day','mood','prev']].copy()
print(f'{"model":38s} balacc  macroF1  logloss')
print(f'{"carry forward":38s} {balanced_accuracy_score(r.mood[te],r.prev[te]):.3f}   {f1_score(r.mood[te],r.prev[te],average="macro"):.3f}      -')
for name,X in sets.items():
    m=LogisticRegression(max_iter=3000,class_weight='balanced').fit(X[tr],r.mood[tr])
    p=m.predict(X[te]); pr=m.predict_proba(X[te])
    preds[name+'|pred']=p
    preds[name+'|p_true']=pr[np.arange(len(p)),[list(m.classes_).index(y) for y in r.mood[te]]]
    print(f'{name:38s} {balanced_accuracy_score(r.mood[te],p):.3f}   {f1_score(r.mood[te],p,average="macro"):.3f}    {log_loss(r.mood[te],pr,labels=m.classes_):.3f}')
    if 'SSM' in name:
        coef=pd.Series(m.coef_[:,list(X.columns).index('x_morning')],index=m.classes_).round(3)
        print('   recovery state coefficient by mood:',coef.to_dict())

preds.to_csv(P+'test_predictions.csv',index=False)
