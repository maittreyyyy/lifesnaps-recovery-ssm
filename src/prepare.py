"""Build model ready tables for the 48 person cohort.

Outputs (data/processed/):
  daily_panel.csv   one row per person per calendar day, gaps kept as NaN
  reports.csv       one row per mood report, linked to its day index
  splits.csv        chronological split per person (first 70% of labeled days train)
"""
import pandas as pd, numpy as np
from pathlib import Path

MOOD = ['ALERT','HAPPY','NEUTRAL','RESTED/RELAXED','SAD','TENSE/ANXIOUS','TIRED']
INPUTS = ['minutesAsleep','sleep_efficiency','resting_hr','steps','rmssd']
OUT = Path('data/processed'); OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv('data/raw/daily_fitbit_sema_df_unprocessed.csv', low_memory=False); d['date'] = pd.to_datetime(d['date'])
h = pd.read_csv('data/raw/hourly_fitbit_sema_df_unprocessed.csv', low_memory=False, usecols=['id','date','hour']+MOOD)
h['date'] = pd.to_datetime(h['date'])

lab_days = d[d[MOOD].notna().all(axis=1)].groupby('id').size()
cohort = lab_days[lab_days >= 14].index

# daily panel on a continuous calendar per person
d = d[d.id.isin(cohort)].copy()
d.loc[d.rmssd <= 0, 'rmssd'] = np.nan            # zero RMSSD is not physiological
panel = []
for pid, g in d.groupby('id'):
    idx = pd.date_range(g.date.min(), g.date.max(), freq='D')
    g = g.set_index('date').reindex(idx)
    g['id'] = pid
    g['day'] = np.arange(len(g))
    panel.append(g[['id','day']+INPUTS].rename_axis('date').reset_index())
panel = pd.concat(panel, ignore_index=True)
# per person z scores (keeps between person level differences out of the dynamics)
for k in INPUTS:
    mu = panel.groupby('id')[k].transform('mean'); sd = panel.groupby('id')[k].transform('std')
    panel[k+'_z'] = (panel[k] - mu) / sd
    panel[k+'_obs'] = panel[k].notna().astype(int)
panel.to_csv(OUT/'daily_panel.csv', index=False)

# mood reports, one category each
r = h[h.id.isin(cohort) & h[MOOD].notna().all(axis=1)].copy()
r['mood'] = r[MOOD].idxmax(axis=1)
r['slot'] = pd.cut(r.hour, [-1,5,13,18,24], labels=['night','morning','afternoon','evening'])
r = r.merge(panel[['id','date','day']], on=['id','date'], how='left')
r = r.sort_values(['id','date','hour'])[['id','date','day','hour','slot','mood']]
r.to_csv(OUT/'reports.csv', index=False)

# chronological split on labeled days
rows = []
for pid, g in r.groupby('id'):
    days = np.sort(g.day.unique()); cut = days[int(np.floor(0.7*len(days)))]
    rows.append({'id': pid, 'test_from_day': int(cut), 'n_label_days': len(days)})
pd.DataFrame(rows).to_csv(OUT/'splits.csv', index=False)

print('people', len(cohort), '| panel days', len(panel), '| reports', len(r))
print('panel input coverage:', {k: round(panel[k].notna().mean(),2) for k in INPUTS})
