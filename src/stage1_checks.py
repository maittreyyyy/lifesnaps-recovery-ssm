import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
P='data/processed/'
panel=pd.read_csv(P+'daily_panel.csv').merge(pd.read_csv(P+'latent_linear.csv'),on=['id','day'])
th=np.load(P+'lssm_params.npy'); phi=np.tanh(th[0]); b_sleep=th[1]
# event locked: short night = sleep more than 1 SD below the person's mean
panel=panel.sort_values(['id','day'])
rows=[]
for pid,g in panel.groupby('id'):
    g=g.set_index('day')
    for d0 in g.index[g.minutesAsleep_z< -1]:
        for k in range(-2,6):
            if d0+k in g.index: rows.append((k,g.loc[d0+k,'x_morning'],g.loc[d0+k,'resting_hr_z'],g.loc[d0+k,'rmssd_z']))
ev=pd.DataFrame(rows,columns=['k','x','rhr','hrv']).groupby('k').agg(['mean','sem','count'])
print('short nights:',int((panel.minutesAsleep_z< -1).sum()))
print(ev['x'].round(3))
fig,ax=plt.subplots(1,3,figsize=(15,4))
for a,c,t in zip(ax,['x','rhr','hrv'],['Latent recovery state','Resting HR (z)','HRV RMSSD (z)']):
    a.errorbar(ev.index,ev[c]['mean'],yerr=1.96*ev[c]['sem'],marker='o',capsize=3)
    a.axvline(0,ls='--',c='gray'); a.axhline(0,c='k',lw=.5); a.set_title(t); a.set_xlabel('days relative to short night')
plt.tight_layout(); plt.savefig('results/event_locked_short_night.png',dpi=130)
k=np.arange(0,15); print('model implied half life of a sleep shock: %.1f days'%(np.log(.5)/np.log(phi)))
