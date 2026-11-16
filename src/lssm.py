"""Stage 1: linear Gaussian state space model of daily physiological recovery.

Latent x_t = recovery on the morning of day t (1 dimension, per person z scores).

Transition (night t-1 -> morning t), driven by known inputs:
    x_t = phi * x_{t-1} + b_sleep * sleep_z[t] + b_eff * eff_z[t] + b_steps * steps_z[t-1] + w,  w ~ N(0, q)
    (sleep on date t is the night that ENDED on morning t, Fitbit dateOfSleep convention)

Observations:
    morning t : rmssd_z[t]   = c_hrv * x_t + v_hrv     (nightly HRV, available at wake time)
    end of t  : rhr_z[t]     = -1    * x_t + v_rhr     (daily resting HR; loading fixed to -1 for scale and sign)

Filtering is causal. The "morning" posterior (after the HRV update, before the resting HR
update of the same day) is what a mood report on day t is allowed to see.
Missing inputs are set to 0 (the person mean); missing observations skip the update.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize

P = 'data/processed/'


def load():
    panel = pd.read_csv(P + 'daily_panel.csv')
    split = pd.read_csv(P + 'splits.csv')
    panel = panel.merge(split[['id', 'test_from_day']], on='id')
    panel['train'] = panel.day < panel.test_from_day
    return panel


def unpack(th):
    phi = np.tanh(th[0])                       # keep |phi| < 1 (stable, decaying memory)
    b = th[1:4]
    c_hrv = th[4]
    q = np.exp(th[5]); r_hrv, r_rhr = 0.1 + np.exp(th[6:8])   # floor on observation noise avoids a degenerate latent that copies one signal
    return phi, b, c_hrv, q, r_hrv, r_rhr


def run_filter(g, th, train_only=False):
    phi, b, c_hrv, q, r_hrv, r_rhr = unpack(th)
    sleep = g.minutesAsleep_z.fillna(0).values
    eff = g.sleep_efficiency_z.fillna(0).values
    steps_prev = np.r_[0.0, g.steps_z.fillna(0).values[:-1]]
    hrv, rhr = g.rmssd_z.values, g.resting_hr_z.values
    use = g.train.values if train_only else np.ones(len(g), bool)
    m, v = 0.0, 1.0
    ll = 0.0
    morn_m, morn_v = np.empty(len(g)), np.empty(len(g))
    for t in range(len(g)):
        # predict
        m = phi * m + b[0] * sleep[t] + b[1] * eff[t] + b[2] * steps_prev[t]
        v = phi * phi * v + q
        # morning update with last night's HRV
        if not np.isnan(hrv[t]):
            s = c_hrv * c_hrv * v + r_hrv; e = hrv[t] - c_hrv * m
            if use[t]: ll += -0.5 * (np.log(2 * np.pi * s) + e * e / s)
            k = v * c_hrv / s; m += k * e; v *= (1 - k * c_hrv)
        morn_m[t], morn_v[t] = m, v
        # end of day update with the day's resting HR
        if not np.isnan(rhr[t]):
            s = v + r_rhr; e = rhr[t] + m
            if use[t]: ll += -0.5 * (np.log(2 * np.pi * s) + e * e / s)
            k = -v / s; m += k * e; v *= (1 + k)
    return ll, morn_m, morn_v


def fit(panel):
    groups = [g.sort_values('day') for _, g in panel.groupby('id')]
    def nll(th):
        return -sum(run_filter(g, th, train_only=True)[0] for g in groups)
    th0 = np.array([np.arctanh(0.5), 0.1, 0.1, -0.1, 0.5, np.log(0.3), np.log(0.7), np.log(0.7)])
    res = minimize(nll, th0, method='L-BFGS-B')
    return res, groups


if __name__ == '__main__':
    panel = load()
    res, groups = fit(panel)
    phi, b, c_hrv, q, r_hrv, r_rhr = unpack(res.x)
    print('converged', res.success, '| train negative log likelihood', round(res.fun, 1))
    print(f'phi (day to day persistence) {phi:.3f}  -> half life {np.log(0.5)/np.log(phi):.2f} days')
    print(f'b_sleep {b[0]:.3f}  b_efficiency {b[1]:.3f}  b_steps_prev_day {b[2]:.3f}')
    print(f'c_hrv {c_hrv:.3f} (rhr loading fixed at -1)  q {q:.3f}  r_hrv {r_hrv:.3f}  r_rhr {r_rhr:.3f}')
    out = []
    for g in groups:
        _, mm, vv = run_filter(g, res.x)
        out.append(pd.DataFrame({'id': g.id.values, 'day': g.day.values, 'x_morning': mm, 'x_var': vv}))
    pd.concat(out).to_csv(P + 'latent_linear.csv', index=False)
    np.save(P + 'lssm_params.npy', res.x)
