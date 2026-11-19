"""LifeSnaps recovery explorer.

Replays a participant's nights one morning at a time, as if a wearable were syncing, and shows the
recovery states estimated by the state space model, the hidden Markov regime, and the mood reports.
Run from the repository root:  streamlit run ui/app.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

D = Path(__file__).resolve().parents[1] / "data" / "processed" / "d2"
st.set_page_config(page_title="LifeSnaps recovery explorer", layout="wide")


@st.cache_data
def load():
    days = pd.read_parquet(D / "daily_states.parquet")
    reps = pd.read_parquet(D / "mood_reports.parquet")
    prm = json.load(open(D / "params.json"))
    return days, reps, prm


if not (D / "params.json").exists():
    st.error(f"Model outputs not found in {D}. Run the export cell of notebooks/lifesnaps_analysis.ipynb and unzip d2_export.zip there.")
    st.stop()
days, reps, prm = load()
days["date"] = pd.to_datetime(days["date"])
reps["date"] = pd.to_datetime(reps["date"])
S = len(prm["phiS"])
CS = np.array(prm["CS"]); ALL = prm["all"]; phiS = np.array(prm["phiS"]); BS = np.array(prm["BS"])

# Name each hidden state by the features that load on it most; orient so that higher means worse physiology
sign = np.ones(S)
names = []
for s in range(S):
    load_s = pd.Series(CS[:, s], index=ALL).drop(["valence", "arousal"], errors="ignore")
    if s == 0 and load_s["hr_nadir"] < 0:
        sign[s] = -1
    top = load_s.abs().sort_values(ascending=False).index[:2]
    names.append(f"state {s + 1} ({', '.join(top)})")
for s in range(S):
    days[f"st{s + 1}"] = sign[s] * days[f"state{s + 1}"]

st.title("LifeSnaps recovery explorer")
st.caption("Night physiology from a Fitbit, summarized as hidden recovery states. Data replayed one morning at a time "
           "from the LifeSnaps study; a live Google Health API connection is future work.")

with st.sidebar:
    st.header("Participant")
    ids = sorted(days.id.unique(), key=lambda i: -days[days.id == i]["hr_nadir"].notna().sum())
    pid = st.selectbox("Participant", ids, format_func=lambda i: f"…{i[-6:]}  ({days[days.id == i].hr_nadir.notna().sum()} nights)")
    g = days[days.id == pid].sort_values("date").reset_index(drop=True)
    day_ix = st.slider("Replay up to morning", 1, len(g), len(g), help="Shows only what a syncing watch would have delivered by this morning.")
    st.markdown("---")
    st.markdown("**Model**: linear Gaussian state space model, Kalman filter, "
                f"{S} hidden states selected by BIC; hidden Markov model with {len(prm['hmm_theta'])} regimes.")

gv = g.iloc[:day_ix]
today = gv.iloc[-1]
tab1, tab2, tab3 = st.tabs(["This morning", "Timeline", "Model results"])

with tab1:
    st.subheader(f"Morning of {today.date:%a %d %b %Y}")
    if pd.isna(today.hr_nadir):
        st.info("No night recorded for this date. The states below are carried forward from the previous days by the model.")
    cols = st.columns(S + 1)
    for s in range(S):
        val = today[f"st{s + 1}"]; prev = gv.iloc[-2][f"st{s + 1}"] if len(gv) > 1 else np.nan
        cols[s].metric(names[s].split(" (")[0].title(), f"{val:+.2f}",
                       None if pd.isna(prev) else f"{val - prev:+.2f} vs yesterday", delta_color="inverse")
    cols[S].metric("Regime (HMM)", f"S{int(today.hmm_state)}")
    st.caption("States are in units of the person's own variability; 0 is their usual. "
               "Higher means a higher heart rate floor, lower HRV, or more disrupted sleep.")

    st.markdown("#### What if tonight is different?")
    c1, c2 = st.columns(2)
    sleep_z = c1.slider("Tonight's sleep vs your usual (SD)", -2.0, 2.0, 0.0, 0.25)
    steps_z = c2.slider("Today's steps vs your usual (SD)", -2.0, 2.0, 0.0, 0.25)
    k = np.arange(0, 6)
    fig, ax = plt.subplots(figsize=(8, 3))
    for s in range(S):
        x0 = today[f"st{s + 1}"]
        u = sign[s] * (BS[s, 0] * sleep_z + BS[s, 1] * steps_z)
        path = [x0] + [phiS[s] ** (j - 1) * (phiS[s] * x0 + u) for j in range(1, 6)]
        ax.plot(k, path, marker="o", label=names[s])
    ax.axhline(0, c="k", lw=.5); ax.set_xlabel("days ahead (tonight's choices apply to day 1)"); ax.set_ylabel("state")
    ax.legend(fontsize=7); fig.tight_layout(); st.pyplot(fig)
    st.caption("Forecast from the fitted state equation, with the effect of tonight's sleep and today's steps entering on day 1. "
               "The fitted persistence is short (half life under one day), so the effect fades quickly.")

    st.markdown("#### Mood reported that day")
    r_today = reps[(reps.id == pid) & (reps.date == today.date)]
    if len(r_today):
        st.write(", ".join(f"{t:%H:%M} {m}" for t, m in zip(r_today.t, r_today.MOOD)))
    else:
        st.write("No mood reports for this day.")
    st.caption("Across six analyses, these states did not predict reported mood beyond the person's own baseline; "
               "the explorer shows mood next to physiology rather than forecasting it.")

with tab2:
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for s in range(S):
        ax[0].plot(gv.date, gv[f"st{s + 1}"], label=names[s], lw=1.2)
    ax[0].axhline(0, c="k", lw=.5); ax[0].legend(fontsize=7, ncol=S); ax[0].set_ylabel("hidden states")
    reg = gv.hmm_state.to_numpy()
    ax[1].scatter(gv.date, reg, c=reg, cmap="tab10", s=18); ax[1].set_ylabel("HMM regime")
    ax[1].set_yticks(range(1, len(prm["hmm_theta"]) + 1))
    rr = reps[(reps.id == pid) & (reps.date <= today.date)]
    moods = prm["cats"]; ypos = {m: i for i, m in enumerate(moods)}
    ax[2].scatter(rr.t, rr.MOOD.map(ypos), s=14, c="tab:purple")
    ax[2].set_yticks(range(len(moods)), moods, fontsize=7); ax[2].set_ylabel("mood report")
    short = gv[gv.minutesAsleep < -1]
    for a in ax:
        for d in short.date: a.axvspan(d - pd.Timedelta("12h"), d + pd.Timedelta("12h"), color="tab:red", alpha=.08)
    fig.tight_layout(); st.pyplot(fig)
    st.caption("Red bands mark short nights (more than 1 SD below the person's usual sleep).")

with tab3:
    st.markdown("#### Held out mood classification (last 30% of each person's reports)")
    st.dataframe(pd.concat([pd.DataFrame(prm["res_lr"]), pd.DataFrame(prm["res_x"]).iloc[1:]], ignore_index=True), hide_index=True)
    st.caption("Chance balanced accuracy for 7 classes is 0.143.")
    st.markdown("#### Hidden state loadings")
    st.dataframe(pd.DataFrame(CS * sign, index=ALL, columns=[n.split(" (")[0] for n in names]).round(2))
    st.markdown("#### HMM regimes: mood probabilities")
    st.dataframe(pd.DataFrame(prm["hmm_theta"], columns=moods, index=[f"S{i + 1}" for i in range(len(prm["hmm_theta"]))]).T.round(2))
    st.markdown("#### Cleaning funnel")
    st.dataframe(pd.DataFrame(prm["funnel"]), hide_index=True)
