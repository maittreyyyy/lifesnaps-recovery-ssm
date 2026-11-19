"""Recovery tracker built on the LifeSnaps state space model.

Each morning shows how recovered the body looks after last night, compared with the person's own usual,
and how tonight's sleep and today's activity are expected to change it. Data are replayed one morning at a
time from the LifeSnaps study, as if a watch were syncing.
Run from the repository root:  streamlit run ui/app.py
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

D = Path(__file__).resolve().parents[1] / "data" / "processed" / "d2"
st.set_page_config(page_title="Recovery tracker", layout="wide")

if not (D / "params.json").exists():
    st.error(f"Model outputs not found in {D}. Run the export cell of notebooks/lifesnaps_analysis.ipynb "
             "and unzip d2_export.zip there.")
    st.stop()


@st.cache_data
def load():
    days = pd.read_parquet(D / "daily_states.parquet")
    reps = pd.read_parquet(D / "mood_reports.parquet")
    prm = json.load(open(D / "params.json"))
    days["date"] = pd.to_datetime(days["date"]); reps["date"] = pd.to_datetime(reps["date"])
    return days, reps, prm


days, reps, prm = load()
CS = np.array(prm["CS"]); ALL = prm["all"]; phiS = np.array(prm["phiS"]); BS = np.array(prm["BS"])
S = len(phiS)

# Orient the states: strain (state 1) and disruption (state 3) are higher when the night was worse.
i_nadir, i_wake = ALL.index("hr_nadir"), ALL.index("wake_per_h")
strain_sign = 1.0 if CS[i_nadir, 0] > 0 else -1.0
dis_idx = int(np.argmax(np.abs(CS[i_wake, :])))
dis_sign = 1.0 if CS[i_wake, dis_idx] > 0 else -1.0
days["strain"] = strain_sign * days["state1"]
days["disruption"] = dis_sign * days[f"state{dis_idx + 1}"]


def score(strain):
    """Recovery score 0 to 100; 50 is the person's usual night, higher is better recovered."""
    return 100 * (1 - 0.5 * (1 + math.erf(strain / math.sqrt(2))))


def label(sc):
    if sc >= 65: return "Well recovered", "#2e7d32"
    if sc >= 40: return "About usual", "#9e9e9e"
    return "Under recovered", "#c62828"


days["score"] = days.strain.map(score)
LEVELS = {"Much less": -2.0, "Less": -1.0, "Usual": 0.0, "More": 1.0, "Much more": 2.0}

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Your data")
    counts = days.groupby("id").hr_nadir.count().sort_values(ascending=False)
    pid = st.selectbox("Person", counts.index, format_func=lambda i: f"Participant {i[-6:]} ({counts[i]} nights)")
    g = days[days.id == pid].sort_values("date").reset_index(drop=True)
    day_ix = st.slider("Show the morning of day", 1, len(g), len(g),
                       help="Replays the study one morning at a time, as if your watch had just synced.")
    st.caption(f"{g.date.iloc[day_ix - 1]:%A %d %B %Y}")
    st.markdown("---")
    st.caption("Demo data from the LifeSnaps study (Fitbit Sense, 2021 to 2022). "
               "A live connection to your own watch is planned.")

gv = g.iloc[:day_ix]; today = gv.iloc[-1]; yest = gv.iloc[-2] if len(gv) > 1 else None

tab1, tab2, tab3 = st.tabs(["Today", "Your trend", "How it works"])

# ---------------- today ----------------
with tab1:
    sc = today.score; name, color = label(sc)
    st.markdown(f"### Good morning. Here is how your body recovered last night.")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        st.markdown(f"<div style='font-size:15px;opacity:.7'>Recovery score</div>"
                    f"<div style='font-size:64px;font-weight:700;color:{color};line-height:1.1'>{sc:.0f}</div>"
                    f"<div style='font-size:20px;color:{color}'>{name}</div>", unsafe_allow_html=True)
        if yest is not None:
            st.caption(f"{sc - yest.score:+.0f} compared with yesterday. 50 is your usual night.")
    with c2:
        h = "higher" if today.hr_nadir > 0.5 else "lower" if today.hr_nadir < -0.5 else "about usual"
        st.metric("Lowest heart rate overnight", h.capitalize() if not np.isnan(today.hr_nadir) else "No data",
                  help="A lower overnight heart rate floor usually means better recovery.")
        hv = today.rmssd_mean
        st.metric("Heart rate variability", ("Higher" if hv > 0.5 else "Lower" if hv < -0.5 else "About usual")
                  if not np.isnan(hv) else "No data", help="Higher HRV usually means better recovery.")
    with c3:
        d = today.disruption
        st.metric("Sleep disruption", "More than usual" if d > 0.5 else "Less than usual" if d < -0.5 else "About usual",
                  help="Based on how often you woke up and how much deep sleep you got.")
        sl = today.minutesAsleep
        st.metric("Sleep length", ("Longer" if sl > 0.5 else "Shorter" if sl < -0.5 else "About usual")
                  if not np.isnan(sl) else "No data")

    if np.isnan(today.hr_nadir):
        st.info("Your watch did not record last night, so this score is carried forward from recent days.")
    elif sc < 40:
        st.warning("Your heart rate stayed higher and your HRV lower than usual overnight. This often follows a short "
                   "night, a hard day, alcohol, stress, or the start of an illness. An easier day may help.")
    elif sc >= 65:
        st.success("Your heart settled well overnight. A good day for a harder effort if you want one.")

    st.markdown("#### Plan tonight")
    p1, p2 = st.columns(2)
    sleep_lvl = p1.select_slider("How much will you sleep tonight?", list(LEVELS), value="Usual")
    act_lvl = p2.select_slider("How active will you be today?", list(LEVELS), value="Usual")
    x0 = today.strain; u = strain_sign * (BS[0, 0] * LEVELS[sleep_lvl] + BS[0, 1] * LEVELS[act_lvl])
    path = [x0] + [phiS[0] ** (j - 1) * (phiS[0] * x0 + u) for j in range(1, 4)]
    scores = [score(v) for v in path]
    tomorrow = scores[1]; tname, tcol = label(tomorrow)
    st.markdown(f"Expected recovery score tomorrow morning: "
                f"<span style='font-size:28px;font-weight:700;color:{tcol}'>{tomorrow:.0f}</span> ({tname.lower()})",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.axhspan(65, 100, color="#2e7d32", alpha=.08); ax.axhspan(0, 40, color="#c62828", alpha=.08)
    ax.plot(["Today", "Tomorrow", "Day 2", "Day 3"], scores, marker="o", c="#1565c0", lw=2)
    ax.axhline(50, c="gray", lw=.8, ls="--"); ax.text(3.05, 50, "your usual", va="center", fontsize=8, color="gray")
    ax.set_ylim(0, 100); ax.set_ylabel("Recovery score"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)
    st.caption("Your body tends to bounce back within a day or two, so one short night has a short effect.")

    r_today = reps[(reps.id == pid) & (reps.date == today.date)]
    st.markdown("#### How you said you felt")
    st.write(", ".join(f"{t:%H:%M} {m.capitalize()}" for t, m in zip(r_today.t, r_today.MOOD)) if len(r_today)
             else "No check ins today.")
    st.caption("In this study, how people felt did not follow their recovery score closely. "
               "Your mood is shown next to your recovery, not predicted from it.")

# ---------------- trend ----------------
with tab2:
    st.markdown("#### Your recovery over time")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].axhspan(65, 100, color="#2e7d32", alpha=.08); ax[0].axhspan(0, 40, color="#c62828", alpha=.08)
    ax[0].plot(gv.date, gv.score, c="#1565c0", lw=1.8)
    ax[0].axhline(50, c="gray", lw=.8, ls="--"); ax[0].set_ylim(0, 100); ax[0].set_ylabel("Recovery score")
    short = gv[gv.minutesAsleep < -1]
    ax[0].scatter(short.date, short.score, c="#c62828", s=30, zorder=3, label="short night")
    ax[0].legend(loc="lower left", fontsize=8); ax[0].spines[["top", "right"]].set_visible(False)
    rr = reps[(reps.id == pid) & (reps.date <= today.date)]
    moods = prm["cats"]; ypos = {m: i for i, m in enumerate(moods)}
    ax[1].scatter(rr.t, rr.MOOD.map(ypos), s=12, c="#6a1b9a")
    ax[1].set_yticks(range(len(moods)), [m.capitalize() for m in moods], fontsize=7)
    ax[1].set_ylabel("How you felt"); ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)
    good = (gv.score >= 65).mean() * 100; bad = (gv.score < 40).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Well recovered mornings", f"{good:.0f}%")
    c2.metric("Under recovered mornings", f"{bad:.0f}%")
    if len(short):
        after = gv.score[gv.minutesAsleep < -1].mean()
        c3.metric("Average score after a short night", f"{after:.0f}")
    st.caption("Red dots mark nights more than one standard deviation shorter than your usual sleep.")

# ---------------- how it works ----------------
with tab3:
    st.markdown("""
**What the score is.** Each night your watch records your heart rate every few seconds, your heart rate variability
every five minutes, your sleep stages, and your skin temperature. The model compares last night with your own usual
nights and estimates a hidden *recovery* level, mostly from how low your heart rate went and how high your HRV was.
A score of 50 means a usual night for you.

**What changes it.** Sleeping longer raises tomorrow's score; a more active day lowers it a little. The effect of one
night fades within about a day.

**What it does not do.** It does not diagnose anything, and in this study it did not predict mood.
""")
    with st.expander("Technical details (for reviewers)"):
        st.markdown(f"Linear Gaussian state space model fitted by maximum likelihood with a Kalman filter; "
                    f"{S} hidden states chosen by BIC (autonomic strain, within night trajectory, sleep fragmentation). "
                    f"Recovery score = 100 × (1 − Φ(strain)). Hidden Markov model with {len(prm['hmm_theta'])} regimes.")
        st.markdown("**Held out mood classification** (chance 0.143)")
        st.dataframe(pd.concat([pd.DataFrame(prm["res_lr"]), pd.DataFrame(prm["res_x"]).iloc[1:]], ignore_index=True),
                     hide_index=True)
        st.markdown("**Hidden state loadings**")
        st.dataframe(pd.DataFrame(CS, index=ALL, columns=[f"state {s + 1}" for s in range(S)]).round(2))
        st.markdown("**Cleaning funnel**")
        st.dataframe(pd.DataFrame(prm["funnel"]), hide_index=True)
