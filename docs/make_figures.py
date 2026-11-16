"""Architecture diagram and UI wireframe for the technical blueprint."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', fc=color, ec='#333', lw=1))
    ax.text(x + w / 2, y + h - 0.06, title, ha='center', va='top', fontsize=10.5, weight='bold')
    ax.text(x + w / 2, y + h - 0.2, '\n'.join(lines), ha='center', va='top', fontsize=7.6, linespacing=1.35)

def arrow(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle='->', lw=1.6, color='#333'))

# architecture
fig, ax = plt.subplots(figsize=(11, 3.4)); ax.set_xlim(0, 4.4); ax.set_ylim(0, 1.15); ax.axis('off')
cols = ['#dbe9f6', '#e2f0d9', '#fde9d9', '#ede7f6']
specs = [
    ('Data', ['LifeSnaps v4 CSVs (Zenodo)', 'daily Fitbit + hourly mood reports', 'personality (IPIP)', '', 'prepare.py:', 'daily panel, per person z scores,', 'wake date alignment, cohort n=48,', 'chronological split']),
    ('Model', ['Stage 1: linear Gaussian SSM', '(Kalman filter, done)', 'Stage 2: per person coupling of', 'recovery state to mood', 'Stage 3 (optional): neural', 'discrepancy term (PyTorch/Pyro)', '', 'Baselines: carry forward, logistic,', 'VAR, OU mood model, LSTM']),
    ('Inference', ['causal filtering each morning', 'next report mood probabilities', 'with uncertainty', '', 'what if simulation of sleep', 'change through fitted dynamics', '', 'person level parameters:', 'recovery half life, coupling']),
    ('User interface', ['Streamlit app', '', 'select participant,', 'sleep what if slider', '', 'shows recovery trajectory,', 'mood forecast, perception gap']),
]
for i, (t, l) in enumerate(specs):
    box(ax, 0.05 + i * 1.1, 0.05, 0.92, 1.0, t, l, cols[i])
    if i < 3: arrow(ax, 0.99 + i * 1.1, 1.13 + i * 1.1, 0.55)
plt.tight_layout(); plt.savefig('architecture.png', dpi=200); plt.close()

# wireframe
fig, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis('off')
ax.add_patch(plt.Rectangle((0.1, 0.1), 10.8, 6.0, fc='white', ec='#333', lw=1.5))
ax.text(0.3, 5.75, 'Recovery and Mood Explorer', fontsize=13, weight='bold')
ax.add_patch(plt.Rectangle((0.3, 0.3), 2.4, 5.2, fc='#f2f2f2', ec='#999'))
side = ['Participant  [ dropdown v ]', '', 'Date range  [ ---|=====|--- ]', '', 'What if: last night sleep',
        '  [ -2h ----o---- +2h ]', '', 'Show HRV   [x]', 'Show mood reports [x]', '', '( Upload my own CSV )', '  planned extension']
for i, s in enumerate(side): ax.text(0.45, 5.25 - i * 0.38, s, fontsize=8.5, family='monospace')
def panel(x, y, w, h, t, body):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc='#fbfbfb', ec='#999'))
    ax.text(x + 0.12, y + h - 0.3, t, fontsize=9.5, weight='bold')
    for i, s in enumerate(body): ax.text(x + 0.12, y + h - 0.7 - i * 0.32, s, fontsize=8)
panel(2.9, 3.1, 7.8, 2.4, 'Recovery trajectory (latent state with 95% band)',
      ['line: recovery state   bars: sleep minutes   dots: resting HR', 'shaded: short nights and high exertion days', 'dashed line: what if trajectory from the slider'])
panel(2.9, 0.3, 3.8, 2.6, 'Tomorrow mood forecast', ['probability bar for each of 7 moods', 'by time of day: morning / afternoon /', 'evening', 'what if bars beside current bars'])
panel(6.9, 0.3, 3.8, 2.6, 'Perception gap', ['person coupling: physiology to mood', 'gauge: weak / moderate / strong', 'recovery half life: x.x days', 'text explanation of the forecast'])
plt.tight_layout(); plt.savefig('wireframe.png', dpi=200); plt.close()
print('figures written')
