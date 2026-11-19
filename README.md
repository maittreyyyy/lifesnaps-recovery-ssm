# Modeling Day to Day Physiological Recovery from Consumer Wearables

A state space model of daily physiological recovery, driven by sleep and activity and observed
through resting heart rate and heart rate variability, applied to self reported mood in the open
**LifeSnaps** dataset (71 participants, Fitbit Sense plus momentary mood reports).

The project tests whether the latent recovery state (1) behaves like physiological recovery and
(2) when coupled to mood per person, improves mood forecasts and relates to emotional stability.

Course: Applied Machine Learning II (EEE 6778), University of Florida, Fall 2026.

## Current status (Deliverable 2: raw intraday pipeline, models, interface)
- **Data:** the raw LifeSnaps Fitbit export (9.1 GB, 73M records) converted to Parquet: heart rate every 5 s,
  HRV every 5 min during sleep, sleep stage timelines, wrist temperature every minute.
- **Cleaning:** 4,141 sleep records to 2,338 valid nights from 57 people (main sleep, 180 to 900 min asleep,
  one night per date, inside each person's mood reporting window).
- **Night features:** HR nadir and its timing, HR drop, mean and late minus early RMSSD, deep fraction and timing,
  wake bouts per hour, wrist temperature, minutes asleep (per person z scores).
- **Models:** Kalman state space models with 1 to 3 hidden states (BIC selects 3: autonomic strain, within night
  trajectory, sleep fragmentation; only strain persists, half life about 0.6 days), a hidden Markov model with
  mood as 7 categories, a mixed model with per person coupling, and logistic baselines.
- **Results (held out):** best mood balanced accuracy 0.357 (carry forward; chance 0.143); physiology changes it by
  -0.016 to +0.001. No hidden state loads on mood and no person shows coupling above 0.2. Short nights raise HR
  nadir (+0.49 SD) and lower RMSSD (-0.33 SD).
- **Interface:** Streamlit explorer that replays nights one morning at a time, with a what if forecast.

Reports: `docs/d2/preliminary_report.pdf` (Deliverable 2), `docs/technical_blueprint.pdf` (Deliverable 1.2).

## Repository structure
```
data/raw/          LifeSnaps CSVs (CC BY 4.0, see data/README.md)
data/processed/d2/ model outputs used by the app (exported by the analysis notebook)
notebooks/         setup.ipynb (Deliverable 1); lifesnaps_analysis.ipynb (raw data to models, Deliverable 2)
src/               prepare.py, lssm.py (Stage 1 model), evaluation and checks
ui/                app.py: Streamlit recovery explorer
results/           figures
docs/              reports (LaTeX and PDF), architecture diagrams, interface screenshots (docs/screens)
scripts/           HiPerGator environment setup
run_all.sh         reproduces all Stage 1 results
```

## Installation
Python 3.11 or newer.
```bash
git clone https://github.com/maittreyyyy/lifesnaps-recovery-ssm.git
cd lifesnaps-recovery-ssm
python -m venv .venv && source .venv/bin/activate      # or use conda
pip install -r requirements.txt
```
On UF HiPerGator, `scripts/setup_env_hipergator.sh <group>` creates a conda environment under
`/blue` and registers a Jupyter kernel for Open OnDemand.

## Running
**1. Raw data analysis (UF HiPerGator or any machine with about 16 GB RAM).** Download `rais_anonymized.zip`
from Zenodo record 7229547 and unzip it under `~/Data Analysis Zenodo/Data/` (or set the `LS_BASE` environment
variable to another base folder). Open `notebooks/lifesnaps_analysis.ipynb` and run all cells. The first run converts
the raw BSON export to Parquet (about 3 minutes); the full notebook takes about 10 minutes. All analysis choices are in
the *Decisions* cell. The last cell writes `d2_export.zip`. The committed notebook already contains every output from the HiPerGator run on the real data, so it can be read without rerunning.

**2. Interface.** Unzip `d2_export.zip` into `data/processed/d2/` (the repository already contains a copy), then:
```bash
streamlit run ui/app.py
```
Pick a participant in the sidebar, move the replay slider to any morning, and use the what if sliders.

**3. Stage 1 (Deliverable 1) on the daily CSVs:** `jupyter lab notebooks/setup.ipynb` and `./run_all.sh`.

## Known issues
- Results come from a single chronological split; bootstrap confidence intervals and leave one subject out
  evaluation are planned for Deliverable 3.
- The raw export contains 134,133 duplicate heart rate rows and 98,586 readings with confidence 0; both are kept by
  default and can be removed in the Decisions cell. A few nights have more deep sleep than total sleep.
- HRV is available for 43 of 71 participants only; missing HRV is a property of the person, not the night.
- The app's forecasts use population parameters and show recovery states only; the evaluation does not support a
  mood forecast.
- A live wearable connection would use the Google Health API (the Fitbit Web API was turned off in October 2026).

## Dataset
LifeSnaps version 4, Zenodo DOI [10.5281/zenodo.7229547](https://doi.org/10.5281/zenodo.7229547),
licensed CC BY 4.0. The daily, hourly, and personality CSVs are included in `data/raw/`.
Citation: S. Yfantidou et al., "LifeSnaps, a 4-month multi-modal dataset capturing unobtrusive
snapshots of our lives in the wild," *Scientific Data*, 9, 663, 2022.

## Author
Maitrey Phatak, University of Florida, m.phatak@ufl.edu
