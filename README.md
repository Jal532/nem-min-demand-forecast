# Forecasting Minimum Operational Demand in the NEM

## Problem
Rooftop solar growth across Australia's National Electricity Market (NEM) is pushing
daytime operational demand toward zero — and sometimes below it — in several regions.
This threatens grid stability, since fewer synchronous generators (coal, gas, hydro)
stay online to provide inertia and voltage support when demand is very low. AEMO has
flagged this as a live operational risk requiring emergency intervention capability.

This project explores that problem using real NEM market data, and builds an early-warning
classifier that predicts, up to 3 hours ahead, whether South Australian demand will fall
below 200 MW.

## Data
- AEMO Aggregated Price and Demand data (public, no auth required):
  https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
- **Modeling:** SA1 (South Australia), summer months (Dec/Jan/Feb) from 2023–2026, at
  5-minute resolution.
- **EDA:** NSW1 and SA1, January 2026, plus SA1 June 2026 for the seasonal comparison.
- Raw CSVs are not committed (re-downloadable from the link above).

## Status: modeling complete (baseline, XGBoost, LSTM); 

### EDA findings
- **NSW1 price is threshold-driven, not linear with demand.** Price stays near zero
  across almost the full demand range, then can spike above $10,000/MWh once demand
  crosses roughly 12,000 MW — but not every high-demand period triggers a spike,
  suggesting other constraints (e.g. generator outages) also matter.
- **SA1 operational demand goes negative, and it's seasonal, not incidental.**
  94 of 8,928 intervals in January 2026 had negative demand, entirely concentrated
  9am–3pm (peaking at 1pm) — vs. zero negative intervals in June 2026. This confirms
  rooftop solar output, not just low baseline demand, as the driving mechanism.

### Modeling: predicting minimum-demand risk 3 hours ahead
Target: will SA1 demand drop below 200 MW at any point in the next 3 hours?
Trained on 2023–2025 summer data (Dec/Jan/Feb), evaluated on held-out 2026 data.

| Model | Row-level PR-AUC | Episodes (n) | Episode recall | Median lead time | False-alarm periods |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.934 | 14 | – | – | – |
| XGBoost @ 0.5 threshold | 0.977 | 14 | 71% (10/14) | 27.5 min | 5 |
| **XGBoost @ validated threshold (0.1)** | **0.977** | **14** | **86% (12/14)** | **42.5 min** | **5** |
| XGBoost + weather features | 0.974 | 14 | 86% (12/14) | 37.5 min | 6 |
| LSTM @ validated threshold (0.4) | 0.973 | 14 | 86% (12/14) | 27.5 min | 5 |

**Key finding:** row-level PR-AUC (0.977) looked near-perfect, but evaluating at the
*episode* level — merging consecutive alarm rows into discrete breach events — showed
the default 0.5 threshold only caught 10 of 14 real episodes, with a median warning of
just 27.5 minutes. Selecting a decision threshold on a held-out 2025 validation set
(avoiding leakage from the 2026 test set) raised recall to 12/14 with 42.5 minutes
median lead time, at no cost in false alarms.

**Adding weather features (Adelaide irradiance/cloud cover) did not improve results**,
and slightly reduced lead time. Likely cause: hourly weather forward-filled onto
5-minute demand data loses the fast-moving cloud cover that matters for a 30–60 minute
warning window, and a single point location may not represent cloud cover across SA1's
whole rooftop solar fleet. Noted as a direction for future work rather than pursued
further, given the small (n=14) episode sample.

**An LSTM sequence model was also tested** as a comparison to the hand-engineered
feature approach, trained with the same leakage-safe threshold validation procedure.
It matched XGBoost's episode recall (12/14) but with a shorter median lead time
(27.5 vs. 42.5 minutes) and marginally lower PR-AUC (0.973 vs. 0.977). This is a
plausible result given data volume: the training set contains only a few dozen
independent breach episodes across 2023–2025, likely too few for an LSTM to learn
temporal structure that was already hand-encoded as lag/cyclic features for XGBoost.
XGBoost @ validated threshold is used as the project's primary model.

**Methodology notes:**
- Evaluated at both row level (PR-AUC) and episode level (recall, lead time, false
  alarms), since row-level metrics treat highly correlated 5-minute rows as
  independent and can overstate real-world performance.
- Decision thresholds were always selected on a validation year distinct from the
  final test year, to avoid leakage.
- A naive persistence baseline ("alarm if current demand < 400 MW") scored PR-AUC
  0.404, confirming the models learn genuine forward-looking signal rather than
  echoing current demand.

### Limitations
- The test set contains only 14 breach episodes from a single summer (2026), so episode-level
  percentages have wide uncertainty (e.g. 12/14 ≈ 86%, 95% interval roughly 60–96%).
- Thresholds and results are specific to SA1 and a 200 MW definition of "breach"; sensitivity
  to other thresholds was not tested.
- The weather-feature explanation above is a hypothesis; it was not tested directly.

## Project structure
- `notebooks/0.1_explore_data.ipynb`: EDA (NSW price thresholds, SA negative demand)
- `notebooks/0.2_baseline_model.ipynb`: logistic regression baseline
- `notebooks/0.3_gradient_boosting.ipynb`: XGBoost, persistence baseline, onset-only evaluation
- `notebooks/0.4_episode_eval_and_weather.ipynb`: episode-level evaluation, threshold validation, weather features
- `notebooks/0.5_lstm.ipynb`: LSTM comparison
- `requirements.txt`: dependencies

## Reproducing
Download the monthly SA1 CSVs from the AEMO link above into `data/`, `pip install -r requirements.txt`, then run the notebooks in order.

