# Demand Forecast & Inventory Optimizer

A calibrated, decision-aware demand-forecasting system on the Walmart M5 dataset.

**The pitch.** Most forecasting projects predict a single number and report MAPE.
This one forecasts a *distribution*, verifies that its prediction intervals are
honestly calibrated, and converts the forecast into a **cost-optimal order
quantity** — because in retail, running out of stock and overstocking do not
cost the same. The model is benchmarked the honest way for time series: a
leak-free, time-aware backtest, never a random split.

**The stakeholder question.** A retailer must decide how many units of each
product to order. Understocking loses sales; overstocking ties up cash and risks
spoilage. A point forecast cannot answer this — the planner needs the
distribution and a decision rule. What is the best forecast at the quantile that
matters, and can the planner trust the intervals?

---

## Dataset

Walmart M5 (Kaggle): daily unit sales for ~30,000 product-store series across 10
stores and 3 US states, over ~5.3 years.

**Scope for this project.** Narrowed to the **FOODS** category in store **CA_1**
— 1,437 product-store series over ~1,941 days. FOODS was chosen for the richest,
most regular demand; a single store keeps the problem tractable while preserving
real-world messiness.

### Data notes (key findings from EDA)

- **Intermittency.** 56.9% of all product-days have zero sales. The distribution
  is heavily right-skewed (median 0, 75th percentile 2, max 648, std ≈ 3× the
  mean). This is lumpy, intermittent demand.
- **Leading zeros.** Long zero runs at the start of a series — and some extended
  mid-series gaps — indicate the product was not stocked, not that demand was
  zero. These are "not on shelf," not real zero-demand observations.
- **Weekday effect.** Weekend sales run ~50% higher than midweek — a strong,
  exploitable weekly cycle.
- **SNAP effect.** Sales on SNAP-eligible days (US food-assistance disbursement
  days) are ~12% higher — a real, smaller demand driver.
- **Price.** Sales correlate −0.15 with price (correct direction, modest).
- **Events.** A single is-event flag shows little/slightly-negative effect,
  because event *types* pull in opposite directions (e.g. Super Bowl lifts food
  sales while Christmas suppresses them). Event *type* matters more than mere
  presence.
- **Hierarchy.** Within FOODS, department FOODS_3 sells roughly twice as much per
  day as FOODS_1 and FOODS_2.

---

## Methodology

### Why a time-aware backtest, not random K-fold

Time-series data is not row-independent: each day's sales relate to the days
around them, and at prediction time only *past* data is available. A random
K-fold split scatters days across train and test, so the model trains on days
that come *after* the days it is tested on — impossible in reality, and it leaks
future information (compounded by lag/rolling features). The result is a test
score that looks excellent in the notebook and collapses in production.

This project uses an **expanding-window backtest**: train on all days up to a
cutoff, predict the next 28 days, move the cutoff forward, repeat (3 folds).
Training data always precedes test data, mirroring real forecasting. Scores are
averaged across folds.

### Why MASE, not MAPE

MAPE divides error by the actual value. With 56.9% zero days, MAPE divides by
zero on the majority of observations — undefined or exploding, and useless here.
MASE scales error by the seasonal-naive baseline's error instead, so zeros cause
no division problem, and it has a directly interpretable meaning: **below 1 beats
the naive baseline.** That frames the project around the right question — does
the model add value over a trivial guess? (As a sanity check, the seasonal-naive
baseline scores MASE ≈ 0.9 against its own scaling, close to the theoretical 1.0,
confirming the metric is implemented correctly.)

### Leak-free features

All features for a given day use past information only: lags (`lag_7/14/28`),
rolling means/stds computed on sales **shifted back one day** (so the window ends
yesterday, never including the day being predicted), all computed **within each
series**. The first 28 days of each series (insufficient history) are dropped.

---

## Results so far

Evaluated on the same 3 expanding-window folds with the same MASE scaling.

| Model | Mean MASE | Notes |
|---|---:|---|
| Seasonal-naive (all series) | 0.908 | Headline baseline to beat |
| SARIMA (3 high-volume series) | 0.546 | vs 0.649 seasonal-naive on the *same* series; modest win but does not scale |
| **LightGBM (global)** | **0.715** | **~21% lower scaled error than baseline** |
| XGBoost (global) | 0.716 | Confirms the result is not library-specific |

**Takeaway.** Per-series SARIMA modestly beats seasonal-naive on hand-picked
high-volume products, but it needs one model per series, only works on non-sparse
data, and cannot use covariates like price or SNAP. A single **global LightGBM**
model scales to all 1,437 intermittent series, uses the engineered features
directly, and cuts scaled error ~21% below the baseline.

---

## Roadmap (in progress)

- [ ] Tune and finalise LightGBM; save with joblib
- [ ] **Probabilistic forecasts** — quantile regression for prediction intervals
- [ ] **Interval calibration** — verify a nominal 90% interval covers ~90% across the backtest
- [ ] **Cost-aware ordering** — newsvendor decision layer for the order quantity
- [ ] **Explainability** — SHAP global + per-prediction
- [ ] **Tests** — pytest on feature leakage + a predict() smoke test

---

## Data

The M5 CSVs are not committed (too large for GitHub). Download from the
[M5 Forecasting – Accuracy competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data)
and place `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv`
in a `data/` folder at the project root.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt          # plus: brew install libomp (macOS, for LightGBM)

# 2. Place the M5 CSVs in data/ (see above)

# 3. Run the notebooks in order
#    notebooks/01_data_prep_eda.ipynb   — load, reshape, EDA
#    notebooks/02_modeling.ipynb        — backtest, baselines, models
```

## Project structure

```
src/        data loading, features, backtest, metrics
notebooks/  EDA and modeling narratives
models/     saved models (gitignored)
tests/      pytest suite
reports/    figures
```

## Tech

scikit-learn · lightgbm · xgboost · statsmodels · shap · pytest · joblib · pandas
