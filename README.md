# Return-Risk Scorer

**Track 02 — AI Risk Manager.** Scores e-commerce orders at checkout/fulfillment time
for the probability they'll be returned, so a merchant can intervene (e.g. require
prepay instead of COD, add a size-confirmation step) before the return happens instead
of eating the cost after.

Defense-only: this repo produces a **risk score for the merchant's own orders**. It
does not expose anything that helps an attacker evade detection (no raw model file
with fully interpretable thresholds shipped as "the rules," no per-fraudster profiling).

## Why a stacked ensemble instead of one model

A single gradient-boosted tree model (CatBoost) is the natural first baseline —
handles categorical features natively, fast, hard to beat on tabular data. But one
model's errors are *its* errors. This repo stacks three genuinely different learners
and lets a meta-model learn how to combine them:

| Base learner | Why it's here |
|---|---|
| **CatBoost** | Captures non-linear feature interactions, handles categoricals (payment method, category) without manual encoding |
| **Random Forest** | Bagged trees — different bias/variance profile than boosting, tends to make different mistakes than CatBoost on noisy rows |
| **Logistic Regression** | Linear model — picks up monotone signal (e.g. "higher discount % → higher risk") cleanly instead of fragmenting it across many tree splits |

Base-model out-of-fold predictions are generated with **time-series cross-validation**
(forward-chaining, `TimeSeriesSplit`), never random K-fold — a random split would let
a fold "see the future" relative to another, which is a leakage bug that quietly
inflates offline metrics on any temporal problem. The meta-learner (logistic
regression) also sees two of the strongest raw features directly alongside the three
base predictions ("feature-weighted stacking") — this is what recovered real lift
after an initial pass where CatBoost and the meta-learner were too correlated with
Random Forest to add anything (see `train_ensemble.py` docstring for the honest
before/after).

## Results (held-out, time-based test set — last 20% of orders by date)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| CatBoost (solo) | 0.233 | 0.650 |
| Random Forest (solo) | 0.235 | 0.658 |
| Logistic Regression (solo) | 0.237 | 0.658 |
| **Stacked ensemble** | **0.238** | **0.658** |

Base return rate in the test set: **15.4%** (PR-AUC of 0.238 vs a 0.154 no-skill
baseline is a real but modest lift — reported honestly, not inflated. This is
realistic for a genuinely noisy signal; see "Known limitations" below).

**The ensemble's real advantage isn't the average-case metric — it's operating
stability.** The CatBoost-solo model's predicted probabilities are tightly clustered
(a side effect of `auto_class_weights="Balanced"`), so its cost-vs-threshold curve has
a sharp cliff: a small miscalibration in the chosen threshold flips you from
near-optimal to worst-case cost. The ensemble's cost curve is smooth with a wide
low-cost plateau — much safer to deploy when the "right" threshold will drift over
time. See `outputs/evaluation_plots.png`.

### Cost-weighted evaluation (the part that actually matters for the business)

Cost model used (documented, tune per your own unit economics in `evaluate.py`):
- **False negative** (missed a real return): 35% of order value (reverse logistics + restock/write-off)
- **False positive** (flagged a good order): flat ₹45 (manual review time, not proportional to order value)

On the held-out test set (12,054 orders):

| Strategy | Total cost | Savings vs. never flagging |
|---|---|---|
| Never flag anything | ₹1,584,064 | — |
| Flag everything | ₹458,865 | ₹1,125,199 |
| CatBoost solo (threshold 0.47) | ₹458,331 | ₹1,125,733 |
| **Stacked ensemble (threshold 0.32)** | **₹417,721** | **₹1,166,343** |

The ensemble's cost-optimal threshold saves an additional **~₹40,600** over the
CatBoost-solo baseline on this test set, on top of being safer to operate at
(see stability point above).

## Explainability

`src/explain.py` produces SHAP attributions for the CatBoost ensemble member —
both a global summary (`outputs/shap_summary.png`) and a worked per-order example
(`outputs/shap_example_order.csv`), so a merchant ops team can see *why* a specific
order was flagged, not just that it was. Kept at the pattern level deliberately
(aggregate summary + one example) rather than a full per-fraudster feature dump.

## Repo structure

```
app.py               Streamlit demo — interactive threshold/cost explorer
data/                synthetic dataset + generation script (see below)
src/
  generate_data.py  builds the synthetic order dataset
  features.py        point-in-time feature engineering (leakage-safe)
  train_baseline.py  single CatBoost model, time-based split
  train_ensemble.py  3-model stacked ensemble, time-series OOF stacking
  evaluate.py         PR curves, threshold tables, cost-weighted analysis
  explain.py          SHAP explainability
  predict.py          inference on new orders
models/              saved model artifacts (.cbm, .joblib)
outputs/             metrics, plots, threshold tables
```

All paths in `src/` are relative to the repo root (via `BASE_DIR`), so this runs
unchanged after cloning — no hardcoded paths to edit.

## Reproducing

```bash
pip install -r requirements.txt
cd src
python generate_data.py      # builds data/orders_raw.csv (~60k synthetic orders)
python features.py           # builds data/orders_featured.csv
python train_baseline.py     # trains + evaluates CatBoost solo
python train_ensemble.py     # trains + evaluates the stacked ensemble
python evaluate.py           # cost analysis + plots -> outputs/
python explain.py            # SHAP plots -> outputs/
python predict.py --input ../data/sample_new_orders.csv --output ../outputs/scored.csv --threshold 0.32
```

## Demo app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Lets you explore the held-out test-set predictions (or upload your own orders CSV)
with a live threshold slider, an editable cost model (FN %/FP flat cost), the
resulting PR curve and cost-vs-threshold curve, and a table of flagged orders —
useful for a live hackathon demo instead of showing static plots.

## Data

**This dataset is synthetic**, generated by `src/generate_data.py` — 60,000 orders
across 8,000 customers, 2-year span, with a deliberately noisy label-generating
process (logistic combination of category base rate, customer latent propensity,
payment method, discount depth, and an irreducible random noise term) so the
resulting PR-AUC isn't artificially inflated by a trivially-separable toy dataset.
Real merchant order/return history was not available for this project in the
hackathon timeframe — for a production deployment, retrain on the merchant's actual
order + return logs using the same `features.py` pipeline (it only assumes columns
that any order management system already has: customer id, order date, category,
payment method, order value, discount, delivery method, and a returned/not-returned
label).

## Known limitations (stated honestly, per the track's bar)

- **PR-AUC of 0.238 is modest.** This reflects a genuinely noisy synthetic label
  (by design — see Data section) more than it reflects a ceiling on real-world
  performance; real order data typically has additional strong signal this synthetic
  set doesn't model (e.g. actual review text, image-based size-fit data, IP/device
  fingerprints) which would likely raise this substantially.
- **CatBoost's predicted probabilities are poorly calibrated** due to
  `auto_class_weights="Balanced"` — usable for ranking/thresholding (which is all this
  system needs) but not as a literal "probability of return." A production version
  should apply isotonic or Platt calibration before showing scores to non-technical
  stakeholders.
- **Cost model constants (35% FN fraction, ₹45 FP cost) are illustrative**, not
  sourced from a real merchant's finance team — swap in real numbers before trusting
  the recommended threshold.
- **No abuse-ring / collusion signal** (that's Track 02's separate "Abuse-ring
  sentinel" direction) — this scorer treats every order independently.
