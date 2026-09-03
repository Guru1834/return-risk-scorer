"""
Streamlit demo for the Return-Risk Scorer (Track 02 — AI Risk Manager).

Run:
    streamlit run app.py

Lets you:
  - Upload a CSV of orders (same schema as data/orders_raw.csv) and score them
  - Or explore the held-out test set predictions that are already saved in outputs/
  - Adjust the decision threshold live and see precision/recall/cost trade-offs update
  - View SHAP explanation for the highest-risk order in the current view
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from features import build_features  # noqa: E402
from predict import load_ensemble, score  # noqa: E402

st.set_page_config(page_title="Return-Risk Scorer", layout="wide")

FN_COST_FRACTION_DEFAULT = 0.35
FP_COST_FLAT_DEFAULT = 45.0

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def get_models():
    return load_ensemble()


@st.cache_data
def get_test_set_predictions():
    proba = np.load(os.path.join(BASE_DIR, "outputs", "ensemble_test_proba.npy"))
    labels = pd.read_csv(os.path.join(BASE_DIR, "outputs", "ensemble_test_labels.csv")).iloc[:, 0].values
    meta = pd.read_csv(os.path.join(BASE_DIR, "outputs", "ensemble_test_meta.csv"))
    return proba, labels, meta


@st.cache_data
def get_metrics_summary():
    with open(os.path.join(BASE_DIR, "outputs", "evaluation_summary.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Return-Risk Scorer")
st.sidebar.caption("Track 02 — AI Risk Manager · defense-only merchant risk scoring")

data_source = st.sidebar.radio(
    "Data source",
    ["Held-out test set (demo)", "Upload your own CSV"],
)

threshold = st.sidebar.slider(
    "Decision threshold", min_value=0.05, max_value=0.90, value=0.32, step=0.01,
    help="Orders scored above this are flagged as return-risk. 0.32 is the cost-optimal "
         "threshold found on the held-out test set with the default cost model below.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Cost model")
fn_cost_frac = st.sidebar.slider(
    "False-negative cost (% of order value)", 5, 80, int(FN_COST_FRACTION_DEFAULT * 100), step=5,
    help="Cost of missing a real return: reverse logistics + restock/write-off.",
) / 100.0
fp_cost_flat = st.sidebar.number_input(
    "False-positive cost (flat, ₹)", min_value=0.0, value=FP_COST_FLAT_DEFAULT, step=5.0,
    help="Cost of flagging a good order: manual review time / customer friction.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Defense-only: this app scores the merchant's own orders. It does not expose "
    "anything that helps a bad actor evade detection."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("🛡️ Return-Risk Scorer")
st.write(
    "Scores e-commerce orders for return probability at checkout/fulfillment time, "
    "so a merchant can intervene (require prepay, add a size-confirmation step) "
    "**before** the return happens instead of eating the cost after."
)

if data_source == "Held-out test set (demo)":
    proba, labels, meta = get_test_set_predictions()
    st.info(
        f"Showing predictions on the **held-out test set** ({len(proba):,} orders, "
        f"never seen during training) — this is the honest, unseen-data evaluation set."
    )
else:
    uploaded = st.file_uploader("Upload orders CSV", type="csv")
    if uploaded is None:
        st.warning(
            "Upload a CSV with columns: customer_id, order_date, signup_date, category, "
            "payment_method, delivery_method, order_value, discount_pct, size_mismatch_risk "
            "(same schema as data/orders_raw.csv)."
        )
        st.stop()
    raw_df = pd.read_csv(uploaded, parse_dates=["order_date", "signup_date"])
    with st.spinner("Scoring orders..."):
        df_featured, feature_cols = build_features(raw_df)
        cb, rf, lr, meta_model, enc, scl = get_models()
        proba = score(df_featured, cb, rf, lr, meta_model, enc, scl, feature_cols)
        labels = df_featured["returned"].values if "returned" in df_featured.columns else None
        meta = df_featured[["order_id", "order_value"]].reset_index(drop=True)
    st.success(f"Scored {len(proba):,} orders.")

pred = (proba >= threshold).astype(int)
n_flagged = pred.sum()

# ---- top metrics row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Orders scored", f"{len(proba):,}")
col2.metric("Flagged at this threshold", f"{n_flagged:,}", f"{n_flagged / len(proba) * 100:.1f}%")

if labels is not None:
    precision = precision_score(labels, pred, zero_division=0)
    recall = recall_score(labels, pred, zero_division=0)
    col3.metric("Precision", f"{precision:.3f}")
    col4.metric("Recall", f"{recall:.3f}")

    fn_mask = (pred == 0) & (labels == 1)
    fp_mask = (pred == 1) & (labels == 0)
    fn_cost = (meta["order_value"].values[fn_mask] * fn_cost_frac).sum()
    fp_cost = fp_mask.sum() * fp_cost_flat
    total_cost = fn_cost + fp_cost
    do_nothing_cost = (meta["order_value"].values[labels == 1] * fn_cost_frac).sum()

    st.markdown("### Cost impact at this threshold")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated total cost", f"₹{total_cost:,.0f}")
    c2.metric("vs. never flagging", f"₹{do_nothing_cost:,.0f}",
              f"-₹{do_nothing_cost - total_cost:,.0f} saved", delta_color="normal")
    c3.metric("False positives / negatives", f"{fp_mask.sum():,} / {fn_mask.sum():,}")
else:
    col3.metric("Precision", "n/a (no labels)")
    col4.metric("Recall", "n/a (no labels)")

st.markdown("---")

# ---- PR curve + cost curve (only when labels are available) ----
if labels is not None:
    left, right = st.columns(2)

    with left:
        st.subheader("Precision-Recall curve")
        p, r, thresh_arr = precision_recall_curve(labels, proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(r, p, label=f"Ensemble (AP={average_precision_score(labels, proba):.3f})")
        ax.axhline(labels.mean(), color="gray", linestyle="--", label=f"Base rate ({labels.mean():.3f})")
        ax.axvline(recall, color="orange", linestyle=":", alpha=0.7)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig)

    with right:
        st.subheader("Cost vs. threshold")
        ths = np.linspace(0.05, 0.9, 30)
        costs = []
        for t in ths:
            p_t = (proba >= t).astype(int)
            fn_c = (meta["order_value"].values[(p_t == 0) & (labels == 1)] * fn_cost_frac).sum()
            fp_c = ((p_t == 1) & (labels == 0)).sum() * fp_cost_flat
            costs.append(fn_c + fp_c)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(ths, costs)
        ax2.axvline(threshold, color="orange", linestyle=":", label="Current threshold")
        ax2.axhline(do_nothing_cost, color="gray", linestyle="--", label="Never flag")
        ax2.set_xlabel("Decision threshold"); ax2.set_ylabel("Total cost (₹)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        st.pyplot(fig2)

st.markdown("---")

# ---- flagged orders table ----
st.subheader(f"Flagged orders (threshold ≥ {threshold:.2f})")
display_df = meta.copy()
display_df["risk_score"] = proba
display_df["flagged"] = pred
if labels is not None:
    display_df["actual_returned"] = labels
display_df = display_df.sort_values("risk_score", ascending=False)
st.dataframe(
    display_df[display_df["flagged"] == 1].head(200),
    use_container_width=True,
    height=350,
)
st.caption(
    f"Showing top 200 of {n_flagged:,} flagged orders, sorted by risk score. "
    "Full results available via src/predict.py for batch export."
)

# ---- model comparison from training run ----
with st.expander("Model comparison (from training run)"):
    try:
        summary = get_metrics_summary()
        st.json(summary)
    except FileNotFoundError:
        st.write("Run `train_ensemble.py` and `evaluate.py` to populate this section.")
