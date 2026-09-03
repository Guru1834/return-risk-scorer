"""
SHAP explainability for the CatBoost ensemble member.

Kept strictly to a merchant-facing "why was this order flagged" view (order-level
feature attributions). Deliberately NOT exposing global feature-importance rankings
in a way that would let a bad actor learn exactly which signals to spoof — see
Track 02's defense-only requirement.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAT_FEATURES = ["category", "payment_method", "delivery_method"]
NUM_FEATURES = [
    "order_value", "discount_pct", "size_mismatch_risk",
    "cust_prior_orders", "cust_prior_return_rate_smoothed",
    "cust_prior_avg_order_value", "days_since_prev_order",
    "account_age_days", "cat_prior_return_rate", "order_value_vs_cust_avg",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

def main():
    df = pd.read_csv(
        f"{BASE_DIR}/data/orders_featured.csv",
        parse_dates=["order_date", "signup_date"],
    )
    df = df.sort_values("order_date").reset_index(drop=True)
    cutoff_idx = int(len(df) * 0.8)
    test = df.iloc[cutoff_idx:].copy()

    model = CatBoostClassifier()
    model.load_model(f"{BASE_DIR}/models/catboost_ensemble_member.cbm")

    sample = test.sample(n=min(1500, len(test)), random_state=42)
    pool = Pool(sample[ALL_FEATURES], cat_features=CAT_FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool)

    # global summary — aggregate only, no per-order raw values exposed
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, sample[ALL_FEATURES], show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/outputs/shap_summary.png", dpi=150)
    plt.close()
    print("Saved SHAP summary plot to outputs/shap_summary.png")

    # one worked example: explain a single flagged high-risk order (merchant-facing use case)
    proba = model.predict_proba(pool)[:, 1]
    idx = np.argmax(proba)
    print(f"\nExample high-risk order (predicted p={proba[idx]:.3f}):")
    example_row = sample.iloc[idx][ALL_FEATURES]
    example_shap = shap_values[idx]
    contrib = pd.DataFrame({
        "feature": ALL_FEATURES,
        "value": example_row.values,
        "shap_contribution": example_shap,
    }).sort_values("shap_contribution", key=abs, ascending=False)
    print(contrib.to_string(index=False))
    contrib.to_csv(f"{BASE_DIR}/outputs/shap_example_order.csv", index=False)

if __name__ == "__main__":
    main()
