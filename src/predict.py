"""
Score new orders with the trained stacked ensemble (CatBoost + RandomForest +
LogisticRegression base learners -> LogisticRegression meta-learner).

Usage:
    python predict.py --input new_orders.csv --output scored_orders.csv --threshold 0.32

Input CSV must contain the same raw columns as data/orders_raw.csv (customer_id,
order_date, signup_date, category, payment_method, delivery_method, order_value,
discount_pct, size_mismatch_risk) so that features.build_features() can compute
point-in-time customer/category history features.

Production note: cust_prior_* and cat_prior_* features here are recomputed from
whatever history is in the input file. In a real deployment these should be served
from a feature store updated as orders happen, not recomputed from a static CSV.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from features import build_features

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAT_FEATURES = ["category", "payment_method", "delivery_method"]
PASSTHROUGH_FEATURES = ["cust_prior_return_rate_smoothed", "cat_prior_return_rate"]
MODEL_DIR = f"{BASE_DIR}/models"


def load_ensemble():
    cb = CatBoostClassifier()
    cb.load_model(f"{MODEL_DIR}/catboost_ensemble_member.cbm")
    rf = joblib.load(f"{MODEL_DIR}/random_forest_ensemble_member.joblib")
    lr = joblib.load(f"{MODEL_DIR}/logreg_ensemble_member.joblib")
    meta = joblib.load(f"{MODEL_DIR}/meta_logreg.joblib")
    enc = joblib.load(f"{MODEL_DIR}/rf_onehot_encoder.joblib")
    scl = joblib.load(f"{MODEL_DIR}/rf_scaler.joblib")
    return cb, rf, lr, meta, enc, scl


def score(df, cb, rf, lr, meta, enc, scl, feature_cols):
    num_features = [c for c in feature_cols if c not in CAT_FEATURES]

    cb_proba = cb.predict_proba(Pool(df[feature_cols], cat_features=CAT_FEATURES))[:, 1]

    cat = df[CAT_FEATURES].astype(str)
    num = df[num_features].astype(float)
    X_rf = np.hstack([enc.transform(cat), scl.transform(num)])
    rf_proba = rf.predict_proba(X_rf)[:, 1]
    lr_proba = lr.predict_proba(X_rf)[:, 1]

    meta_X = np.column_stack([cb_proba, rf_proba, lr_proba, df[PASSTHROUGH_FEATURES].values])
    ensemble_proba = meta.predict_proba(meta_X)[:, 1]
    return ensemble_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.32,
                         help="Default is the cost-optimal threshold found in evaluate.py "
                              "for the example cost model — RETUNE for your own FN/FP costs.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["order_date", "signup_date"])
    df, feature_cols = build_features(df)

    cb, rf, lr, meta, enc, scl = load_ensemble()
    df["risk_score"] = score(df, cb, rf, lr, meta, enc, scl, feature_cols)
    df["flagged"] = (df["risk_score"] >= args.threshold).astype(int)

    df[["order_id", "customer_id", "order_date", "risk_score", "flagged"]].to_csv(
        args.output, index=False
    )
    print(f"Scored {len(df)} orders -> {args.output}")
    print(f"Flagged {df['flagged'].sum()} ({df['flagged'].mean()*100:.1f}%) at threshold {args.threshold}")


if __name__ == "__main__":
    main()
