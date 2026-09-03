"""
Point-in-time feature engineering.

The single most common mistake in return/fraud risk models is leakage: computing
a customer's "return rate" using their FULL history (including orders that happen
after the order being scored). That inflates offline metrics and collapses in
production. Every rolling/aggregate feature here is computed using expanding
windows that only look backward from each order's timestamp.
"""

import pandas as pd
import numpy as np

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    # --- expanding (point-in-time) customer history, shifted by 1 so current order excluded ---
    grp = df.groupby("customer_id")
    df["cust_prior_orders"] = grp.cumcount()
    df["cust_prior_returns"] = (
        grp["returned"].apply(lambda s: s.shift().expanding().sum()).reset_index(drop=True).fillna(0.0)
    )
    df["cust_prior_return_rate"] = (df["cust_prior_returns"] / df["cust_prior_orders"]).fillna(0.0)
    # Laplace-smoothed version to avoid wild variance for customers with 1-2 orders
    df["cust_prior_return_rate_smoothed"] = (
        (df["cust_prior_returns"] + 1) / (df["cust_prior_orders"] + 5)
    )

    df["cust_prior_avg_order_value"] = (
        grp["order_value"].apply(lambda s: s.shift().expanding().mean()).reset_index(drop=True)
    )
    df["cust_prior_avg_order_value"] = df["cust_prior_avg_order_value"].fillna(df["order_value"].median())

    # days since previous order (recency / burst behavior)
    df["days_since_prev_order"] = (
        grp["order_date"].diff().dt.days
    )
    df["days_since_prev_order"] = df["days_since_prev_order"].fillna(9999)  # first order sentinel

    # account age at order time
    df["account_age_days"] = (df["order_date"] - df["signup_date"]).dt.days

    # category-level prior return rate — computed with a *global* expanding window ordered by date,
    # so no future leakage across the whole dataset either
    df = df.sort_values("order_date").reset_index(drop=True)
    cat_grp = df.groupby("category")
    df["cat_prior_orders"] = cat_grp.cumcount()
    df["cat_prior_returns"] = (
        cat_grp["returned"].apply(lambda s: s.shift().expanding().sum()).reset_index(drop=True).fillna(0.0)
    )
    df["cat_prior_return_rate"] = (
        (df["cat_prior_returns"] + 2) / (df["cat_prior_orders"] + 10)  # smoothed, cold-start safe
    )

    # order value relative to customer's own history (spike detection)
    df["order_value_vs_cust_avg"] = df["order_value"] / df["cust_prior_avg_order_value"].replace(0, np.nan)
    df["order_value_vs_cust_avg"] = df["order_value_vs_cust_avg"].fillna(1.0).clip(0, 20)

    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    feature_cols = [
        "category", "payment_method", "delivery_method",           # categorical
        "order_value", "discount_pct", "size_mismatch_risk",       # order-level
        "cust_prior_orders", "cust_prior_return_rate_smoothed",     # customer history
        "cust_prior_avg_order_value", "days_since_prev_order",
        "account_age_days", "cat_prior_return_rate",
        "order_value_vs_cust_avg",
    ]
    return df, feature_cols


if __name__ == "__main__":
    df = pd.read_csv(
        f"{BASE_DIR}/data/orders_raw.csv",
        parse_dates=["order_date", "signup_date"],
    )
    df, feature_cols = build_features(df)
    df.to_csv(f"{BASE_DIR}/data/orders_featured.csv", index=False)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Saved {len(df)} rows to data/orders_featured.csv")
