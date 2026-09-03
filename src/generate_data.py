"""
Generates a synthetic but realistic e-commerce order dataset for return-risk scoring.

Design goals (so the eval is credible, not a toy):
  - Temporal structure: orders span a date range, customer history evolves over time
  - Class imbalance: ~12-15% return rate, roughly matching real e-commerce
  - Signal is present but noisy/overlapping (some legit orders look risky, some risky
    orders look clean) — avoids an artificially separable dataset
  - Leakage-free: no feature computed using data from AFTER the order date
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RNG = np.random.default_rng(42)
N_CUSTOMERS = 8000
N_ORDERS = 60000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)

CATEGORIES = ["apparel", "footwear", "electronics", "home", "beauty", "accessories", "books", "toys"]
CATEGORY_BASE_RETURN_RATE = {
    "apparel": 0.22, "footwear": 0.20, "electronics": 0.09, "home": 0.08,
    "beauty": 0.06, "accessories": 0.10, "books": 0.03, "toys": 0.07,
}
PAYMENT_METHODS = ["cod", "upi", "card", "netbanking", "wallet"]
PAYMENT_RISK_MULT = {"cod": 1.35, "upi": 0.95, "card": 0.85, "netbanking": 0.9, "wallet": 1.05}
DELIVERY_METHODS = ["standard", "express"]

def random_dates(n, start, end):
    delta_days = (end - start).days
    offsets = RNG.integers(0, delta_days, size=n)
    return [start + timedelta(days=int(o)) for o in offsets]

def main():
    # ---- customers: each has a latent "return propensity" (unobserved ground truth driver) ----
    customer_ids = np.arange(1, N_CUSTOMERS + 1)
    latent_propensity = RNG.beta(2, 8, size=N_CUSTOMERS)  # most customers low propensity, long tail
    signup_dates = random_dates(N_CUSTOMERS, START_DATE - timedelta(days=400), START_DATE)
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "latent_propensity": latent_propensity,
        "signup_date": signup_dates,
    })

    # ---- orders ----
    order_customer = RNG.choice(customer_ids, size=N_ORDERS)
    order_dates = random_dates(N_ORDERS, START_DATE, END_DATE)
    categories = RNG.choice(CATEGORIES, size=N_ORDERS, p=_normalize_probs())
    payment = RNG.choice(PAYMENT_METHODS, size=N_ORDERS, p=[0.30, 0.28, 0.25, 0.10, 0.07])
    delivery = RNG.choice(DELIVERY_METHODS, size=N_ORDERS, p=[0.8, 0.2])

    # order value: lognormal, category-dependent scale
    cat_scale = {"apparel": 1200, "footwear": 1800, "electronics": 6000, "home": 2200,
                 "beauty": 800, "accessories": 900, "books": 400, "toys": 1000}
    order_value = np.array([
        RNG.lognormal(mean=np.log(cat_scale[c]), sigma=0.6) for c in categories
    ]).round(2)

    # size/fit mismatch flag — more likely for apparel/footwear, weak independent signal
    size_mismatch_risk = RNG.random(N_ORDERS) < np.where(
        np.isin(categories, ["apparel", "footwear"]), 0.18, 0.03
    )

    # discount depth — heavy discounts weakly correlate with return risk (bargain-hunting behavior)
    discount_pct = np.clip(RNG.beta(2, 6, size=N_ORDERS), 0, 0.8)

    df = pd.DataFrame({
        "order_id": np.arange(1, N_ORDERS + 1),
        "customer_id": order_customer,
        "order_date": order_dates,
        "category": categories,
        "payment_method": payment,
        "delivery_method": delivery,
        "order_value": order_value,
        "discount_pct": discount_pct.round(3),
        "size_mismatch_risk": size_mismatch_risk.astype(int),
    })
    df = df.merge(customers, on="customer_id", how="left")
    df = df.sort_values("order_date").reset_index(drop=True)

    # ---- generate the label using a noisy latent function (logistic combination) ----
    base_rate = df["category"].map(CATEGORY_BASE_RETURN_RATE).values
    payment_mult = df["payment_method"].map(PAYMENT_RISK_MULT).values
    value_z = (np.log(df["order_value"]) - np.log(df["order_value"]).mean()) / np.log(df["order_value"]).std()

    logit = (
        np.log(base_rate / (1 - base_rate))
        + 1.8 * (df["latent_propensity"] - df["latent_propensity"].mean())
        + 0.35 * np.log(payment_mult)
        + 0.25 * df["size_mismatch_risk"]
        + 0.15 * value_z
        + 0.9 * (df["discount_pct"] - df["discount_pct"].mean())
        + RNG.normal(0, 0.55, size=N_ORDERS)  # irreducible noise — keeps it non-separable
    )
    prob_return = 1 / (1 + np.exp(-logit))
    df["returned"] = (RNG.random(N_ORDERS) < prob_return).astype(int)

    df = df.drop(columns=["latent_propensity"])  # not observable in real life — drop before saving

    print(f"Total orders: {len(df)}")
    print(f"Return rate: {df['returned'].mean():.3f}")
    print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")

    df.to_csv(f"{BASE_DIR}/data/orders_raw.csv", index=False)
    customers.drop(columns=["latent_propensity"]).to_csv(
        f"{BASE_DIR}/data/customers.csv", index=False
    )
    print("Saved data/orders_raw.csv and data/customers.csv")

def _normalize_probs():
    w = np.array([0.28, 0.14, 0.16, 0.14, 0.10, 0.10, 0.05, 0.03])
    return w / w.sum()

if __name__ == "__main__":
    main()
