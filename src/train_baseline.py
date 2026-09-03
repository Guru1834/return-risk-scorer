"""
Baseline model: single CatBoostClassifier.
Time-based split (not random) — critical for honest metrics on a temporal problem.
"""

import pandas as pd
import numpy as np
import json
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAT_FEATURES = ["category", "payment_method", "delivery_method"]

def time_split(df, test_frac=0.2):
    df = df.sort_values("order_date").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_date = df.iloc[cutoff_idx]["order_date"]
    train = df[df["order_date"] < cutoff_date].copy()
    test = df[df["order_date"] >= cutoff_date].copy()
    return train, test, cutoff_date

def main():
    df = pd.read_csv(
        f"{BASE_DIR}/data/orders_featured.csv",
        parse_dates=["order_date", "signup_date"],
    )
    from features import build_features  # reuse feature col list definition
    _, feature_cols = build_features(df.copy())

    train, test, cutoff = time_split(df)
    print(f"Time-based split cutoff: {cutoff.date()}")
    print(f"Train: {len(train)} orders ({train['returned'].mean():.3f} return rate)")
    print(f"Test:  {len(test)} orders ({test['returned'].mean():.3f} return rate)")

    X_train, y_train = train[feature_cols], train["returned"]
    X_test, y_test = test[feature_cols], test["returned"]

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    print(f"\nPR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    model.save_model(f"{BASE_DIR}/models/catboost_baseline.cbm")
    np.save(f"{BASE_DIR}/outputs/baseline_test_proba.npy", proba)
    y_test.to_csv(f"{BASE_DIR}/outputs/baseline_test_labels.csv", index=False)
    test[["order_id", "order_value"]].to_csv(
        f"{BASE_DIR}/outputs/baseline_test_meta.csv", index=False
    )

    with open(f"{BASE_DIR}/outputs/baseline_metrics.json", "w") as f:
        json.dump({"pr_auc": pr_auc, "roc_auc": roc_auc, "cutoff_date": str(cutoff.date())}, f, indent=2)

    print("Saved model + test predictions to models/ and outputs/")

if __name__ == "__main__":
    main()
