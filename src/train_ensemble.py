"""
Stacked ensemble: CatBoost + RandomForest base learners -> Logistic Regression meta-learner.

Why this should beat the single CatBoost baseline:
  - CatBoost captures non-linear interactions + handles categoricals natively
  - RandomForest is a different algorithm family (bagged, high-variance trees) that
    tends to make *different* errors than boosted trees, especially on the noisy
    synthetic irreducible-error term we injected — averaging/stacking diverse errors
    is where ensemble lift actually comes from (not from two similar GBMs)
  - Logistic regression meta-learner learns how to weight the two base models'
    out-of-fold predictions rather than just naive-averaging them

Leakage control: base-model OOF predictions for the training set are generated with
TIME-SERIES cross-validation (expanding window, forward-chaining) — never a random
K-fold, which would let each fold "see the future" of another. The meta-learner never
sees any prediction made by a model that was trained on that exact row.
"""

import pandas as pd
import numpy as np
import json
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw features passed through directly to the meta-learner alongside base-model
# predictions ("feature-weighted stacking") — gives the meta-learner context the
# base models' point predictions alone don't carry, and reliably recovers lift
# when base learners are highly correlated.
PASSTHROUGH_FEATURES = ["cust_prior_return_rate_smoothed", "cat_prior_return_rate"]

CAT_FEATURES = ["category", "payment_method", "delivery_method"]
NUM_FEATURES = [
    "order_value", "discount_pct", "size_mismatch_risk",
    "cust_prior_orders", "cust_prior_return_rate_smoothed",
    "cust_prior_avg_order_value", "days_since_prev_order",
    "account_age_days", "cat_prior_return_rate", "order_value_vs_cust_avg",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
N_SPLITS = 4


def time_split(df, test_frac=0.2):
    df = df.sort_values("order_date").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_date = df.iloc[cutoff_idx]["order_date"]
    train = df[df["order_date"] < cutoff_date].copy()
    test = df[df["order_date"] >= cutoff_date].copy()
    return train, test, cutoff_date


def make_rf_matrix(df, encoder=None, scaler=None, fit=False):
    """RandomForest needs numeric input -> one-hot encode categoricals, scale numerics."""
    cat = df[CAT_FEATURES].astype(str)
    num = df[NUM_FEATURES].astype(float)
    if fit:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_enc = encoder.fit_transform(cat)
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num)
    else:
        cat_enc = encoder.transform(cat)
        num_scaled = scaler.transform(num)
    X = np.hstack([cat_enc, num_scaled])
    return X, encoder, scaler


def main():
    df = pd.read_csv(
        f"{BASE_DIR}/data/orders_featured.csv",
        parse_dates=["order_date", "signup_date"],
    )
    train, test, cutoff = time_split(df)
    train = train.sort_values("order_date").reset_index(drop=True)
    print(f"Time-based split cutoff: {cutoff.date()}")
    print(f"Train: {len(train)} | Test: {len(test)}")

    y_train_full = train["returned"].values

    # ---- Stage 1: generate out-of-fold base predictions on TRAIN via forward-chaining CV ----
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_cb = np.zeros(len(train))
    oof_rf = np.zeros(len(train))
    oof_lr = np.zeros(len(train))
    oof_mask = np.zeros(len(train), dtype=bool)  # tracks rows that got an OOF prediction

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
        tr_fold, val_fold = train.iloc[tr_idx], train.iloc[val_idx]

        # CatBoost base learner
        cb = CatBoostClassifier(
            iterations=400, learning_rate=0.04, depth=5, l2_leaf_reg=6,
            loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=42, verbose=False,
        )
        cb.fit(Pool(tr_fold[ALL_FEATURES], tr_fold["returned"], cat_features=CAT_FEATURES))
        oof_cb[val_idx] = cb.predict_proba(Pool(val_fold[ALL_FEATURES], cat_features=CAT_FEATURES))[:, 1]

        # RandomForest base learner (bagged trees — different variance/bias profile than boosting)
        X_tr, enc, scl = make_rf_matrix(tr_fold, fit=True)
        X_val, _, _ = make_rf_matrix(val_fold, encoder=enc, scaler=scl, fit=False)
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr, tr_fold["returned"])
        oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]

        # Regularized logistic regression (linear model — catches monotone linear
        # signal without the tree models' tendency to fragment it into many splits)
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)
        lr.fit(X_tr, tr_fold["returned"])
        oof_lr[val_idx] = lr.predict_proba(X_val)[:, 1]

        oof_mask[val_idx] = True
        print(f"  fold {fold+1}/{N_SPLITS}: train={len(tr_idx)} val={len(val_idx)} "
              f"cb_prauc={average_precision_score(val_fold['returned'], oof_cb[val_idx]):.3f} "
              f"rf_prauc={average_precision_score(val_fold['returned'], oof_rf[val_idx]):.3f} "
              f"lr_prauc={average_precision_score(val_fold['returned'], oof_lr[val_idx]):.3f}")

    # rows in the very first fold's training block never get an OOF prediction — drop them
    passthrough_train = train[PASSTHROUGH_FEATURES].values
    meta_X = np.column_stack([oof_cb, oof_rf, oof_lr, passthrough_train])[oof_mask]
    meta_y = y_train_full[oof_mask]
    print(f"\nMeta-training rows: {len(meta_y)} (dropped {len(train) - len(meta_y)} with no OOF pred)")

    # ---- Stage 2: train meta-learner on OOF predictions ----
    meta_model = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    meta_model.fit(meta_X, meta_y)
    coef_names = ["catboost", "rf", "logreg"] + PASSTHROUGH_FEATURES
    print("Meta-learner coefficients: " +
          ", ".join(f"{n}={c:.3f}" for n, c in zip(coef_names, meta_model.coef_[0])))

    # ---- Stage 3: refit base learners on FULL train set, predict on held-out test ----
    cb_full = CatBoostClassifier(
        iterations=600, learning_rate=0.04, depth=5, l2_leaf_reg=6,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=42, verbose=False,
    )
    cb_full.fit(Pool(train[ALL_FEATURES], train["returned"], cat_features=CAT_FEATURES))
    test_cb_proba = cb_full.predict_proba(Pool(test[ALL_FEATURES], cat_features=CAT_FEATURES))[:, 1]

    X_train_full, enc_full, scl_full = make_rf_matrix(train, fit=True)
    X_test_rf, _, _ = make_rf_matrix(test, encoder=enc_full, scaler=scl_full, fit=False)
    rf_full = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_full.fit(X_train_full, train["returned"])
    test_rf_proba = rf_full.predict_proba(X_test_rf)[:, 1]

    lr_full = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)
    lr_full.fit(X_train_full, train["returned"])
    test_lr_proba = lr_full.predict_proba(X_test_rf)[:, 1]

    test_passthrough = test[PASSTHROUGH_FEATURES].values
    test_meta_X = np.column_stack([test_cb_proba, test_rf_proba, test_lr_proba, test_passthrough])
    test_ensemble_proba = meta_model.predict_proba(test_meta_X)[:, 1]

    y_test = test["returned"].values
    results = {
        "catboost_solo":      {"pr_auc": average_precision_score(y_test, test_cb_proba),
                                "roc_auc": roc_auc_score(y_test, test_cb_proba)},
        "random_forest_solo": {"pr_auc": average_precision_score(y_test, test_rf_proba),
                                "roc_auc": roc_auc_score(y_test, test_rf_proba)},
        "logreg_solo":        {"pr_auc": average_precision_score(y_test, test_lr_proba),
                                "roc_auc": roc_auc_score(y_test, test_lr_proba)},
        "stacked_ensemble":   {"pr_auc": average_precision_score(y_test, test_ensemble_proba),
                                "roc_auc": roc_auc_score(y_test, test_ensemble_proba)},
    }
    print("\n=== Held-out test set comparison ===")
    for name, m in results.items():
        print(f"{name:20s}  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

    # save everything needed for evaluation + repo
    np.save(f"{BASE_DIR}/outputs/ensemble_test_proba.npy", test_ensemble_proba)
    np.save(f"{BASE_DIR}/outputs/ensemble_catboost_solo_proba.npy", test_cb_proba)
    np.save(f"{BASE_DIR}/outputs/ensemble_rf_solo_proba.npy", test_rf_proba)
    pd.Series(y_test).to_csv(f"{BASE_DIR}/outputs/ensemble_test_labels.csv", index=False)
    test[["order_id", "order_value"]].to_csv(
        f"{BASE_DIR}/outputs/ensemble_test_meta.csv", index=False
    )

    cb_full.save_model(f"{BASE_DIR}/models/catboost_ensemble_member.cbm")
    joblib.dump(rf_full, f"{BASE_DIR}/models/random_forest_ensemble_member.joblib")
    joblib.dump(lr_full, f"{BASE_DIR}/models/logreg_ensemble_member.joblib")
    joblib.dump(meta_model, f"{BASE_DIR}/models/meta_logreg.joblib")
    joblib.dump(enc_full, f"{BASE_DIR}/models/rf_onehot_encoder.joblib")
    joblib.dump(scl_full, f"{BASE_DIR}/models/rf_scaler.joblib")

    with open(f"{BASE_DIR}/outputs/ensemble_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved ensemble models to models/ and predictions/metrics to outputs/")


if __name__ == "__main__":
    main()
