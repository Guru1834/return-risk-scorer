"""
Evaluation with an explicit cost model — this is what the track's judging bar asks for:
"Honest metrics including false-positive cost."

Cost model (documented assumptions, tune per real merchant economics):
  - False negative (missed a return-risk order): merchant eats the full return cost —
    reverse logistics + restocking/write-off, modeled as a fraction of order value.
  - False positive (flagged a good order): friction cost — manual review time and/or
    a customer-experience hit if the flag changes checkout behavior (e.g. blocking COD,
    requiring prepay). Modeled as a fixed small cost, NOT proportional to order value,
    because a false alarm on a ₹200 order and a ₹20,000 order costs the ops team roughly
    the same review time.
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FN_COST_FRACTION = 0.35   # fraction of order value lost when a real return-risk order is missed
FP_COST_FLAT = 45.0       # flat cost (INR) of a false-positive manual review / friction event

OUT = f"{BASE_DIR}/outputs"


def load(prefix):
    proba = np.load(f"{OUT}/{prefix}_test_proba.npy") if prefix == "baseline" else np.load(f"{OUT}/ensemble_test_proba.npy")
    labels = pd.read_csv(f"{OUT}/{prefix}_test_labels.csv").iloc[:, 0].values
    meta = pd.read_csv(f"{OUT}/{prefix}_test_meta.csv")
    return proba, labels, meta


def threshold_table(proba, labels, thresholds=None):
    # Fixed thresholds are uninformative when a model's predicted probabilities are
    # tightly clustered (a known side effect of auto_class_weights="Balanced" on
    # CatBoost) — use score quantiles instead so the table always shows meaningful
    # operating points regardless of how a given model's output is distributed.
    if thresholds is None:
        thresholds = np.unique(np.quantile(proba, [0.99, 0.97, 0.94, 0.90, 0.80, 0.70, 0.55]))[::-1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(labels, pred, zero_division=0),
            "recall": recall_score(labels, pred, zero_division=0),
            "f1": f1_score(labels, pred, zero_division=0),
            "flagged_pct": pred.mean(),
        })
    return pd.DataFrame(rows)


def cost_curve(proba, labels, order_values, thresholds=np.linspace(0.05, 0.9, 35)):
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fn_mask = (pred == 0) & (labels == 1)
        fp_mask = (pred == 1) & (labels == 0)
        fn_cost = (order_values[fn_mask] * FN_COST_FRACTION).sum()
        fp_cost = fp_mask.sum() * FP_COST_FLAT
        total_cost = fn_cost + fp_cost
        rows.append({"threshold": t, "fn_cost": fn_cost, "fp_cost": fp_cost, "total_cost": total_cost,
                      "n_flagged": pred.sum()})
    return pd.DataFrame(rows)


def main():
    base_proba, base_labels, base_meta = load("baseline")
    ens_proba, ens_labels, ens_meta = load("ensemble")

    print("=== Threshold table: BASELINE (CatBoost solo) ===")
    bt = threshold_table(base_proba, base_labels)
    print(bt.to_string(index=False))

    print("\n=== Threshold table: STACKED ENSEMBLE ===")
    et = threshold_table(ens_proba, ens_labels)
    print(et.to_string(index=False))

    base_cost = cost_curve(base_proba, base_labels, base_meta["order_value"].values)
    ens_cost = cost_curve(ens_proba, ens_labels, ens_meta["order_value"].values)

    # "do nothing" cost — baseline where nothing is ever flagged (pure FN cost)
    do_nothing_cost = (base_meta["order_value"].values[base_labels == 1] * FN_COST_FRACTION).sum()
    # "flag everything" cost — pure FP cost
    flag_all_cost = (base_labels == 0).sum() * FP_COST_FLAT

    base_best = base_cost.loc[base_cost["total_cost"].idxmin()]
    ens_best = ens_cost.loc[ens_cost["total_cost"].idxmin()]

    print(f"\n=== Cost analysis (FN={FN_COST_FRACTION*100:.0f}% of order value, FP=₹{FP_COST_FLAT} flat) ===")
    print(f"Do-nothing baseline cost (never flag):     ₹{do_nothing_cost:,.0f}")
    print(f"Flag-everything cost:                       ₹{flag_all_cost:,.0f}")
    print(f"CatBoost solo — best threshold {base_best['threshold']:.2f}: total cost ₹{base_best['total_cost']:,.0f} "
          f"(saves ₹{do_nothing_cost - base_best['total_cost']:,.0f} vs doing nothing)")
    print(f"Stacked ensemble — best threshold {ens_best['threshold']:.2f}: total cost ₹{ens_best['total_cost']:,.0f} "
          f"(saves ₹{do_nothing_cost - ens_best['total_cost']:,.0f} vs doing nothing)")

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    p_b, r_b, _ = precision_recall_curve(base_labels, base_proba)
    p_e, r_e, _ = precision_recall_curve(ens_labels, ens_proba)
    axes[0].plot(r_b, p_b, label=f"CatBoost solo (AP={average_precision_score(base_labels, base_proba):.3f})")
    axes[0].plot(r_e, p_e, label=f"Stacked ensemble (AP={average_precision_score(ens_labels, ens_proba):.3f})")
    axes[0].axhline(base_labels.mean(), color="gray", linestyle="--", label=f"Base rate ({base_labels.mean():.3f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(base_cost["threshold"], base_cost["total_cost"], label="CatBoost solo")
    axes[1].plot(ens_cost["threshold"], ens_cost["total_cost"], label="Stacked ensemble")
    axes[1].axhline(do_nothing_cost, color="gray", linestyle="--", label="Do nothing (never flag)")
    axes[1].set_xlabel("Decision threshold"); axes[1].set_ylabel("Total cost (₹)")
    axes[1].set_title("Cost vs Threshold"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/evaluation_plots.png", dpi=150)
    print(f"\nSaved plots to {OUT}/evaluation_plots.png")

    summary = {
        "cost_model": {"fn_cost_fraction": FN_COST_FRACTION, "fp_cost_flat_inr": FP_COST_FLAT},
        "do_nothing_cost_inr": float(do_nothing_cost),
        "flag_all_cost_inr": float(flag_all_cost),
        "catboost_solo": {
            "best_threshold": float(base_best["threshold"]),
            "total_cost_inr": float(base_best["total_cost"]),
            "savings_vs_do_nothing_inr": float(do_nothing_cost - base_best["total_cost"]),
            "pr_auc": float(average_precision_score(base_labels, base_proba)),
        },
        "stacked_ensemble": {
            "best_threshold": float(ens_best["threshold"]),
            "total_cost_inr": float(ens_best["total_cost"]),
            "savings_vs_do_nothing_inr": float(do_nothing_cost - ens_best["total_cost"]),
            "pr_auc": float(average_precision_score(ens_labels, ens_proba)),
        },
    }
    with open(f"{OUT}/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    bt.to_csv(f"{OUT}/baseline_threshold_table.csv", index=False)
    et.to_csv(f"{OUT}/ensemble_threshold_table.csv", index=False)
    print(f"Saved evaluation_summary.json + threshold tables to {OUT}/")


if __name__ == "__main__":
    main()
