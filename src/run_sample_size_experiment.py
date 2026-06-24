"""
Sample-size sweep for the unsupervised anomaly-detection comparison.

Reuses the exact preprocessing and model specs from
02_train_and_evaluate.ipynb, but runs every model at several subsample
sizes and writes tables + plots to ../results/<size>/, plus a cross-size
comparison in ../results/comparison/.

Run:  ~/miniconda3/envs/uni/bin/python run_sample_size_experiment.py
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save figures, never show
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

from anomaly_utils import anomaly_scores, mad_threshold, evaluate, MAD_K, DEFAULT_NU

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RANDOM_STATE = 42
SAMPLE_SIZES = [30_000, 80_000, 150_000, 230_000]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "creditcard.csv")
RESULTS = os.path.join(HERE, "..", "results")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]


def size_label(n):
    return f"{n // 1000}k"


def build_models():
    """(name, estimator, uses_fit_predict) — label-free; MAD sets the budget."""
    return [
        ("One-Class SVM",
         OneClassSVM(kernel="rbf", gamma=0.1, nu=DEFAULT_NU), False),
        ("One-Class SVM (SGD)",
         SGDOneClassSVM(nu=DEFAULT_NU, max_iter=1000,
                        learning_rate="optimal", random_state=RANDOM_STATE), False),
        ("Isolation Forest",
         IsolationForest(n_estimators=100, contamination="auto",
                         max_samples="auto", random_state=RANDOM_STATE), False),
        ("Local Outlier Factor",
         LocalOutlierFactor(n_neighbors=20, contamination="auto",
                            novelty=False), True),
        ("Robust Covariance",
         EllipticEnvelope(contamination=0.1, random_state=RANDOM_STATE), False),
    ]


def run_one_size(X, y_true, n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sample_idx = X.sample(n=n, random_state=RANDOM_STATE).index
    X_sample = X.loc[sample_idx]
    y_sample = y_true.loc[sample_idx].to_numpy()
    true_rate = y_sample.mean()

    print(f"\n=== {size_label(n)}: {n:,} rows | "
          f"{y_sample.sum():,} frauds | "
          f"true fraud rate={true_rate:.4%} ===", flush=True)

    results, timing, metrics, reports = {}, {}, [], []

    for name, model, uses_fit_predict in build_models():
        print(f"  {name} ...", end="", flush=True)
        if uses_fit_predict:
            t0 = time.time()
            model.fit_predict(X_sample)
            fit_time = time.time() - t0
            t0 = time.time()
            scores = anomaly_scores(model, X_sample, True)
            pred_time = time.time() - t0
        else:
            t0 = time.time()
            model.fit(X_sample)
            fit_time = time.time() - t0
            t0 = time.time()
            scores = anomaly_scores(model, X_sample, False)
            pred_time = time.time() - t0

        preds = mad_threshold(scores)
        results[name] = preds
        timing[name] = {"Fit Time (s)": fit_time,
                        "Predict Time (s)": pred_time}
        metrics.append({"Model": name,
                        **evaluate(y_sample, preds, scores),
                        "Implied Contam.": preds.mean()})
        reports.append(
            f"{'=' * 50}\n{name}\n{'=' * 50}\n"
            + classification_report(y_sample, preds,
                                    target_names=["Normal", "Fraud"],
                                    zero_division=0))
        print(f" fit={fit_time:.2f}s score={pred_time:.2f}s "
              f"flagged={int(preds.sum()):,} ({preds.mean():.4%})", flush=True)

    metrics_df = pd.DataFrame(metrics).set_index("Model")
    timing_df = pd.DataFrame(timing).T
    timing_df["Total (s)"] = timing_df.sum(axis=1)
    best_f1 = metrics_df["F1 Score"].idxmax()
    best_auc = metrics_df["ROC-AUC"].idxmax()

    # ---- tables ----
    metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"))
    timing_df.round(4).to_csv(os.path.join(out_dir, "timing.csv"))
    with open(os.path.join(out_dir, "classification_reports.txt"), "w") as f:
        f.write("\n\n".join(reports))

    pct_cols = ["Accuracy", "Precision", "Recall", "F1 Score",
                "ROC-AUC", "Implied Contam."]
    disp = metrics_df.copy()
    for c in pct_cols:
        disp[c] = (disp[c] * 100).round(2).astype(str) + " %"
    disp["R2"] = metrics_df["R2"].round(4).astype(str)
    disp = disp[["Accuracy", "Precision", "Recall", "F1 Score",
                 "ROC-AUC", "R2", "Implied Contam."]]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Sample size: {n:,} rows\n"
                f"Frauds: {int(y_sample.sum()):,}\n"
                f"True fraud rate (reference only): {true_rate:.4%}\n"
                f"Best F1: {best_f1} ({metrics_df.loc[best_f1, 'F1 Score']:.2%})\n"
                f"Best ROC-AUC: {best_auc} "
                f"({metrics_df.loc[best_auc, 'ROC-AUC']:.4f})\n\n"
                f"{disp}\n\n"
                f"{timing_df.round(3)}\n")

    # ---- metrics bar chart ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metric_names = ["Accuracy", "Precision", "Recall",
                    "F1 Score", "ROC-AUC", "R2"]
    for ax, metric in zip(axes.ravel(), metric_names):
        vals = metrics_df[metric].values
        bars = ax.bar(range(len(metrics_df)), vals, color=COLORS)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        if metric == "R2":
            lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
            pad = 0.1 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad + 0.05)
            fmt = lambda v: f"{v:.3f}"
        else:
            ax.set_ylim(0, 1.15)
            fmt = lambda v: f"{v:.1%}"
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels(metrics_df.index, rotation=30, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt(v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    plt.suptitle(f"Model Performance — {size_label(n)} sample",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_bar.png"), dpi=120)
    plt.close(fig)

    # ---- confusion matrices ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.ravel()
    for i, (name, preds) in enumerate(results.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix(y_sample, preds),
                                      display_labels=["Normal", "Fraud"])
        disp.plot(ax=axes_flat[i], cmap="Blues", values_format=",d")
        axes_flat[i].set_title(name, fontsize=11, fontweight="bold")
    axes_flat[5].set_visible(False)
    plt.suptitle(f"Confusion Matrices — {size_label(n)} sample",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=120)
    plt.close(fig)

    # ---- timing chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    timing_df[["Fit Time (s)", "Predict Time (s)"]].plot(
        kind="barh", stacked=True, ax=ax, color=["#2196F3", "#FF9800"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Computational Efficiency — {size_label(n)} sample")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "timing.png"), dpi=120)
    plt.close(fig)

    return metrics_df, timing_df


def main():
    print("Loading and preprocessing ...", flush=True)
    df = pd.read_csv(DATA).drop(columns=["Time"])
    df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
    X = df.drop(columns=["Class"])
    y_true = df["Class"]
    print(f"  {X.shape[0]:,} rows | "
          f"{y_true.sum():,} frauds ({y_true.mean():.4%})", flush=True)

    rows = []
    for n in SAMPLE_SIZES:
        m_df, t_df = run_one_size(X, y_true, n,
                                  os.path.join(RESULTS, size_label(n)))
        for model in m_df.index:
            rows.append({
                "Sample Size": n,
                "Model": model,
                "Accuracy": m_df.loc[model, "Accuracy"],
                "Precision": m_df.loc[model, "Precision"],
                "Recall": m_df.loc[model, "Recall"],
                "F1 Score": m_df.loc[model, "F1 Score"],
                "ROC-AUC": m_df.loc[model, "ROC-AUC"],
                "R2": m_df.loc[model, "R2"],
                "Implied Contamination": m_df.loc[model, "Implied Contam."],
                "Total Time (s)": t_df.loc[model, "Total (s)"],
            })

    # ---- cross-size comparison ----
    comp_dir = os.path.join(RESULTS, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    long_df = pd.DataFrame(rows)
    long_df.to_csv(os.path.join(comp_dir, "all_metrics.csv"), index=False)

    models = long_df["Model"].unique()
    for metric, fname in [("F1 Score", "f1_vs_size.png"),
                          ("Recall", "recall_vs_size.png"),
                          ("Precision", "precision_vs_size.png"),
                          ("ROC-AUC", "rocauc_vs_size.png"),
                          ("R2", "r2_vs_size.png"),
                          ("Total Time (s)", "runtime_vs_size.png")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, c in zip(models, COLORS):
            sub = long_df[long_df["Model"] == model]
            ax.plot(sub["Sample Size"], sub[metric], marker="o",
                    label=model, color=c, linewidth=2)
        ax.set_xlabel("Sample size (rows)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs. sample size")
        if metric == "Total Time (s)":
            ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.set_xticks(SAMPLE_SIZES)
        ax.set_xticklabels([size_label(n) for n in SAMPLE_SIZES])
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, fname), dpi=120)
        plt.close(fig)

    # pivot tables (model x size) per metric
    with open(os.path.join(comp_dir, "summary.md"), "w") as f:
        f.write("# Sample-size sweep — summary\n\n")
        for metric in ["F1 Score", "Recall", "Precision", "ROC-AUC", "R2",
                       "Implied Contamination", "Total Time (s)"]:
            piv = long_df.pivot(index="Model", columns="Sample Size",
                                values=metric)
            piv.columns = [size_label(c) for c in piv.columns]
            f.write(f"## {metric}\n\n{piv.round(4).to_markdown()}\n\n")

    print(f"\nDone. Results in {os.path.abspath(RESULTS)}", flush=True)


if __name__ == "__main__":
    main()
