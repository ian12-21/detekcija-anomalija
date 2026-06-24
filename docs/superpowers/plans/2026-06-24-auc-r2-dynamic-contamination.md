# ROC-AUC + R² Metrics and MAD-Based Contamination — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ROC-AUC and R² metrics and replace the label-derived fixed contamination with a per-model, data-driven MAD cutoff, across the sweep script and notebook 02, then regenerate results and the README.

**Architecture:** All three features share one new primitive — a unified per-model anomaly score (higher = more anomalous). It is extracted into a new `src/anomaly_utils.py` module (scoring, MAD thresholding, metric evaluation) that both `run_sample_size_experiment.py` and `02_train_and_evaluate.ipynb` import, eliminating the current pipeline duplication.

**Tech Stack:** Python 3.10+, scikit-learn, NumPy, pandas, matplotlib, seaborn; pytest for unit tests; Jupyter/nbconvert for notebook execution.

## Global Constraints

- **No new runtime dependencies.** `roc_auc_score` / `r2_score` are already in scikit-learn; the MAD cutoff is pure NumPy.
- **`MAD_K = 3.0`** (modified z-score cutoff; tunable constant). **`DEFAULT_NU = 0.05`** (label-free `nu` for the One-Class SVM family).
- **Score convention:** higher = more anomalous, obtained by negating `decision_function` (or `negative_outlier_factor_` for LOF/`fit_predict`).
- **R² basis:** `r2_score(y_true, preds)` on binary predictions (NOT on scores).
- **Budget mode:** the MAD cutoff fully **replaces** the old `contamination = max(y_sample.mean(), 1e-4)`. `y_sample.mean()` is retained only as a reference number in printouts/summaries, never to drive a model.
- **Metric column order everywhere:** `Accuracy, Precision, Recall, F1 Score, ROC-AUC, R2, Implied Contam.`
- **Prerequisite (environment):** an interpreter with numpy, pandas, scikit-learn, matplotlib, seaborn (plus pytest and jupyter for tasks 1, 4, 5) must be available. On the dev machine at planning time none was found — resolve this before Task 1 (point to the existing ML env, or create a venv and `pip install pandas numpy scikit-learn matplotlib seaborn jupyter pytest`). In commands below, `python` means that environment's interpreter.

---

## File Structure

- **Create** `src/anomaly_utils.py` — the only home for `anomaly_scores`, `mad_threshold`, `evaluate`, and the `MAD_K` / `DEFAULT_NU` constants.
- **Create** `tests/test_anomaly_utils.py` — pytest unit tests for the pure helpers (highest-risk logic: sign and threshold).
- **Modify** `src/run_sample_size_experiment.py` — import the module; label-free models; MAD predictions; 6 metrics + implied contamination; 2×3 metrics bar; cross-size ROC-AUC/R²/contamination outputs and plots.
- **Modify** `src/02_train_and_evaluate.ipynb` — mirror via the module in cells 1, 6, 8, 9, 11, 13.
- **Modify** `results/**` — regenerated artifacts (Task 4).
- **Modify** `README.md` — refreshed result tables + contamination narrative (Task 5).

---

## Task 1: Shared `anomaly_utils` module (TDD)

**Files:**
- Create: `src/anomaly_utils.py`
- Test: `tests/test_anomaly_utils.py`

**Interfaces:**
- Produces:
  - `anomaly_scores(model, X, uses_fit_predict: bool) -> np.ndarray` (higher = more anomalous)
  - `mad_threshold(scores, k: float = MAD_K) -> np.ndarray[int]` (0/1 flags)
  - `evaluate(y_true, preds, scores) -> dict` with keys `Accuracy, Precision, Recall, F1 Score, ROC-AUC, R2`
  - constants `MAD_K = 3.0`, `DEFAULT_NU = 0.05`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_anomaly_utils.py`:

```python
import sys, os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# make src/ importable regardless of where pytest runs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from anomaly_utils import anomaly_scores, mad_threshold, evaluate, MAD_K, DEFAULT_NU


def test_constants():
    assert MAD_K == 3.0
    assert DEFAULT_NU == 0.05


def test_mad_threshold_flags_single_outlier():
    scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], dtype=float)
    flags = mad_threshold(scores)
    assert flags.sum() == 1
    assert flags[-1] == 1
    assert set(np.unique(flags)).issubset({0, 1})


def test_mad_threshold_no_outliers():
    scores = np.arange(11, dtype=float)  # spread but no extreme tail
    assert mad_threshold(scores).sum() == 0


def test_mad_threshold_constant_does_not_crash():
    scores = np.ones(10, dtype=float)  # MAD == 0 -> guarded
    flags = mad_threshold(scores)
    assert flags.sum() == 0


def test_anomaly_scores_orientation_higher_is_more_anomalous():
    rng = np.random.RandomState(0)
    inliers = rng.normal(0, 0.5, size=(100, 2))
    outliers = rng.normal(8, 0.5, size=(8, 2))
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(100), np.ones(8)]
    model = IsolationForest(contamination="auto", random_state=0).fit(X)
    scores = anomaly_scores(model, X, uses_fit_predict=False)
    assert roc_auc_score(y, scores) > 0.8  # fails if the sign is flipped


def test_evaluate_keys_and_perfect_case():
    y = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    m = evaluate(y, preds, scores)
    assert set(m) == {"Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "R2"}
    assert m["Accuracy"] == 1.0
    assert m["ROC-AUC"] == 1.0
    assert m["R2"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_anomaly_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anomaly_utils'`.

- [ ] **Step 3: Implement the module**

Create `src/anomaly_utils.py`:

```python
"""Shared scoring, thresholding, and metric helpers for the anomaly-detection
comparison. Imported by both run_sample_size_experiment.py and notebook 02 so the
scoring / contamination / metric logic lives in exactly one place.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, r2_score,
)

MAD_K = 3.0          # modified z-score cutoff (Iglewicz-Hoaglin textbook value is 3.5)
DEFAULT_NU = 0.05    # label-free nu for the One-Class SVM family


def anomaly_scores(model, X, uses_fit_predict):
    """Unified anomaly score where higher = more anomalous.

    sklearn detectors return decision_function > 0 for inliers; LOF in
    fit_predict mode exposes negative_outlier_factor_ instead. Negating both
    yields a consistent "higher = more anomalous" score aligned with y=1 (fraud),
    which is what roc_auc_score and the MAD threshold expect.
    """
    if uses_fit_predict:
        return -model.negative_outlier_factor_
    return -model.decision_function(X)


def mad_threshold(scores, k=MAD_K):
    """Data-driven anomaly flags via the robust modified z-score.

    Flags points whose score lies more than k scaled-MADs above the median
    (1.4826 * MAD approximates the std under normality). The fraction of 1s is
    the dynamic contamination. Shift-invariant: a constant offset in the input
    leaves the result unchanged. Returns an int 0/1 array.
    """
    scores = np.asarray(scores, dtype=float)
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    scale = 1.4826 * mad
    if scale == 0:
        scale = np.finfo(float).eps
    z = (scores - med) / scale
    return (z > k).astype(int)


def evaluate(y_true, preds, scores):
    """All six performance metrics for one model.

    ROC-AUC uses the continuous scores (threshold-independent) and is NaN-guarded
    against a single-class y_true; R2 uses the binary predictions.
    """
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1 Score": f1_score(y_true, preds, zero_division=0),
        "ROC-AUC": auc,
        "R2": r2_score(y_true, preds),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_anomaly_utils.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/anomaly_utils.py tests/test_anomaly_utils.py
git commit -m "feat: add anomaly_utils (scores, MAD threshold, 6-metric evaluate)"
```

---

## Task 2: Refactor the sweep script to use the module

**Files:**
- Modify: `src/run_sample_size_experiment.py`

**Interfaces:**
- Consumes from Task 1: `anomaly_scores`, `mad_threshold`, `evaluate`, `MAD_K`, `DEFAULT_NU`.
- Produces: regenerated `results/<size>/` and `results/comparison/` content with the new columns/plots (full data written in Task 4; here verified at 30k only).

- [ ] **Step 1: Replace the metrics import and add the module import**

In `src/run_sample_size_experiment.py`, replace the import block (currently lines 23-31, the `from sklearn.metrics import (...)` through `EllipticEnvelope`) with:

```python
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

from anomaly_utils import anomaly_scores, mad_threshold, evaluate, MAD_K, DEFAULT_NU
```

- [ ] **Step 2: Make `build_models` label-free and delete `convert_predictions`**

Replace the whole `build_models(contamination)` function and the `convert_predictions` function with:

```python
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
```

(`convert_predictions` is no longer referenced anywhere — remove it entirely.)

- [ ] **Step 3: Rewrite the per-model loop in `run_one_size`**

Replace from the `contamination = max(...)` line through the end of the `for name, model, uses_fit_predict in build_models(contamination):` loop (currently lines 79-117) with:

```python
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
```

- [ ] **Step 4: Update the post-loop dataframes, `best`, and the summary block**

Replace the block from `metrics_df = pd.DataFrame(...)` through the `with open(... "summary.txt" ...)` write (currently lines 119-135) with:

```python
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
```

- [ ] **Step 5: Make the metrics bar chart 2×3 (six panels, R² dynamic range)**

Replace the metrics-bar-chart block (currently lines 137-155, `fig, axes = plt.subplots(2, 2, ...)` through its `plt.close(fig)`) with:

```python
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
```

(The confusion-matrix and timing-chart blocks are unchanged.)

- [ ] **Step 6: Add the new columns to the cross-size rows in `main`**

In `main()`, replace the `rows.append({...})` dict (currently lines 199-207) with:

```python
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
```

- [ ] **Step 7: Add ROC-AUC/R² trend plots and extend the pivot summary**

In `main()`, replace the comparison-plot metric list (currently lines 216-219) with:

```python
    for metric, fname in [("F1 Score", "f1_vs_size.png"),
                          ("Recall", "recall_vs_size.png"),
                          ("Precision", "precision_vs_size.png"),
                          ("ROC-AUC", "rocauc_vs_size.png"),
                          ("R2", "r2_vs_size.png"),
                          ("Total Time (s)", "runtime_vs_size.png")]:
```

And replace the pivot-summary metric list (currently line 240) with:

```python
        for metric in ["F1 Score", "Recall", "Precision", "ROC-AUC", "R2",
                       "Implied Contamination", "Total Time (s)"]:
```

- [ ] **Step 8: Smoke-test at 30k only**

Temporarily set the sweep to one small size: change `SAMPLE_SIZES = [30_000, 80_000, 150_000, 230_000]` to `SAMPLE_SIZES = [30_000]`.

Run: `cd src && python run_sample_size_experiment.py`
Expected: completes without error; console shows `flagged=… (…%)` per model; `results/30k/metrics.csv` now has `ROC-AUC,R2,Implied Contam.` columns; `results/30k/summary.txt` shows the 6-metric table + `Best ROC-AUC`; `results/comparison/` contains `rocauc_vs_size.png` and `r2_vs_size.png`.

Verify sanity:

```bash
python -c "import pandas as pd; d=pd.read_csv('results/30k/metrics.csv'); print(d); assert d['ROC-AUC'].between(0,1).all(); assert d['Implied Contam.'].between(0,1).all(); print('OK')"
```

Then **revert** `SAMPLE_SIZES` back to `[30_000, 80_000, 150_000, 230_000]`.

- [ ] **Step 9: Commit (code only, not regenerated results yet)**

```bash
git add src/run_sample_size_experiment.py
git commit -m "feat: MAD contamination + ROC-AUC/R2 in sweep script"
```

---

## Task 3: Mirror the changes into notebook 02

**Files:**
- Modify: `src/02_train_and_evaluate.ipynb` (code cells 1, 6, 8, 9, 11, 13)

**Interfaces:**
- Consumes from Task 1: `anomaly_scores`, `mad_threshold`, `evaluate`, `DEFAULT_NU`.

Use the NotebookEdit tool (or edit cell `source` arrays) to replace each cell's full source. Cell indices are from the current notebook (`02_train_and_evaluate.ipynb`).

- [ ] **Step 1: Cell 1 — imports**

Replace cell 1 source with:

```python
import pandas as pd
import numpy as np
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
from anomaly_utils import anomaly_scores, mad_threshold, evaluate, DEFAULT_NU
import time
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
RANDOM_STATE = 42
```

- [ ] **Step 2: Cell 6 — drop label-derived CONTAMINATION**

Replace cell 6 source with:

```python
SAMPLE_SIZE = 80000

sample_idx = X.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).index
X_sample = X.loc[sample_idx]
y_sample = y_true.loc[sample_idx].to_numpy()

TRUE_RATE = y_sample.mean()  # ground-truth fraud rate, for reference only

print(f"Subsample: {len(X_sample):,} rows")
print(f"Frauds in subsample: {y_sample.sum():,}")
print(f"True fraud rate (reference only): {TRUE_RATE:.4%}")
print("Anomaly budget is now data-driven (per-model MAD threshold).")
```

- [ ] **Step 3: Cell 8 — label-free model specs, drop convert_predictions**

Replace cell 8 source with:

```python
# (name, estimator, uses_fit_predict) — label-free; MAD sets the anomaly budget
model_specs = [
    ("One-Class SVM",
     OneClassSVM(kernel="rbf", gamma=0.1, nu=DEFAULT_NU), False),
    ("One-Class SVM (SGD)",
     SGDOneClassSVM(nu=DEFAULT_NU, max_iter=1000, learning_rate="optimal",
                    random_state=RANDOM_STATE), False),
    ("Isolation Forest",
     IsolationForest(n_estimators=100, contamination="auto",
                     max_samples="auto", random_state=RANDOM_STATE), False),
    ("Local Outlier Factor",
     LocalOutlierFactor(n_neighbors=20, contamination="auto",
                        novelty=False), True),
    ("Robust Covariance",
     EllipticEnvelope(contamination=0.1, random_state=RANDOM_STATE), False),
]

results = {}
scores_map = {}
timing = {}
```

- [ ] **Step 4: Cell 9 — training loop uses scores + MAD**

Replace cell 9 source with:

```python
for name, model, uses_fit_predict in model_specs:
    print(f"Training {name} ...")
    if uses_fit_predict:
        start = time.time()
        model.fit_predict(X_sample)
        fit_time = time.time() - start
        start = time.time()
        scores = anomaly_scores(model, X_sample, True)
        pred_time = time.time() - start
    else:
        start = time.time()
        model.fit(X_sample)
        fit_time = time.time() - start
        start = time.time()
        scores = anomaly_scores(model, X_sample, False)
        pred_time = time.time() - start

    preds = mad_threshold(scores)
    results[name] = preds
    scores_map[name] = scores
    timing[name] = {"fit": fit_time, "predict": pred_time}
    print(f"  done. fit={fit_time:.2f}s score={pred_time:.2f}s | "
          f"flagged {preds.sum():,} / {len(preds):,} "
          f"({preds.mean():.4%}) as anomalies")
```

- [ ] **Step 5: Cell 11 — six metrics + implied contamination**

Replace cell 11 source with:

```python
metrics = []
for name, preds in results.items():
    metrics.append({"Model": name,
                    **evaluate(y_sample, preds, scores_map[name]),
                    "Implied Contam.": preds.mean()})

metrics_df = pd.DataFrame(metrics).set_index("Model")

pct_cols = ["Accuracy", "Precision", "Recall", "F1 Score",
            "ROC-AUC", "Implied Contam."]
disp = metrics_df.copy()
for c in pct_cols:
    disp[c] = (disp[c] * 100).round(2).astype(str) + " %"
disp["R2"] = metrics_df["R2"].round(4).astype(str)
disp = disp[["Accuracy", "Precision", "Recall", "F1 Score",
             "ROC-AUC", "R2", "Implied Contam."]]

print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(disp)
best_f1 = metrics_df["F1 Score"].idxmax()
best_auc = metrics_df["ROC-AUC"].idxmax()
print(f"\nBest F1 Score: {best_f1} ({metrics_df.loc[best_f1, 'F1 Score']:.2%})")
print(f"Best ROC-AUC: {best_auc} ({metrics_df.loc[best_auc, 'ROC-AUC']:.4f})")
```

- [ ] **Step 6: Cell 13 — 2×3 metrics bar chart**

Replace cell 13 source with:

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "R2"]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

for ax, metric in zip(axes.ravel(), metric_names):
    values = metrics_df[metric].values
    bars = ax.bar(range(len(metrics_df)), values, color=colors)
    ax.set_title(metric, fontsize=14, fontweight="bold")
    if metric == "R2":
        lo, hi = min(values.min(), 0.0), max(values.max(), 0.0)
        pad = 0.1 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad + 0.05)
        fmt = lambda v: f"{v:.3f}"
    else:
        ax.set_ylim(0, 1.15)
        fmt = lambda v: f"{v:.1%}"
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df.index, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
```

(Cells 15 confusion matrices, 17 timing, 19 classification reports are unchanged.)

- [ ] **Step 7: Execute the notebook to refresh stored outputs**

Run: `cd src && python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 02_train_and_evaluate.ipynb`
Expected: completes with no cell errors; the metrics table cell now prints `ROC-AUC`, `R2`, `Implied Contam.` and `Best ROC-AUC`; the bar chart shows six panels.

- [ ] **Step 8: Commit**

```bash
git add src/02_train_and_evaluate.ipynb
git commit -m "feat: mirror MAD contamination + ROC-AUC/R2 in notebook 02"
```

---

## Task 4: Regenerate the full results sweep

**Files:**
- Modify: `results/30k/**`, `results/80k/**`, `results/150k/**`, `results/230k/**`, `results/comparison/**`

- [ ] **Step 1: Confirm full sizes are restored**

Verify `src/run_sample_size_experiment.py` has `SAMPLE_SIZES = [30_000, 80_000, 150_000, 230_000]` (reverted from the Task 2 smoke test).

- [ ] **Step 2: Run the full sweep (~10–15 min, OCSVM-bound at 230k)**

Run: `cd src && python run_sample_size_experiment.py`
Expected: four `=== Nk: … ===` blocks complete; final line `Done. Results in …/results`.

- [ ] **Step 3: Sanity-check the regenerated comparison**

```bash
python -c "import pandas as pd; d=pd.read_csv('results/comparison/all_metrics.csv'); print(d[['Sample Size','Model','F1 Score','ROC-AUC','R2','Implied Contamination']].to_string()); assert d['ROC-AUC'].between(0,1).all()"
```
Expected: every row has a sensible ROC-AUC ∈ [0,1]; `Implied Contamination` is in the same ballpark as the true fraud rate (order 1e-3 to 1e-1, not 0 for every model).

- [ ] **Step 4: Commit the regenerated artifacts**

```bash
git add results/
git commit -m "chore: regenerate results with MAD contamination + ROC-AUC/R2"
```

---

## Task 5: Refresh the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the regenerated numbers**

Read `results/comparison/summary.md` and each `results/<size>/summary.txt`. These hold the exact values for the tables below.

- [ ] **Step 2: Update the "Common Subsample" / contamination prose**

In the "Common Subsample (fair comparison)" and "Fully Unsupervised Setup" sections, replace the description of `contamination` = observed fraud fraction with the new behaviour: each model is scored by its own continuous anomaly score, and anomalies are flagged by a per-model robust MAD cutoff (`MAD_K = 3.0`), so the anomaly budget is **data-driven and label-free** rather than set from the true fraud rate. Note One-Class SVM now uses a fixed `nu = 0.05`.

- [ ] **Step 3: Update the 80k results table (currently ~lines 105-112)**

Add `ROC-AUC` and `R²` columns and replace all five rows with the values from `results/80k/summary.txt`. Update the "Winner" line to cite both Best F1 and Best ROC-AUC.

- [ ] **Step 4: Update the sample-size tables (currently ~lines 164-180)**

Replace the "F1 Score by sample size" numbers from `results/comparison/summary.md`, and add a new "ROC-AUC by sample size" table from the same file. Update the surrounding analysis bullets where specific numbers are quoted.

- [ ] **Step 5: Resolve the contamination limitation**

In the "Limitations" section, remove/rewrite the bullet stating `contamination` is a fixed threshold equal to the true fraud rate — it is now resolved. Note ROC-AUC as the threshold-independent metric and the MAD cutoff as the data-driven budget; add a one-line caveat that the MAD cutoff has its own knob (`MAD_K`).

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: update README for MAD contamination and ROC-AUC/R2 metrics"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** ROC-AUC (Tasks 1-4), R² on binary preds (Tasks 1-4), MAD contamination replacing fixed budget (Tasks 1-4), shared module (Task 1), script (Task 2), notebook (Task 3), regenerate results (Task 4), README incl. resolving the contamination limitation (Task 5), tests for the pure helpers (Task 1). All spec sections map to a task.
- **Type/name consistency:** `anomaly_scores(model, X, uses_fit_predict)`, `mad_threshold(scores, k=MAD_K)`, `evaluate(y_true, preds, scores)` and the metric keys/column order are identical across Tasks 1-3. Cross-size key `Implied Contamination` vs. per-size column `Implied Contam.` is mapped explicitly in Task 2 Step 6.
- **Placeholders:** none — every code/edit step shows full content; README task specifies exact source files for the data-dependent numbers.
