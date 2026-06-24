# Design: ROC-AUC + R² metrics and data-driven (MAD) contamination

**Date:** 2026-06-24
**Branch:** `feature/auc-r2-dynamic-contamination`
**Status:** Approved

## Problem

The unsupervised anomaly-detection comparison currently reports four metrics —
Accuracy, Precision, Recall, F1 — all derived from **thresholded binary
predictions**. Two gaps:

1. **No threshold-independent metric.** Every reported number depends on a single
   contamination threshold, so the comparison cannot separate a model's *ranking*
   quality from the chosen cutoff.
2. **The contamination budget uses the labels.** `contamination =
   max(y_sample.mean(), 1e-4)` sets the anomaly budget from the *true* fraud rate,
   so the "fully unsupervised" claim is overstated and the budget is not
   data-driven.

## Goals

- Add **ROC-AUC** (threshold-independent, from continuous anomaly scores).
- Add **R²** (coefficient of determination, computed on the binary predictions).
- Replace the label-derived fixed contamination with a **data-driven MAD cutoff**
  applied per model to its own score distribution.
- Apply all three to **both** `src/run_sample_size_experiment.py` and
  `src/02_train_and_evaluate.ipynb`, then regenerate `results/` and refresh the
  README tables.

## Non-goals

- Per-model hyperparameter tuning beyond what the contamination change requires.
- Adding a second "fixed-budget vs dynamic-budget" comparison mode — the dynamic
  budget **replaces** the fixed one.
- New runtime dependencies.

## Key decisions (locked)

| Decision | Choice |
|---|---|
| Contamination rule | Robust MAD cutoff (per-model, data-driven) |
| Budget mode | Replace the label-derived budget entirely |
| R² basis | `r2_score(y_true, preds)` on binary predictions |
| Scope | Script + notebook 02, then regenerate results + README |

## Architecture

All three features are powered by one new primitive — a **unified per-model
anomaly score** (higher = more anomalous). Extract it once into a shared module
imported by both the script and the notebook, removing the current duplication of
the pipeline across the two files.

### New module: `src/anomaly_utils.py`

Pure, deterministic, independently testable functions:

```python
MAD_K = 3.0          # modified z-score cutoff (Iglewicz-Hoaglin uses 3.5)
DEFAULT_NU = 0.05    # label-free nu for the One-Class SVM family

def anomaly_scores(model, X, uses_fit_predict):
    """Unified anomaly score, higher = more anomalous.
    fit_predict models (LOF): -model.negative_outlier_factor_
    else:                     -model.decision_function(X)
    """

def mad_threshold(scores, k=MAD_K):
    """Data-driven anomaly flags via robust modified z-score.
    med = median(scores); mad = median(|scores - med|)
    z = (scores - med) / (1.4826 * mad)   # 1.4826*MAD ~= std under normality
    flag where z > k. Guards mad == 0. Returns int 0/1 array.
    The fraction flagged IS the dynamic contamination.
    """

def evaluate(y_true, preds, scores):
    """Returns dict: Accuracy, Precision, Recall, F1 Score, ROC-AUC, R2.
    ROC-AUC = roc_auc_score(y_true, scores)  (guarded -> np.nan if single-class)
    R2      = r2_score(y_true, preds)
    """
```

**Why these definitions are correct:**
- sklearn `decision_function` is positive for inliers, negative for outliers, so
  negating yields "higher = more anomalous", which aligns score direction with
  `y=1` (fraud) for `roc_auc_score`.
- LOF in `fit_predict` mode exposes no `decision_function`; its
  `negative_outlier_factor_` attribute (more negative = more outlier) is negated
  to the same convention.
- MAD is **shift-invariant**, so any constant offset sklearn applies to
  `decision_function` (e.g. EllipticEnvelope's contamination-dependent offset)
  does not change the MAD-thresholded result. This is what lets us leave
  per-model contamination params at defaults and still get a clean, comparable
  cutoff.

### Label-free model setup

`build_models()` no longer takes a contamination argument:

| Model | Label-free setting |
|---|---|
| One-Class SVM (RBF) | fixed `nu=DEFAULT_NU` |
| One-Class SVM (SGD) | fixed `nu=DEFAULT_NU` |
| Isolation Forest | `contamination="auto"` |
| Local Outlier Factor | `contamination="auto"`, `novelty=False` |
| Robust Covariance | `contamination=0.1` (offset only; MAD overrides) |

Per-model flow becomes: **fit → `anomaly_scores()` → `mad_threshold()` →
predictions → `evaluate()`**. The native `predict()` / `convert_predictions()`
path is removed; MAD is the sole decision rule. `y_sample.mean()` is no longer
used to drive any model — the pipeline is genuinely unsupervised. Each model also
reports **implied contamination = `preds.mean()`**.

Timing: `Fit Time` unchanged; `Predict Time` now measures the scoring step
(`decision_function`) rather than `predict`. LOF keeps `fit_predict` as fit time
with predict time 0 (scores come from the fitted attribute).

## Outputs

### Per size (`results/<size>/`)
- `metrics.csv` — add `ROC-AUC`, `R2` columns.
- `summary.txt` — 6-metric table; per-model **implied contamination** vs. the true
  fraud rate; report **Best F1** and **Best ROC-AUC**.
- `metrics_bar.png` — 2×2 grid → **2×3** (six panels). The R² panel uses a dynamic
  y-range because R² can be slightly negative; ROC-AUC keeps the 0–1.15 range.
- `confusion_matrices.png`, `timing.png` — unchanged in structure.

### Cross-size (`results/comparison/`)
- `all_metrics.csv` and `summary.md` pivots — add `ROC-AUC`, `R2`, and
  `Implied Contamination`.
- New trend plots: `rocauc_vs_size.png`, `r2_vs_size.png`.

## Notebook `02_train_and_evaluate.ipynb`

Mirror the script by importing `anomaly_utils`. Update: the imports cell
(add `roc_auc_score`, `r2_score` or import the helpers), the model-build /
contamination cell (label-free + MAD), and the metrics cell (6 metrics +
implied contamination). Re-execute to refresh stored outputs.

## README

Refresh the results tables (with new ROC-AUC / R² columns and dynamic-budget
numbers) and rewrite the contamination notes. This **resolves** the "fixed
contamination is a label-derived threshold" item currently listed under
Limitations.

## Verification

- `tests/test_anomaly_utils.py` (pytest), focused on the highest-risk logic:
  - `mad_threshold` flags the obvious outliers in a synthetic array and none in a
    flat array; handles `mad == 0` without crashing.
  - score orientation: on separable synthetic data, `roc_auc_score(y,
    anomaly_scores(...))` > 0.5 (guards against a sign flip).
- Smoke-run the script at 30k; confirm ROC-AUC ∈ [0, 1] and implied contamination
  is sane (same order as the true fraud rate), then full regenerate (~10 min,
  dominated by One-Class SVM at 230k).

## Risks / implications

- **Every existing result number changes** (new budget). Intended.
- One-Class SVM with fixed `nu=0.05` (vs. `nu≈0.0016`) behaves differently —
  likely flags more points. Documented as a deliberate label-free choice.
- MAD can flag zero points for a near-degenerate score distribution; metrics use
  `zero_division=0` and the ROC-AUC guard, so this is handled.

## Build sequence

1. `src/anomaly_utils.py` + `tests/test_anomaly_utils.py` (TDD on the pure
   helpers).
2. Refactor `run_sample_size_experiment.py` to use the module (label-free models,
   MAD predictions, 6 metrics, implied contamination, updated plots/summaries,
   cross-size additions).
3. Mirror into `02_train_and_evaluate.ipynb`.
4. Regenerate `results/` (smoke 30k → full sweep) and re-execute the notebook.
5. Refresh README tables and contamination narrative.
