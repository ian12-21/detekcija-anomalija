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
