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
