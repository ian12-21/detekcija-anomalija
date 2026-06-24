import sys, os
import numpy as np
from sklearn.metrics import roc_auc_score

# make src/ importable regardless of where pytest runs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torch_anomaly import AutoencoderDetector
from anomaly_utils import anomaly_scores


def _synthetic():
    rng = np.random.RandomState(0)
    inliers = rng.normal(0, 1, size=(300, 8))
    outliers = rng.normal(8, 1, size=(15, 8))
    X = np.vstack([inliers, outliers]).astype(np.float32)
    y = np.r_[np.zeros(300), np.ones(15)]
    return X, y


def test_reconstruction_error_orientation():
    # frauds (outliers) must score HIGHER through the project's anomaly_scores path
    X, y = _synthetic()
    det = AutoencoderDetector(encoder_dims=(6, 3), epochs=60, batch_size=64,
                              device="cpu", random_state=0, verbose=False).fit(X)
    scores = anomaly_scores(det, X, uses_fit_predict=False)
    assert roc_auc_score(y, scores) > 0.8  # fails if decision_function sign is flipped


def test_scores_shape_and_finite():
    X, _ = _synthetic()
    det = AutoencoderDetector(encoder_dims=(6, 3), epochs=5, batch_size=64,
                              device="cpu", verbose=False).fit(X)
    s = det.decision_function(X)
    assert s.shape == (X.shape[0],)
    assert np.all(np.isfinite(s))


def test_cpu_device_runs_without_cuda():
    X, _ = _synthetic()
    det = AutoencoderDetector(device="cpu", epochs=2, batch_size=64, verbose=False)
    det.fit(X)
    assert det.device == "cpu"
    assert det.decision_function(X).shape == (X.shape[0],)
