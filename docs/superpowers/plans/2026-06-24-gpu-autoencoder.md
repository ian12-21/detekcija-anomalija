# GPU Autoencoder Anomaly Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPU-accelerated PyTorch autoencoder as a 6th unsupervised anomaly detector, scored by per-row reconstruction error, integrated into the existing comparison on the same ROC-AUC footing.

**Architecture:** A new `src/torch_anomaly.py` exposes `AutoencoderDetector` with a scikit-learn-style `fit` / `decision_function` API. `decision_function` returns the **negated** reconstruction error, so the project's existing `anomaly_scores(model, X, uses_fit_predict=False)` (which computes `-decision_function`) recovers "higher = more anomalous" with no change to `anomaly_utils.py`. The detector is registered as an unsupervised model and flows through the existing `anomaly_scores → mad_threshold → evaluate` path.

**Tech Stack:** Python 3.10, PyTorch (CUDA), NumPy, scikit-learn, pandas, matplotlib — existing project plus `torch`.

## Global Constraints

- **`anomaly_utils.py` is the metric single source of truth — do not modify it.** The new detector adapts to it, not the reverse.
- **Score convention (verbatim):** `decision_function(X)` returns `-per_row_MSE`, so `anomaly_scores(model, X, False) == -decision_function(X)` is the reconstruction error (higher = more anomalous), aligned with `y=1` (fraud).
- **Device:** `cuda` if available else `cpu`; the chosen device is printed once; the CPU path must run end-to-end with no GPU.
- **Reproducibility:** seed torch with `RANDOM_STATE = 42`. GPU reductions are not bit-deterministic; that is acceptable.
- **Fixed hyperparameters (no tuning, no early stopping):** `encoder_dims=(20, 14)`, `epochs=30`, `batch_size=2048`, `lr=1e-3`.
- **Inputs are already standardized** (PCA features + StandardScaled `Amount`), so the AE uses a **linear output layer + MSE loss** (no sigmoid).
- **The comparison now has 7 models** (6 unsupervised + the Logistic Regression baseline). `COLORS` (currently 6) and the confusion-matrix grid (currently 2×3 = 6) must grow.
- **Tests force `device="cpu"`** so they run on any machine.
- **Commit style:** short, concise, no trailers.

---

### Task 1: Install CUDA PyTorch and verify the GPU

**Files:** none (environment setup).

**Interfaces:**
- Produces: a working `import torch` with `torch.cuda.is_available() == True` on the RTX 3050 Ti.

- [ ] **Step 1: Install a CUDA-enabled PyTorch wheel**

Run (the `cu124` wheel is backward-compatible with the installed driver 610.62; if you prefer a different CUDA series, pick the matching wheel from pytorch.org):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

- [ ] **Step 2: Verify CUDA is visible to PyTorch**

Run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"
```

Expected: a line like `2.x.x+cu124 True NVIDIA GeForce RTX 3050 Ti Laptop GPU`.
If it prints `False`, the CPU-only wheel was installed — uninstall (`pip uninstall -y torch`) and re-run Step 1 with the `--index-url` flag.

- [ ] **Step 3: No commit** (environment change only, nothing tracked).

---

### Task 2: `AutoencoderDetector` (TDD)

**Files:**
- Create: `src/torch_anomaly.py`
- Test: `tests/test_torch_anomaly.py`

**Interfaces:**
- Consumes: `anomaly_utils.anomaly_scores` (existing, for the orientation test).
- Produces:
  - `class AutoencoderDetector(encoder_dims=(20, 14), epochs=30, batch_size=2048, lr=1e-3, device=None, random_state=42, verbose=True)`
  - `.fit(X) -> self` — `X` is a 2-D array-like / DataFrame of float features.
  - `.decision_function(X) -> np.ndarray` of shape `(n,)`, dtype float, equal to `-per_row_MSE`.
  - `.device` attribute: the resolved device string (`"cuda"` or `"cpu"`).
  - Registered later as `("Autoencoder (GPU)", AutoencoderDetector(...), False)` (i.e. `uses_fit_predict=False`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_torch_anomaly.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_torch_anomaly.py -v
```

Expected: collection/import error `ModuleNotFoundError: No module named 'torch_anomaly'` (file does not exist yet).

- [ ] **Step 3: Implement `AutoencoderDetector`**

Create `src/torch_anomaly.py`:

```python
"""GPU autoencoder anomaly detector with a scikit-learn-style API.

Imported by run_sample_size_experiment.py and notebook 02 as a 6th unsupervised
model. Anomaly score = per-row reconstruction error (MSE across features).
decision_function returns the NEGATED error so the project's
anomaly_scores(model, X, uses_fit_predict=False) == -decision_function(X)
recovers "higher = more anomalous", matching y=1 (fraud). anomaly_utils.py is
left untouched as the metric single source of truth.
"""
import numpy as np
import torch
import torch.nn as nn


class _AE(nn.Module):
    """Symmetric autoencoder: input -> encoder_dims (bottleneck) -> input.

    ReLU on hidden layers, linear output (inputs are standardized, can be
    negative, so no bounded output activation).
    """

    def __init__(self, input_dim, encoder_dims):
        super().__init__()
        dims = [input_dim, *encoder_dims]
        enc = []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        dec = []
        rev = list(reversed(dims))
        for i in range(len(rev) - 1):
            dec.append(nn.Linear(rev[i], rev[i + 1]))
            if i < len(rev) - 2:          # ReLU on hidden, linear on the output
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """Reconstruction-error anomaly detector trained on the unlabeled features."""

    def __init__(self, encoder_dims=(20, 14), epochs=30, batch_size=2048,
                 lr=1e-3, device=None, random_state=42, verbose=True):
        self.encoder_dims = tuple(encoder_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None

    def _to_tensor(self, X):
        return torch.from_numpy(np.asarray(X, dtype=np.float32))

    def fit(self, X):
        torch.manual_seed(self.random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        data = self._to_tensor(X).to(self.device)
        n, input_dim = data.shape
        self.model_ = _AE(input_dim, self.encoder_dims).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        g = torch.Generator().manual_seed(self.random_state)  # CPU generator, deterministic
        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, self.batch_size):
                batch = data[perm[i:i + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model_(batch), batch)
                loss.backward()
                opt.step()
        if self.device == "cuda":
            torch.cuda.synchronize()  # CUDA is async; finish before the caller stops its timer
        if self.verbose:
            print(f"[AutoencoderDetector] trained on {self.device}", flush=True)
        return self

    def decision_function(self, X):
        self.model_.eval()
        data = self._to_tensor(X).to(self.device)
        errs = torch.empty(data.shape[0], device=self.device)
        with torch.no_grad():
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                errs[i:i + batch.shape[0]] = ((self.model_(batch) - batch) ** 2).mean(dim=1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return (-errs).cpu().numpy()   # NEGATED: higher decision_function = more normal
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_torch_anomaly.py -v
```

Expected: 3 passed. If `test_reconstruction_error_orientation` is marginal, the synthetic separation is large (inliers ~N(0,1), outliers ~N(8,1)) — confirm `encoder_dims=(6, 3)` and `epochs=60` in the test, not the implementation defaults.

- [ ] **Step 5: Confirm the rest of the suite still passes**

Run:

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (existing `test_anomaly_utils.py` + the 3 new ones).

- [ ] **Step 6: Commit**

```bash
git add src/torch_anomaly.py tests/test_torch_anomaly.py
git commit -m "feat: add GPU autoencoder anomaly detector"
```

---

### Task 3: Register the detector in the sweep script + fix plot capacity

**Files:**
- Modify: `src/run_sample_size_experiment.py`

**Interfaces:**
- Consumes: `AutoencoderDetector` from Task 2.
- Produces: an `"Autoencoder (GPU)"` row in every per-size and cross-size output.

- [ ] **Step 1: Import the detector**

In `src/run_sample_size_experiment.py`, just below the existing
`from anomaly_utils import ...` line, add:

```python
from torch_anomaly import AutoencoderDetector
```

- [ ] **Step 2: Add the detector to `build_models()`**

In `build_models()`, add this as the final tuple of the returned list (after `Robust Covariance`):

```python
        ("Autoencoder (GPU)",
         AutoencoderDetector(encoder_dims=(20, 14), epochs=30, batch_size=2048,
                             lr=1e-3, random_state=RANDOM_STATE, verbose=False),
         False),
```

- [ ] **Step 3: Extend `COLORS` to 7 entries**

Replace the existing `COLORS` line:

```python
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#E91E63"]
```

with (adds cyan for the 7th model so `zip(models, COLORS)` in the cross-size plots and the bar charts cover every model):

```python
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#E91E63", "#00BCD4"]
```

- [ ] **Step 4: Grow the confusion-matrix grid from 2×3 to 3×3**

In `run_one_size()`, find the confusion-matrix block:

```python
    # ---- confusion matrices ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.ravel()
    for i, (name, (preds, y_used)) in enumerate(results.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix(y_used, preds),
                                      display_labels=["Normal", "Fraud"])
        disp.plot(ax=axes_flat[i], cmap="Blues", values_format=",d")
        axes_flat[i].set_title(name, fontsize=11, fontweight="bold")
```

Replace it with (3×3 = 9 axes for 7 models, hiding the 2 unused):

```python
    # ---- confusion matrices ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes_flat = axes.ravel()
    for i, (name, (preds, y_used)) in enumerate(results.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix(y_used, preds),
                                      display_labels=["Normal", "Fraud"])
        disp.plot(ax=axes_flat[i], cmap="Blues", values_format=",d")
        axes_flat[i].set_title(name, fontsize=11, fontweight="bold")
    for j in range(len(results), len(axes_flat)):
        axes_flat[j].axis("off")
```

- [ ] **Step 5: Smoke-run at 30k to verify integration + GPU use**

Temporarily limit the sweep: edit `SAMPLE_SIZES = [30_000, 80_000, 150_000, 230_000]` to `SAMPLE_SIZES = [30_000]`, then run:

```bash
cd src && python run_sample_size_experiment.py
```

Expected in the console: a `[AutoencoderDetector] trained on cuda` line (confirms GPU), and an `Autoencoder (GPU) ... flagged=... ` line with no error. Then check the output:

```bash
python -c "import pandas as pd; d=pd.read_csv('../results/30k/metrics.csv', index_col=0); print(d.loc['Autoencoder (GPU)', ['ROC-AUC','F1 Score']]); assert 0 <= d.loc['Autoencoder (GPU)','ROC-AUC'] <= 1"
```

Expected: prints the autoencoder's ROC-AUC and F1, assertion passes. Confirm `results/30k/confusion_matrices.png` exists and shows 7 populated panels.

- [ ] **Step 6: Restore the full sweep sizes**

Revert `SAMPLE_SIZES` back to `[30_000, 80_000, 150_000, 230_000]`.

- [ ] **Step 7: Commit**

```bash
git add src/run_sample_size_experiment.py
git commit -m "feat: register GPU autoencoder in sample-size sweep"
```

---

### Task 4: Mirror into notebook `02_train_and_evaluate.ipynb`

**Files:**
- Modify: `src/02_train_and_evaluate.ipynb`

**Interfaces:**
- Consumes: `AutoencoderDetector` from Task 2.
- Produces: the autoencoder appears in the notebook's stored metrics/plots, matching the script.

- [ ] **Step 1: Add the import**

In the notebook's first code cell (the imports cell), add after the existing helper imports:

```python
from torch_anomaly import AutoencoderDetector
```

- [ ] **Step 2: Add the detector to `model_specs`**

In the `model_specs` cell, add as the final tuple (after `Robust Covariance`):

```python
    ("Autoencoder (GPU)",
     AutoencoderDetector(encoder_dims=(20, 14), epochs=30, batch_size=2048,
                         lr=1e-3, random_state=RANDOM_STATE, verbose=False), False),
```

- [ ] **Step 3: Apply the same plot-capacity fixes as the script**

In the notebook: extend the `COLORS` list to the 7-entry version from Task 3 Step 3, and change the confusion-matrix `plt.subplots(2, 3, ...)` to the 3×3 version with the hide-unused-axes loop from Task 3 Step 4.

- [ ] **Step 4: Re-execute the notebook top-to-bottom**

Run:

```bash
cd src && jupyter nbconvert --to notebook --execute --inplace 02_train_and_evaluate.ipynb --ExecutePreprocessor.timeout=1200
```

Expected: exit 0, no errors, the `Autoencoder (GPU)` row present in the metrics table output and a `trained on cuda` line in the model-loop cell output.

- [ ] **Step 5: Commit**

```bash
git add src/02_train_and_evaluate.ipynb
git commit -m "feat: mirror GPU autoencoder into notebook 02"
```

---

### Task 5: Full regenerate + README update

**Files:**
- Modify: `README.md`
- Regenerates: `results/**`

**Interfaces:**
- Consumes: the integrated script from Task 3.
- Produces: refreshed result tables/plots and README narrative including the CPU/GPU timing caveat.

- [ ] **Step 1: Run the full sweep**

Run (≈20 min, dominated by One-Class SVM at 230k; the autoencoder adds only seconds per size on GPU):

```bash
cd src && python run_sample_size_experiment.py
```

Expected: completes with `Done. Results in ...`, and `trained on cuda` printed at each of the four sizes.

- [ ] **Step 2: Read the new numbers for the README**

Run:

```bash
python -c "import pandas as pd; d=pd.read_csv('results/comparison/all_metrics.csv'); print(d[d.Model=='Autoencoder (GPU)'][['Sample Size','ROC-AUC','F1 Score','Total Time (s)']].to_string(index=False))"
```

Expected: four rows (30k/80k/150k/230k) of the autoencoder's ROC-AUC, F1, and runtime — the values you will paste into the README tables.

- [ ] **Step 3: Add `torch` to the README dependencies**

In `README.md`, in the `## Dependencies` list and the `pip install ...` line under `## How to Run`, add `torch`. Note next to it: *"PyTorch with CUDA — install from https://pytorch.org for your CUDA version (used by the GPU autoencoder; falls back to CPU automatically)."*

- [ ] **Step 4: Add the Autoencoder row to the README results tables**

Add an `Autoencoder (GPU)` row to the 80k results table, the ROC-AUC-by-size table, the F1-by-size table, and the runtime-by-size table, using the values from Step 2. Add the model to the algorithms table with: *"Autoencoder (GPU) — neural net trained to reconstruct the data; reconstruction error = anomaly score. The only model trained on the GPU (PyTorch)."*

- [ ] **Step 5: Add the CPU/GPU caveat paragraph**

Under the results/efficiency discussion, add (verbatim intent from the spec):

> The Autoencoder is the only model trained on the GPU (PyTorch on an RTX 3050 Ti); the other six are CPU-only scikit-learn. Its timing column therefore is **not apples-to-apples** — it includes GPU compute plus host↔device transfer on different hardware. **ROC-AUC, being threshold- and hardware-independent, is the fair cross-model comparison.** GPU training is also inherently CPU+GPU cooperative (the CPU loads and batches data; the GPU does the forward/backward math), so "GPU-only" training does not exist.

- [ ] **Step 6: Commit**

```bash
git add README.md results
git commit -m "docs: add GPU autoencoder results and CPU/GPU caveat"
```

---

## Self-Review

**Spec coverage:**
- GPU autoencoder as 6th unsupervised model → Task 2 (class) + Task 3 (register). ✓
- Both pipelines → Task 3 (script) + Task 4 (notebook). ✓
- Score convention / `anomaly_utils.py` untouched → Task 2 `decision_function` returns `-MSE`; orientation test verifies via `anomaly_scores`. ✓
- Device + CPU fallback → Task 2 (`self.device`, `test_cpu_device_runs_without_cuda`). ✓
- Accurate GPU timing → Task 2 (`torch.cuda.synchronize()` in `fit`/`decision_function`). ✓
- Plot capacity for 7 models → Task 3 Steps 3–4, mirrored in Task 4 Step 3. ✓
- Dependency install → Task 1; README deps → Task 5 Step 3. ✓
- Tests (orientation, shape, fallback) → Task 2 Step 1. ✓
- Regenerate results + CPU/GPU caveat → Task 5. ✓

**Placeholder scan:** no TBD/TODO; all code and commands are concrete. ✓

**Type consistency:** `AutoencoderDetector(encoder_dims, epochs, batch_size, lr, device, random_state, verbose)`, `.fit(X)->self`, `.decision_function(X)->np.ndarray (n,)`, `.device:str`, registered tuple `(name, detector, False)` — identical across Tasks 2, 3, 4. ✓
