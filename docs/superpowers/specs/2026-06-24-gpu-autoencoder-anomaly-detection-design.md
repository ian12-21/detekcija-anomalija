# Design: GPU-accelerated autoencoder anomaly detector (PyTorch)

**Date:** 2026-06-24
**Branch:** `feature/gpu-autoencoder` (proposed)
**Status:** Approved

## Problem

The project's stated next goal is to **learn GPU/CUDA computing** by training a
model on the user's RTX 3050 Ti (4 GB, Ampere SM 8.6, Windows 11, driver 610.62)
instead of the CPU. The existing comparison cannot be moved to the GPU as-is:

1. **scikit-learn is CPU-only by design.** All six current models (One-Class SVM,
   SGD-OCSVM, Isolation Forest, LOF, Robust Covariance, Logistic Regression) have
   no GPU path — no device flag exists.
2. **The only "drop-in GPU sklearn" (RAPIDS cuML) is not viable here.** It does
   not run natively on Windows (needs WSL2 + Linux) and does not implement most of
   these estimators (no `OneClassSVM`, no `EllipticEnvelope`).
3. **Raw CUDA (Numba/CuPy kernels) is deferred.** The user is time-constrained and
   chose the lower-risk PyTorch route now, leaving the kernel-level work for later.

The only path that genuinely uses the GPU today, on native Windows, with low
implementation risk, is a **PyTorch deep model**.

## Goals

- Add a **GPU-accelerated autoencoder anomaly detector** as a **6th unsupervised
  model** in the comparison, scored by **per-row reconstruction error**.
- Integrate it into **both** mirrored pipelines (`run_sample_size_experiment.py`
  and `02_train_and_evaluate.ipynb`) so it flows through the existing
  `anomaly_scores → mad_threshold → evaluate` path with no special-casing and
  lands in every results table/plot on the same **ROC-AUC** footing.
- Run training and scoring on CUDA when available, with a **CPU fallback** so the
  sweep still runs for a grader without a GPU.
- Refresh `results/` and the README, including an explicit **CPU/GPU timing
  caveat**.

## Non-goals

- **No Deep SVDD** — the autoencoder is chosen for fewest failure modes (Deep SVDD
  can collapse its hypersphere; an AE basically cannot).
- **No hyperparameter search, no early stopping, no learning-rate schedule** —
  fixed, documented hyperparameters.
- **No raw CUDA kernels** (Numba/CuPy) — explicitly the "later" pass.
- **No attempt to move the existing six models to the GPU** — not feasible today.
- **No new metric** — reuse the six already in `anomaly_utils.evaluate`.

## Key decisions (locked)

| Decision | Choice |
|---|---|
| Deep model | Symmetric autoencoder, reconstruction-error score |
| Score convention | `decision_function` returns **negative** per-row MSE, so the existing `anomaly_scores` (which negates it) yields "higher = more anomalous" |
| Where the code lives | New `src/torch_anomaly.py`; **`anomaly_utils.py` stays untouched** (metric SoT) |
| Model group | Unsupervised — fit and score on the full unlabeled subsample, like the other five |
| Device | `cuda` if available else `cpu`; printed; CPU fallback always works |
| Reproducibility | `torch.manual_seed(RANDOM_STATE)`; document that exact CUDA determinism is not guaranteed |
| Scope | Script + notebook 02, then regenerate `results/` + README |

## Architecture

### New module: `src/torch_anomaly.py`

A single class with a **minimal scikit-learn-style API** so it drops into the
existing pipeline unchanged. It does *not* subclass sklearn — it just exposes the
two methods `anomaly_scores(...)` relies on for a non-`fit_predict` model:

```python
class AutoencoderDetector:
    """GPU autoencoder anomaly detector with a scikit-learn-style API.

    Anomaly score = per-row reconstruction error (MSE across features).
    decision_function returns the NEGATED error so that the project's existing
    anomaly_scores(model, X, uses_fit_predict=False) == -decision_function(X)
    recovers "higher = more anomalous", matching y=1 (fraud).
    """

    def __init__(self, encoder_dims=(20, 14), epochs=30, batch_size=2048,
                 lr=1e-3, device=None, random_state=42):
        ...

    def fit(self, X):            # train loop; returns self
        ...

    def decision_function(self, X):   # returns -reconstruction_error, shape (n,)
        ...
```

- **Network:** `input_dim → 20 → 14 (bottleneck) → 20 → input_dim`. ReLU on
  hidden layers, **linear output**, **MSE loss**, **Adam(lr=1e-3)**.
  `input_dim = 29` (V1–V28 + scaled `Amount`).
- **Why linear output, not sigmoid:** inputs are already ~standardized (PCA
  features + StandardScaled `Amount`) and can be negative, so a bounded output
  activation would be wrong.
- **Training:** mini-batched (`batch_size=2048`), `epochs=30`, shuffled. Inputs
  cast to `float32`. The model and batches are tiny — **4 GB is never a
  constraint** (this is exactly why PyTorch is the low-risk choice).
- **Scoring:** `eval()` + `torch.no_grad()`, batched forward pass, per-row MSE
  across the feature dimension. Returned negated.
- **Device handling:** `device = device or ("cuda" if torch.cuda.is_available()
  else "cpu")`. The chosen device is printed once. The same code path runs on CPU.
- **Reproducibility:** seed torch (and CUDA) with `random_state`. A short note
  that GPU floating-point reductions are not bit-deterministic; ROC-AUC is stable
  across runs regardless.

### The CPU/GPU reality (documented, not worked around)

Even this model is **CPU + GPU cooperative**, never GPU-only: the CPU loads/scales
the data and copies each batch to the GPU (host→device); the GPU does the
forward/backward math; the CPU drives the loop. This is the intended, normal shape
of GPU training and is called out in the README so the comparison is honest.

### Accurate GPU timing

CUDA kernels are asynchronous, so naive `time.time()` around a `.fit()` would
under-measure. The detector (or the calling harness) calls
`torch.cuda.synchronize()` immediately before stopping each timer when on CUDA, so
the reported **Fit Time** and **Predict Time** reflect real GPU work plus
host↔device transfer.

## Integration into the two mirrored pipelines

### `src/run_sample_size_experiment.py`

- Add the detector as the **6th entry** of `build_models()`:
  `("Autoencoder (GPU)", AutoencoderDetector(...), False)`. It is unsupervised, so
  it uses the existing fit → `anomaly_scores` → `mad_threshold` → `evaluate` flow
  with **zero special-casing**; `uses_fit_predict=False`.
- **Plot-capacity fixes** (the comparison now has 7 models — 6 unsupervised + the
  LR baseline — not 6):
  - `COLORS` currently has 6 entries; extend to **≥7** so `zip(models, COLORS)` in
    the cross-size plots and the `color=COLORS` bar charts cover every model
    (`zip` would otherwise silently drop the 7th model's trend line).
  - `confusion_matrices.png` is a hardcoded **2×3 = 6** subplot grid indexed per
    model; grow it to **3×3** (or 2×4) and hide unused axes so a 7th model does
    not overflow `axes_flat`.
  - `metrics_bar.png` (2×3 over the six *metrics*) needs no grid change — bars
    just grow — but relies on the extended `COLORS`.

### `src/02_train_and_evaluate.ipynb`

Mirror the script: import `AutoencoderDetector` from `torch_anomaly`, add it to
`model_specs`, apply the same `COLORS`/confusion-grid fixes, and re-execute to
refresh stored outputs.

## Outputs

- **No schema changes** to `metrics.csv` / `summary.txt` — the autoencoder is just
  a new model **row**, carrying the same six metrics + implied contamination.
- `metrics_bar.png`, `confusion_matrices.png`, and every `*_vs_size.png` gain the
  autoencoder automatically once the color/grid capacity is fixed.

## Dependencies

- **PyTorch with CUDA**, installed from the appropriate index, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  (exact `cuXXX` pinned during implementation; driver 610.62 is recent and
  backward-compatible with current wheels). Add `torch` to the README deps list.
- No other new dependencies. `numpy`/`pandas` already present.

## Verification

- **Unit tests** in `tests/test_torch_anomaly.py`, forced to `device="cpu"` so
  they run anywhere (CI, a grader without CUDA):
  - **Orientation:** on synthetic inliers + a cluster of outliers, the AE's
    reconstruction-error score gives `roc_auc_score(y, score) > 0.8` (guards
    against a sign flip in `decision_function`).
  - **Shape/finiteness:** `decision_function(X)` returns finite floats of shape
    `(n,)`.
  - **CPU fallback:** constructing with `device="cpu"` trains and scores without a
    GPU present.
- **Smoke run** the script at 30k; confirm the autoencoder's ROC-AUC ∈ [0, 1] and
  implied contamination is sane, and that CUDA is actually used
  (`device: cuda` printed, GPU active). Then full regenerate.

## Risks / implications

- **Adds a `torch` dependency and a GPU code path.** Mitigated by the CPU
  fallback: the sweep still runs end-to-end without CUDA.
- **CPU↔GPU timing is not apples-to-apples.** The AE's wall-clock includes GPU
  compute + host↔device transfer on different hardware than the CPU models. The
  README states this; **ROC-AUC (hardware-independent) is the fair comparison.**
- **Every plot's model count grows from 6 to 7.** The color/confusion-grid fixes
  above are required or the 7th series is dropped / overflows.
- **GPU results are not bit-reproducible.** Seeded for stability; ROC-AUC is
  robust across runs.
- **VRAM is not a real risk** at this model size, but batch size stays modest
  (2048) to keep headroom on the 4 GB card.

## Build sequence

1. `src/torch_anomaly.py` (`AutoencoderDetector`) + `tests/test_torch_anomaly.py`
   (TDD on the CPU-forced detector: orientation, shape, fallback).
2. Install CUDA PyTorch; confirm `torch.cuda.is_available()` on the RTX 3050 Ti.
3. Integrate into `run_sample_size_experiment.py` (add to `build_models()`, extend
   `COLORS`, grow the confusion-matrix grid).
4. Smoke-run at 30k on GPU; verify device + sane metrics; then full regenerate.
5. Mirror into `02_train_and_evaluate.ipynb` and re-execute.
6. Refresh README: add the autoencoder row to the results tables and a short
   GPU/autoencoder paragraph **including the CPU/GPU timing caveat**.
