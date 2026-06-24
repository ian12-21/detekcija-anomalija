# Design Spec: IEEE Seminar Paper (LaTeX, Croatian)

**Date:** 2026-06-24
**Status:** Approved (design); pending spec review
**Author of work:** placeholder (to be filled by user)

## Goal

Produce a new seminar paper in **LaTeX** that strictly follows the IEEE
conference template shipped in
`Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/`
(`ieeeconf.cls`, `main.tex`). The paper is written **in Croatian**, mirrors the
exact section structure prescribed by `main.tex`, and is grounded in the actual
project implementation (`src/`) and the current results in `results/`. Detailed
but concise.

## Non-goals

- Do **not** modify `main.tex` (it stays as the reference template).
- Do **not** re-run experiments or change `results/`. The paper reports the
  numbers already produced.
- Do **not** alter IEEE margins / class options. Croatian-encoding packages are
  additive only.

## Title

> **ML5 — Strojno učenje za detekciju anomalija / Machine Learning for Anomaly
> Detection**

Author block: **placeholder** (`\author{Ime Prezime ...}` style, user fills in).

## Build & file layout

- New file: `Preparation_of_Papers_for_IEEE_Sponsored_Conferences_and_Symposia/seminar.tex`
  — placed alongside `ieeeconf.cls` so it compiles directly. `main.tex` untouched.
- Document class identical to template: `\documentclass[letterpaper, 10pt, conference]{ieeeconf}`
  with `\IEEEoverridecommandlockouts` and `\overrideIEEEmargins`.
- **Additive preamble** (does not touch margins/format), enabling Croatian:
  - `\usepackage[utf8]{inputenc}`
  - `\usepackage[T1]{fontenc}`
  - `\usepackage[croatian]{babel}`
  - `\usepackage{lmodern}` (clean T1 glyphs, avoids font-substitution warnings)
  - `\usepackage{graphicx}` (figures)
  - `\usepackage{amsmath,amssymb}` (MAD formula, R²)
  - `\usepackage{tikz}` (data-flow diagram)
- Figures referenced from `results/` via relative path `../results/...`.
- Compilation target: pdfLaTeX (template default). The paper must compile to PDF
  with no missing-reference/citation errors.

## Language conventions

- Prose in **Croatian** (proper diacritics: š, č, ć, ž, đ).
- Algorithm names kept in their original form (One-Class SVM, Isolation Forest,
  Local Outlier Factor, Robust Covariance / Elliptic Envelope, Autoencoder,
  Logistic Regression), with Croatian description — standard in Croatian ML
  writing.
- Headline metric framed as **ROC-AUC** (ranking quality), per project
  conventions (CLAUDE.md): the label-free MAD cutoff deliberately over-flags vs
  the 0.17% base rate, so thresholded F1/precision stay low by design.

## Section-by-section content plan

Section order is fixed by `main.tex`:
Abstract → I. Introduction → II. Methodology → III. Case Study (A. Datasets,
B. Evaluation) → IV. Results and Discussion → V. Conclusion → References.

### Abstract (≤150 words, Croatian)
Motivation (card fraud, extreme 0.17% imbalance) → Objective (validate Agyemang
2024 on large real imbalanced data; compare 6 unsupervised detectors + a
supervised LR baseline) → Methods (label-free per-model MAD cutoff, ROC-AUC as
primary metric, sample-size sweep 30k–230k) → Results (Isolation Forest /
One-Class SVM ROC-AUC 0.94–0.98 and stable; the ranking↔thresholding gap; IF best
quality/speed trade-off).

### I. Introduction
- Card-fraud detection problem; extreme class imbalance (~1 fraud per 577),
  why accuracy is misleading.
- Why **unsupervised** detection (labels scarce/expensive in practice).
- Agyemang (2024) is a **toy simulation study** (220 synthetic 2-D points);
  its own Conclusion/Limitations calls for validation on real-world data — that
  gap is this paper's motivation.
- Goal + contribution: identical algorithm lineup/hyperparameters applied to a
  large, imbalanced, high-dimensional real dataset, with a label-free thresholding
  scheme and a scalability sweep.
- References to relevant works (Agyemang, the canonical algorithm papers).

### II. Methodology
- **Algorithm lineup (Table I)** — 6 unsupervised detectors + LR baseline, with
  key hyperparameters taken verbatim from `build_models()` /
  `build_supervised_models()`:
  - One-Class SVM: `kernel=rbf, gamma=0.1, nu=0.05`
  - One-Class SVM (SGD): `nu=0.05, max_iter=1000, learning_rate=optimal`
  - Isolation Forest: `n_estimators=100, contamination=auto, max_samples=auto`
  - Local Outlier Factor: `n_neighbors=20, contamination=auto, novelty=False` (fit_predict)
  - Robust Covariance (EllipticEnvelope): `contamination=0.1`
  - Autoencoder (GPU): `encoder_dims=(20,14), epochs=30, batch_size=2048, lr=1e-3`
  - Logistic Regression (supervised baseline): `max_iter=1000, class_weight=balanced`
- **Unified anomaly score** — "higher = more anomalous" by negating
  `decision_function` (and `negative_outlier_factor_` for LOF's fit_predict),
  aligned with y=1 (fraud). This sign convention is load-bearing for ROC-AUC.
- **Label-free MAD cutoff** — robust modified z-score, with equation:
  z = (s − median(s)) / (1.4826·MAD), flag if z > k, k = MAD_K = 3.0.
  The flagged fraction is the data-driven contamination; the true fraud rate
  never enters detection.
- **Metrics** — ROC-AUC (continuous scores, threshold-independent, primary) plus
  Accuracy, Precision, Recall, F1, R² (binary preds); "implied contamination" =
  fraction flagged.
- **GPU Autoencoder** — symmetric 30→20→14→20→30, ReLU hidden + linear output,
  Adam, MSE reconstruction error as score (PyTorch). Note the **CPU/GPU caveat**:
  only the AE runs on GPU, so its runtime is not apples-to-apples; ROC-AUC is the
  fair cross-model comparison.
- **TikZ data-flow diagram (Fig. 1)** — pipeline:
  raw data → preprocessing (drop `Time`, StandardScaler on `Amount`) → random
  subsample (keep ~0.17% imbalance) → model fit → anomaly score → MAD threshold →
  metrics. Show the LR baseline branch (70/30 stratified split) as a side path.

### III. Case Study
- **A. Datasets** — Kaggle Credit Card Fraud: 284,807 transactions, 492 fraud
  (0.17%), V1–V28 (PCA, anonymized), `Amount`, `Time`, `Class` (eval only).
  Small dataset-composition table (Table II). Preprocessing (drop `Time`, scale
  `Amount`, keep duplicates). Why a common subsample (OCSVM/LOF/EllipticEnvelope
  ~O(n²)) and why a 4-size sweep (30k/80k/150k/230k) preserving natural imbalance.
  Two regimes: unsupervised models scored on the full subsample (no split); LR on
  a 70/30 stratified hold-out.
- **B. Evaluation** — the 6 metrics; why ROC-AUC is primary (threshold-
  independent); implied contamination as the over-flagging diagnostic;
  `RANDOM_STATE = 42` for reproducibility.

### IV. Results and Discussion
- **Main 80k metrics table (Table III)** — all 7 models, columns
  Accuracy/Precision/Recall/F1/ROC-AUC/R²/Implied Contam. Numbers from
  `results/80k/summary.txt`:
  - One-Class SVM: 99.29 / 10.14 / 44.09 / 16.49 / **0.9635** / −3.47 / 0.69%
  - One-Class SVM (SGD): 99.35 / 0 / 0 / 0 / 0.0504 / −3.11 / 0.49%
  - Isolation Forest: 95.60 / 2.97 / 84.25 / 5.73 / 0.9612 / −26.75 / 4.51%
  - Local Outlier Factor: 92.77 / 0.44 / 19.69 / 0.86 / 0.5481 / −44.63 / 7.14%
  - Robust Covariance: 61.08 / 0.38 / 92.91 / 0.75 / 0.8882 / −244.53 / 39.05%
  - Autoencoder (GPU): 92.55 / 1.82 / 86.61 / 3.56 / 0.9523 / −46.01 / 7.57%
  - Logistic Regression: 97.30 / 5.15 / 92.11 / 9.76 / 0.9606 / −16.05 / 2.83%
- **Figures (curated set, 3 from `results/`):**
  - Fig. 2 — ROC-AUC vs sample size: `results/comparison/rocauc_vs_size.png`
  - Fig. 3 — Total runtime vs sample size (log scale): `results/comparison/runtime_vs_size.png`
  - Fig. 4 — 80k performance overview: `results/80k/metrics_bar.png`
- **Cross-size tables (from `results/comparison/summary.md`):**
  - Table IV — ROC-AUC across 30k/80k/150k/230k
  - Table V — F1 across sizes
  - Table VI — Total runtime (s) across sizes
- **Discussion:**
  - Isolation Forest & One-Class SVM — strong, stable rankers (ROC-AUC 0.94–0.98).
  - Robust Covariance — good ranker (~0.89–0.94) but MAD over-flags ~40% → F1≈0.
  - LOF — collapses to ~random (0.77 → ~0.51) as dimensionality/scale bite.
  - SGD One-Class SVM — inverted (ROC-AUC < 0.5), an optimization pathology.
  - Autoencoder (GPU) — stable (0.93–0.97), high recall, GPU-fast (2.6–6.8 s).
  - **The ranking↔thresholding gap** — strong models rank fraud well yet
    thresholded F1 stays low because the label-free MAD cutoff over-flags vs the
    0.17% base rate (R² correspondingly very negative; not a discriminator).
  - **Scalability** — OCSVM runtime ≈ O(n²) (16.5→894 s for 30k→230k), Isolation
    Forest effectively flat (0.3→1.7 s): more data buys OCSVM time, not quality.
- **Comparison to Agyemang (Table VII)** — F1 for outliers, paper vs this project:
  OCSVM 66.67→16.49, OCSVM-SGD 9.52→0.00, IF 64.41→5.73, LOF 9.52→0.86,
  Robust Cov 66.67→0.75. Discuss: F1 drop is the cost of label-free thresholding
  on real 0.17% data, not a ranking failure; LOF/SGD fail in both studies.

### V. Conclusion
Restate objective (validate Agyemang on real, imbalanced, high-dimensional data),
main findings (label-free detection; ROC-AUC shows IF/OCSVM excellent & stable;
IF best quality/speed trade-off), limitations (subsample variance; MAD over-
flagging; R² not discriminative; SGD-OCSVM degeneracy), and future work
(threshold calibration without labels is the open problem).

### References (~9, IEEE `thebibliography`)
1. Agyemang (2024), *Scientific African* 26, e02386 (DOI).
2. Credit Card Fraud Detection dataset — Kaggle.
3. Pedregosa et al. (2011) — scikit-learn, *JMLR*.
4. Schölkopf et al. (2001) — One-Class SVM.
5. Liu, Ting, Zhou (2008) — Isolation Forest.
6. Breunig et al. (2000) — LOF.
7. Rousseeuw & Van Driessen (1999) — Minimum Covariance Determinant.
8. Iglewicz & Hoaglin (1993) — MAD / modified z-score.
9. Paszke et al. (2019) — PyTorch.

## Figures & tables summary

- **Figures (4):** Fig. 1 TikZ pipeline (authored); Fig. 2 ROC-AUC vs size;
  Fig. 3 runtime vs size (log); Fig. 4 80k metrics overview.
- **Tables (7):** I lineup+hyperparams; II dataset composition; III 80k metrics;
  IV ROC-AUC vs size; V F1 vs size; VI runtime vs size; VII Agyemang comparison.
  (VI may be dropped if space-constrained, since Fig. 3 covers runtime — finalize
  in the plan.)

## Acceptance criteria

1. `seminar.tex` exists alongside `ieeeconf.cls`; `main.tex` unchanged.
2. Compiles with pdfLaTeX to a PDF with **no undefined references/citations** and
   no missing-figure errors.
3. Section structure matches `main.tex` exactly (Abstract, Introduction,
   Methodology, Case Study with Datasets + Evaluation subsections, Results and
   Discussion, Conclusion, References).
4. Croatian throughout, with correct diacritics rendering.
5. Every reported number traces to `results/` (80k `summary.txt` and comparison
   `summary.md`); no invented figures.
6. The 4 figures and the agreed tables are present and captioned (captions below
   figures, table heads above tables, IEEE style).
7. ROC-AUC framed as the primary metric; the ranking↔thresholding gap and the
   CPU/GPU caveat are stated.
8. ~9 references, all cited in the text.

## Risks / open points

- **babel-croatian × ieeeconf**: should be compatible; if a clash appears, fall
  back to `\usepackage[T1]{fontenc}` + manual hyphenation only (no babel). Verify
  at compile.
- **Figure paths**: PNGs live under `../results/`; confirm pdfLaTeX accepts PNG
  (it does). Keep `\includegraphics[width=\columnwidth]{...}`.
- **Space**: 7 tables + 4 figures is heavy for a 2-column conference length; the
  plan may merge/trim (esp. Table VI) to keep it concise.
