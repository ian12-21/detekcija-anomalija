# Anomaly Detection Using Unsupervised Machine Learning

Seminar project for **Strojno učenje (Machine Learning) — ML5** — Technical Faculty, University of Rijeka.

Detecting fraudulent credit card transactions using unsupervised machine learning algorithms.

Based on: Agyemang, E.F. (2024) *"Anomaly detection using unsupervised machine learning algorithms: A simulation study"*, Scientific African 26, e02386.

---

## Project Structure

```
detekcija-anomalija/
├── data/
│   └── creditcard.csv              # Credit Card Fraud Detection dataset
├── src/
│   ├── 01_explore_data.ipynb       # Data exploration & analysis
│   └── 02_train_and_evaluate.ipynb # Preprocessing, training & evaluation
└── README.md
```

---

## Dataset

**Source:** Credit Card Fraud Detection dataset (Kaggle)

| Property              | Value                  |
|----------------------|------------------------|
| Total transactions   | 284,807                |
| Normal (Class=0)     | 284,315 (99.83%)       |
| Fraud (Class=1)      | 492 (0.17%)            |
| Features             | 31 (Time, V1-V28, Amount, Class) |
| Imbalance ratio      | 1 fraud per 577 normal |
| Missing values       | 0                      |

**Features:**
- `V1-V28` — PCA-transformed features (anonymized, already scaled, uncorrelated)
- `Amount` — transaction amount (requires scaling)
- `Time` — seconds since first transaction (dropped during preprocessing)
- `Class` — label (0 = normal, 1 = fraud), used only for evaluation

---

## Notebook 01: Data Exploration

Key findings:

- **Extreme class imbalance** — only 0.17% of transactions are fraudulent, making accuracy a misleading metric
- **PCA features are uncorrelated** — confirmed by the correlation heatmap (diagonal pattern)
- **Amount differs between classes** — normal median is $22.00, fraud median is $9.25 (fraudulent transactions tend to be smaller)
- **Features V14, V12, V17** show the clearest distribution differences between normal and fraud
- **Dataset spans ~48 hours** of transactions

**Plots generated:**
1. Class distribution bar chart
2. Amount and Time distributions (normal vs fraud)
3. Correlation heatmap of V1-V28
4. Feature distribution comparison (9 key features)

---

## Notebook 02: Preprocessing, Training & Evaluation

### Preprocessing

1. **Dropped `Time`** — not useful for anomaly pattern detection
2. **Scaled `Amount`** using StandardScaler — to match the V1-V28 range (mean=0, std=1)
3. **Kept duplicates** (1,081 rows) — duplicate transactions can be legitimate

### Fully Unsupervised Setup

Following the reference paper, detection is **fully unsupervised** — no labels
are used for training and there is **no train/test split**. Every model is fit
and scored on the same unlabeled feature matrix; `Class` is used **only** for
the final evaluation.

### Common Subsample (fair comparison)

One-Class SVM, LOF and Elliptic Envelope scale poorly to ~285K rows. To compare
all five algorithms **fairly**, every model is fit and scored on the *same*
random subsample of **80,000 rows**, keeping the natural ~0.17% imbalance
(127 frauds). `contamination` / `nu` is set to the observed fraud fraction of
the subsample (≈0.159%), so every model targets the same anomaly budget.

### Algorithms

| # | Algorithm | Key Idea |
|---|-----------|----------|
| 1 | **One-Class SVM** | Kernel trick to find a boundary around the dense region |
| 2 | **One-Class SVM (SGD)** | Same idea via Stochastic Gradient Descent |
| 3 | **Isolation Forest** | Random trees — anomalies need fewer splits to isolate |
| 4 | **Local Outlier Factor** | Compares local density of a point to its neighbors (`novelty=False`, `fit_predict`) |
| 5 | **Robust Covariance** | Fits a Gaussian ellipse — points outside = anomalies |

All five trained and scored on the identical 80,000-row subsample.

### Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| One-Class SVM | 96.34% | 2.22% | 51.18% | 4.25% |
| One-Class SVM (SGD) | 99.84% | 0.00% | 0.00% | 0.00% |
| **Isolation Forest** | **99.79%** | **33.86%** | **33.86%** | **33.86%** |
| Local Outlier Factor | 99.68% | 0.00% | 0.00% | 0.00% |
| Robust Covariance | 99.73% | 15.75% | 15.75% | 15.75% |

**Winner: Isolation Forest** with the best F1 score (33.86%).

### Analysis

- **Isolation Forest (best)** — best F1 by a clear margin and effectively instant. Random-partition isolation handles the high-dimensional PCA space well even without labels.
- **Robust Covariance (moderate)** — caught a fair share of fraud; the Gaussian-ellipse assumption partly holds after scaling.
- **One-Class SVM (high recall, near-zero precision)** — recall ~51% but it flagged 2,931 points out of 80,000 (only 127 are fraud), so precision collapses to ~2%. Far too many false alarms.
- **One-Class SVM (SGD) — failed** — predicted zero anomalies. This is an optimization/scaling sensitivity of the linear SGD solver, not a real ranking of the method.
- **Local Outlier Factor — collapsed** — at this contamination budget LOF's flagged points missed the fraud entirely (F1 0%). Sensitive to `n_neighbors` and the global contamination on this data.

Models 3–5 flag *exactly* 127 points — the `contamination` budget at work — so for them precision = recall = F1.

### Computational Efficiency

| Model | Fit Time (s) | Predict Time (s) | Total (s) |
|-------|-------------|------------------|-----------|
| One-Class SVM | 47.09 | 12.28 | 59.38 |
| One-Class SVM (SGD) | 0.09 | 0.01 | 0.10 |
| Isolation Forest | 0.32 | 0.21 | 0.53 |
| Local Outlier Factor | 4.64 | 0.00 | 4.64 |
| Robust Covariance | 7.11 | 0.02 | 7.13 |

One-Class SVM (RBF) is by far the slowest (~O(n²)). Isolation Forest is both the
most accurate **and** the fastest with a real `predict`. LOF fits and predicts in
one `fit_predict` step, so its predict time is reported as 0.

**Plots generated:**
1. Performance metrics bar chart (accuracy, precision, recall, F1)
2. Confusion matrices for all 5 algorithms
3. Computational efficiency comparison (horizontal bar chart)

### Limitations

- Metrics come from an 80,000-row subsample (~127 frauds), so precision/recall
  still carry variance — raise `SAMPLE_SIZE` in the notebook to stabilize them.
- `contamination` is a fixed threshold equal to the true fraud rate, not tuned
  per model.
- `SGDOneClassSVM` can collapse to a degenerate all-normal solution; reported
  with that caveat rather than as a method ranking.

---

## Key Takeaway

The reference paper found **Isolation Forest** performed best, and our
fully-unsupervised, equal-sample comparison on real-world data reproduces that
result — Isolation Forest wins on both F1 and runtime. Unsupervised detection on
this extremely imbalanced data is genuinely hard (best F1 ≈ 34%), which itself
is a key finding for the report.

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Start Jupyter
cd detekcija-anomalija/src
jupyter notebook
```

Run notebooks in order (from `src/`):
1. `01_explore_data.ipynb`
2. `02_train_and_evaluate.ipynb`

---

## Dependencies

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

---

## References

- Agyemang, E.F. (2024). Anomaly detection using unsupervised machine learning algorithms: A simulation study. *Scientific African*, 26, e02386. https://doi.org/10.1016/j.sciaf.2024.e02386
- Credit Card Fraud Detection Dataset — Kaggle
- scikit-learn documentation — https://scikit-learn.org
