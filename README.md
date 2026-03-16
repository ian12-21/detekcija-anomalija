# Anomaly Detection using Unsupervised ML

Seminar project for Evolucijsko računarstvo — detecting anomalies (fraud) 
in credit card transaction data using unsupervised machine learning algorithms.

Based on: Agyemang, E.F. (2024) "Anomaly detection using unsupervised machine 
learning algorithms: A simulation study", Scientific African.

## Project Structure

```
anomaly-detection/
├── data/                    # Put your CSV here
│   └── creditcard.csv
├── src/
│   ├── 01_explore_data.py   # Step 1: Understand the data
│   ├── 02_preprocessing.py  # Step 2: Clean & prepare (TODO)
│   ├── 03_train_models.py   # Step 3: Train 5 algorithms (TODO)
│   └── 04_evaluate.py       # Step 4: Compare results (TODO)
├── results/                 # Saved plots and metrics
└── README.md
```

## Algorithms (from the paper)

1. **One-Class SVM** — boundary-based, uses kernel trick
2. **One-Class SVM with SGD** — scalable variant using stochastic gradient descent
3. **Isolation Forest** — isolation-based, good for high-dimensional data
4. **Local Outlier Factor (LOF)** — density-based, compares local densities
5. **Robust Covariance (Elliptic Envelope)** — assumes Gaussian distribution

## How to Run

```bash
cd src/
python 01_explore_data.py
```

## Dependencies

```
pip install pandas numpy scikit-learn matplotlib seaborn
```
