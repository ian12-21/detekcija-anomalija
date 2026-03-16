"""
Step 1: Data Exploration
========================
This script loads the credit card fraud dataset and helps us understand:
- What the data looks like (shape, types, missing values)
- How imbalanced it is (normal vs anomaly ratio)
- Basic statistics and distributions

Run this first before touching any ML algorithms!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Configuration ──────────────────────────────────────────────
DATA_PATH = "../data/creditcard.csv"  # adjust if your file is named differently
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# ── 2. Basic Info ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA TYPES & MISSING VALUES")
print("=" * 60)

print(f"\nData types:\n{df.dtypes.value_counts()}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# ── 3. Class Distribution (THE MOST IMPORTANT PART) ──────────
print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

print(f"\nNormal transactions (Class=0): {class_counts[0]} ({class_pct[0]:.3f}%)")
print(f"Fraudulent transactions (Class=1): {class_counts[1]} ({class_pct[1]:.3f}%)")
print(f"Imbalance ratio: 1 fraud per {class_counts[0] // class_counts[1]} normal transactions")

# Plot class distribution
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#2196F3", "#F44336"]
bars = ax.bar(["Normal (0)", "Fraud (1)"], class_counts.values, color=colors)
ax.set_ylabel("Count")
ax.set_title("Class Distribution")
# Add count labels on bars
for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_class_distribution.png", dpi=150)
plt.close()
print("Saved: 01_class_distribution.png")

# ── 4. Statistical Summary ────────────────────────────────────
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

# Focus on non-PCA features (Time and Amount) since V1-V28 are already transformed
print("\n'Time' column:")
print(f"  Range: {df['Time'].min():.0f} - {df['Time'].max():.0f} seconds")
print(f"  This spans ~{df['Time'].max() / 3600:.1f} hours of transactions")

print(f"\n'Amount' column:")
print(f"  Mean: ${df['Amount'].mean():.2f}")
print(f"  Median: ${df['Amount'].median():.2f}")
print(f"  Max: ${df['Amount'].max():.2f}")
print(f"  Std: ${df['Amount'].std():.2f}")

# Compare Amount between classes
print(f"\nAmount by class:")
for cls in [0, 1]:
    subset = df[df["Class"] == cls]["Amount"]
    label = "Normal" if cls == 0 else "Fraud"
    print(f"  {label}: mean=${subset.mean():.2f}, median=${subset.median():.2f}, max=${subset.max():.2f}")

# ── 5. Feature Distributions ─────────────────────────────────
print("\n" + "=" * 60)
print("PLOTTING FEATURE DISTRIBUTIONS")
print("=" * 60)

# Plot Amount distribution by class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Amount distribution
for cls, color, label in [(0, "#2196F3", "Normal"), (1, "#F44336", "Fraud")]:
    subset = df[df["Class"] == cls]["Amount"]
    axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Density")
axes[0].set_title("Transaction Amount Distribution")
axes[0].legend()
axes[0].set_xlim(0, 500)  # zoom in, most transactions are small

# Time distribution
for cls, color, label in [(0, "#2196F3", "Normal"), (1, "#F44336", "Fraud")]:
    subset = df[df["Class"] == cls]["Time"]
    axes[1].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Density")
axes[1].set_title("Transaction Time Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02_amount_time_distributions.png", dpi=150)
plt.close()
print("Saved: 02_amount_time_distributions.png")

# ── 6. Correlation Heatmap (V features) ──────────────────────
# Check if V features are correlated (they shouldn't be much since they're PCA components)
v_cols = [f"V{i}" for i in range(1, 29)]
corr_matrix = df[v_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation Between PCA Features (V1-V28)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: 03_correlation_heatmap.png")

# ── 7. Key V Features — Normal vs Fraud ──────────────────────
# Plot a few V features that tend to differ most between classes
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
key_features = ["V1", "V3", "V4", "V7", "V10", "V12", "V14", "V17", "Amount"]

for ax, feat in zip(axes.ravel(), key_features):
    for cls, color, label in [(0, "#2196F3", "Normal"), (1, "#F44336", "Fraud")]:
        subset = df[df["Class"] == cls][feat]
        ax.hist(subset, bins=50, alpha=0.5, color=color, label=label, density=True)
    ax.set_title(feat)
    ax.legend(fontsize=8)

plt.suptitle("Feature Distributions: Normal vs Fraud", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/04_key_features_comparison.png", dpi=150)
plt.close()
print("Saved: 04_key_features_comparison.png")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPLORATION SUMMARY")
print("=" * 60)
print(f"""
Key takeaways:
1. Dataset has {df.shape[0]:,} transactions with {df.shape[1]} features
2. Extremely imbalanced: only {class_pct[1]:.3f}% are fraud
3. V1-V28 are PCA-transformed features (already scaled)
4. 'Time' and 'Amount' are the only original features
5. Amount needs scaling before feeding to ML algorithms
6. V features are mostly uncorrelated (expected from PCA)

Next step: Preprocessing & algorithm implementation (02_preprocessing.py)
""")
