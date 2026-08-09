# Enhanced 03_model_training.py – Cell-by-Cell Code

> **How to use**: Each section below maps to a numbered Databricks cell.
> Copy-paste the code block directly into the matching cell.
> Cells that should be **replaced** are marked **[REPLACE]**.
> New cells to **add** are marked **[NEW]**.

---

## Cell 1 — Setup & Installs  [REPLACE]

```python
# MAGIC %pip install catboost shap imbalanced-learn mlflow

import os, sys, json, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering on Databricks
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.catboost
import shap

# Add src to path
sys.path.append('/Workspace/Users/mohamed.c.elshenity@gmail.com/fraud/src/model')

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from train      import train_model
from evaluate   import evaluate_model
from threshold  import find_best_threshold
from shap_explainer import compute_shap_reasons
```

---

## Cell 2 — Spark Session  [REPLACE]

```python
spark = SparkSession.builder \
    .appName("FraudDetectionModelTraining") \
    .config("spark.sql.shuffle.partitions", "16") \
    .getOrCreate()

print(f"Spark version: {spark.version}")
```

---

## Cell 3 — Load Processed Data  [REPLACE]

```python
# Load from Delta table written by notebook 02
df = spark.table("fraud_features")
print(f"Processed data loaded: {df.count():,} records")
df.printSchema()
```

---

## Cell 4 — Inspect Class Imbalance  [NEW]

```python
label_dist = df.groupBy("is_fraud").count().toPandas()
total = label_dist["count"].sum()
label_dist["pct"] = (label_dist["count"] / total * 100).round(2)
print("Class distribution:")
print(label_dist.to_string(index=False))

fraud_count   = int(label_dist.loc[label_dist.is_fraud == 1, "count"])
legit_count   = int(label_dist.loc[label_dist.is_fraud == 0, "count"])
imbalance_ratio = round(legit_count / fraud_count, 1)
print(f"\nImbalance ratio (legit:fraud) = {imbalance_ratio}:1")
```

---

## Cell 5 — Time-Based Train / Val / Test Split  [REPLACE]

```python
# Sort by unix_time (temporal ordering – prevents data leakage)
df = df.orderBy("unix_time")

total_count = df.count()
train_end   = int(total_count * 0.70)
val_end     = int(total_count * 0.85)

window = Window.orderBy("unix_time")
df = df.withColumn("row_idx", F.row_number().over(window))

train_df = df.filter(F.col("row_idx") <= train_end)
val_df   = df.filter((F.col("row_idx") > train_end) & (F.col("row_idx") <= val_end))
test_df  = df.filter(F.col("row_idx") > val_end)

print(f"Train : {train_df.count():,}")
print(f"Val   : {val_df.count():,}")
print(f"Test  : {test_df.count():,}")
```

---

## Cell 6 — Define Feature Columns  [REPLACE]

```python
# All features that come out of the feature-engineering notebooks
FEATURE_COLS = [
    # ── Numeric / Static ──────────────────────────────────────────
    "amt",                    # transaction amount
    "city_pop",               # customer city population
    "hour",                   # hour of day (0-23)
    "day_of_week",            # day of week (1=Sunday)
    "month",                  # calendar month
    # ── Geospatial ────────────────────────────────────────────────
    "distance",               # Euclidean customer-merchant distance
    "haversine_distance",     # Great-circle distance (km)  ← was unused
    # ── Window / Velocity features ────────────────────────────────
    "txn_count_1h",           # # txns by card in last 1 hour
    "txn_count_24h",          # # txns by card in last 24 hours
    "avg_amt_24h",            # avg spend in last 24 hours
    "spend_24h",              # total spend in last 24 hours
    "unique_merchants_24h",   # distinct merchants in last 24 hours
    "time_since_last_txn",    # seconds since previous txn on this card
    # ── Lookup / Fraud-Rate features ──────────────────────────────
    "category_fraud_rate",    # historical fraud rate per category
    "category_txn_count",     # # txns per category (volume signal)
    "merchant_fraud_rate",    # historical fraud rate per merchant
    "merchant_txn_count",     # # txns per merchant
    "state_fraud_rate",       # historical fraud rate per state
    # ── Categorical (passed as-is to CatBoost) ────────────────────
    "category", "merchant", "state", "gender", "city", "zip", "job",
]

# Keep only columns that actually exist in this run's processed table
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
label_col    = "is_fraud"

print(f"Using {len(feature_cols)} features:")
for c in feature_cols:
    print(f"  {c}")
```

---

## Cell 7 — Convert to Pandas (with memory guard)  [REPLACE]

```python
# We stay within Databricks driver RAM by converting each split
# only once and immediately freeing the Spark DataFrames afterwards.

X_train_pd = train_df.select(feature_cols).toPandas()
y_train_pd = train_df.select(label_col).toPandas()[label_col].astype(int)
train_df.unpersist()

X_val_pd   = val_df.select(feature_cols).toPandas()
y_val_pd   = val_df.select(label_col).toPandas()[label_col].astype(int)
val_df.unpersist()

X_test_pd  = test_df.select(feature_cols).toPandas()
y_test_pd  = test_df.select(label_col).toPandas()[label_col].astype(int)
test_df.unpersist()

print(f"X_train: {X_train_pd.shape}  Fraud: {y_train_pd.sum():,}")
print(f"X_val  : {X_val_pd.shape}    Fraud: {y_val_pd.sum():,}")
print(f"X_test : {X_test_pd.shape}   Fraud: {y_test_pd.sum():,}")
```

---

## Cell 8 — Class Balancing via Hybrid SMOTENC + Undersample  [NEW]

> **Why this approach?**
> Full SMOTENC (1:1 ratio) blows up RAM. Instead we:
> 1. SMOTENC to raise fraud from ~2 % to ~10 % (ratio 0.10) – small synthetic set.
> 2. Random-undersample the majority to 3× the minority – caps total rows.
> This mimics the golden notebook's balance without OOM.

```python
from imblearn.over_sampling  import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline       import Pipeline

# Identify which columns are categorical (by dtype or name)
CAT_NAMES = ["category", "merchant", "state", "gender", "city", "zip", "job"]
cat_cols_present = [c for c in CAT_NAMES if c in feature_cols]
cat_indices = [feature_cols.index(c) for c in cat_cols_present]

print(f"Categorical indices: {cat_indices}")
print(f"Train shape before resampling: {X_train_pd.shape}")
print(f"Fraud count before: {y_train_pd.sum():,} ({y_train_pd.mean()*100:.2f}%)")

# Build pipeline
over  = SMOTENC(categorical_features=cat_indices,
                sampling_strategy=0.10,   # raise minority to 10% of majority
                random_state=42)
under = RandomUnderSampler(sampling_strategy=0.33,  # final ratio ~1:3
                           random_state=42)

pipeline = Pipeline(steps=[("over", over), ("under", under)])
X_res, y_res = pipeline.fit_resample(X_train_pd, y_train_pd)

print(f"\nTrain shape after resampling : {X_res.shape}")
print(f"Fraud count after : {y_res.sum():,} ({y_res.mean()*100:.2f}%)")
```

---

## Cell 9 — Train CatBoost with Enhanced Params & MLflow  [REPLACE]

```python
# Categorical features list for CatBoost (column names)
CAT_NAMES_FOR_CB = [c for c in CAT_NAMES if c in feature_cols]

mlflow.set_experiment("fraud_detection_v2")

with mlflow.start_run(run_name="catboost_smotenc_v1") as run:

    # ── Model hyper-parameters ──────────────────────────────────
    model_params = {
        "iterations"           : 1500,
        "learning_rate"        : 0.03,      # slower LR → better generalisation
        "depth"                : 7,
        "eval_metric"          : "AUC",
        "early_stopping_rounds": 100,
        "verbose"              : 200,
        "random_seed"          : 42,
        "loss_function"        : "Logloss",
        # class_weights give extra push to recall without SMOTE
        "class_weights"        : [1, imbalance_ratio],
        "l2_leaf_reg"          : 5,
        "border_count"         : 128,
    }

    mlflow.log_params({
        "resampling"   : "SMOTENC+UnderSampler",
        "feature_count": len(feature_cols),
        **{k: v for k, v in model_params.items() if k != "verbose"},
    })

    # ── Train ────────────────────────────────────────────────────
    model = train_model(
        X_res, y_res,
        X_val_pd, y_val_pd,
        model_params=model_params,
    )

    # ── Threshold search ─────────────────────────────────────────
    best_threshold = find_best_threshold(model, X_val_pd, y_val_pd, n_folds=3)
    mlflow.log_param("optimal_threshold", round(best_threshold, 4))

    # ── Evaluate on held-out test set ────────────────────────────
    metrics = evaluate_model(model, X_test_pd, y_test_pd, threshold=best_threshold)

    mlflow.log_metric("test_auc",       metrics["auc"])
    mlflow.log_metric("test_f1",        metrics["f1"])
    mlflow.log_metric("test_precision", metrics["precision"])
    mlflow.log_metric("test_recall",    metrics["recall"])

    print("=" * 50)
    print(f"  Test AUC       : {metrics['auc']:.4f}")
    print(f"  Test F1        : {metrics['f1']:.4f}")
    print(f"  Test Precision : {metrics['precision']:.4f}")
    print(f"  Test Recall    : {metrics['recall']:.4f}")
    print(f"  Threshold      : {best_threshold:.4f}")
    print("=" * 50)

    # ── Log model to MLflow ─────────────────────────────────────
    mlflow.catboost.log_model(model, "catboost_fraud_model")
    run_id = run.info.run_id
    print(f"\nMLflow run ID: {run_id}")
```

---

## Cell 10 — Confusion Matrix Heatmap  [NEW]

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = np.array(metrics["confusion_matrix"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix – Test Set")
plt.tight_layout()
plt.savefig("/tmp/confusion_matrix.png", dpi=150)
mlflow.log_artifact("/tmp/confusion_matrix.png")
plt.show()
print("Saved confusion matrix.")
```

---

## Cell 11 — Feature Importance (CatBoost native)  [NEW]

```python
feature_importance = pd.DataFrame({
    "feature"   : feature_cols,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False)

print("Top 20 features by importance:")
print(feature_importance.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=feature_importance.head(20), y="feature", x="importance",
            orient="h", ax=ax, palette="viridis")
ax.set_title("Feature Importance (CatBoost)")
plt.tight_layout()
plt.savefig("/tmp/feature_importance.png", dpi=150)
mlflow.log_artifact("/tmp/feature_importance.png")
plt.show()
```

---

## Cell 12 — SHAP Summary Plot (previously unused)  [NEW]

```python
# Use a 2 000-row sample to keep SHAP fast on Databricks Free
sample_idx = np.random.choice(len(X_test_pd), size=min(2000, len(X_test_pd)), replace=False)
X_shap_sample = X_test_pd.iloc[sample_idx]

shap_df, shap_values = compute_shap_reasons(model, X_shap_sample)

# Beeswarm / summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap_sample, show=False)
plt.tight_layout()
plt.savefig("/tmp/shap_summary.png", dpi=150)
mlflow.log_artifact("/tmp/shap_summary.png")
plt.show()
print("SHAP summary saved.")
```

---

## Cell 13 — Pickle Model + Metadata  [NEW]

```python
import pickle, json

# Build a self-contained artefact dict
model_artifact = {
    "model"          : model,
    "feature_cols"   : feature_cols,
    "cat_features"   : CAT_NAMES_FOR_CB,
    "threshold"      : best_threshold,
    "metrics"        : {k: v for k, v in metrics.items() if k != "confusion_matrix"},
    "imbalance_ratio": imbalance_ratio,
    "mlflow_run_id"  : run_id,
}

PICKLE_PATH = "/Workspace/Users/mohamed.c.elshenity@gmail.com/fraud/model_artifact.pkl"
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(model_artifact, f)

print(f"Model artifact pickled to: {PICKLE_PATH}")
print(json.dumps(model_artifact["metrics"], indent=2))
```

---

## Cell 14 — Quick Reload Smoke-Test  [NEW]

```python
with open(PICKLE_PATH, "rb") as f:
    loaded = pickle.load(f)

print("Loaded keys:", list(loaded.keys()))
print("Threshold  :", loaded["threshold"])
print("Metrics    :", loaded["metrics"])

# Quick sanity: re-run evaluation with loaded model
metrics_check = evaluate_model(
    loaded["model"], X_test_pd, y_test_pd,
    threshold=loaded["threshold"]
)
print(f"\nSanity-check AUC: {metrics_check['auc']:.4f}  (should match above)")
```

---

## Cell 15 — Stop Spark  [REPLACE]

```python
spark.catalog.clearCache()
spark.stop()
print("Spark session stopped.")
```

---

# Summary of Changes vs. Original 03_model_training.py

| # | What changed | Why |
|---|---|---|
| Cell 4 | Added class-imbalance check | Know the ratio before choosing strategy |
| Cell 6 | Feature list expanded | Activates `haversine_distance`, lookup fraud-rate columns, velocity features – all were computed but never fed to the model |
| Cell 8 | SMOTENC + RandomUnderSampler pipeline | Balances training data without OOM – mirrors the golden notebook's strategy |
| Cell 9 | `class_weights`, deeper tree (depth 7), slower LR, L2 reg, 1 500 iters | Matches golden notebook CatBoost tuning; `class_weights` double-guards recall |
| Cell 10 | Confusion matrix heatmap | Makes TP/FP/FN instantly readable; logged to MLflow |
| Cell 11 | Feature importance bar chart | Confirms the new features actually contribute |
| Cell 12 | SHAP summary | Activates the previously dead `shap_explainer.py` |
| Cell 13-14 | Pickle + smoke-test | Stores the full artefact (model + metadata) for downstream scoring |
