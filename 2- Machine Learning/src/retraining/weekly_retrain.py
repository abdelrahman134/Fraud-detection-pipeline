# src/retraining/weekly_retrain.py
import sys, os, subprocess

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost", "shap"])

import mlflow, mlflow.catboost
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F

sys.path.append(os.path.dirname(os.getcwd()))
from features.static_features import build_static_features
from features.window_features import build_window_features
from features.geospatial_features import build_geospatial_features
from features.lookup_features import build_lookup_features
from model.train import train_model
from model.threshold import find_best_threshold
from model.evaluate import evaluate_model

def run_weekly_retraining():
    spark = SparkSession.builder.appName("FraudDetectionWeeklyRetrain").getOrCreate()
    
    # 1. Load historical labeled data from Delta Lake / S3
    df_raw = spark.table("fraud_features") # Or spark.read.parquet("s3://bucket/historical/")
    
    # 2. Extract features
    df = build_static_features(df_raw)
    df = build_window_features(df)
    df = build_geospatial_features(df)
    
    # 3. Time-based Train / Val / Test Split
    df = df.orderBy("unix_time")
    total = df.count()
    train_end, val_end = int(total * 0.70), int(total * 0.85)
    
    w = Window.orderBy("unix_time")
    df = df.withColumn("row_idx", F.row_number().over(w))
    train_df = df.filter(F.col("row_idx") <= train_end)
    val_df   = df.filter((F.col("row_idx") > train_end) & (F.col("row_idx") <= val_end))
    test_df  = df.filter(F.col("row_idx") > val_end)
    
    # Target Encoding
    train_df = build_lookup_features(train_df, train_df)
    val_df   = build_lookup_features(val_df, train_df)
    test_df  = build_lookup_features(test_df, train_df)
    
    # All features from notebook 03
    FEATURE_COLS = [
        # ── Numeric / Static ──────────────────────────────────────────
        "amt", "city_pop", "hour", "day_of_week", "month",
        # ── Geospatial ────────────────────────────────────────────────
        "distance", "haversine_distance",
        # ── Window / Velocity features ────────────────────────────────
        "txn_count_1h", "txn_count_24h", "avg_amt_24h", "spend_24h",
        "unique_merchants_24h", "time_since_last_txn",
        # ── Lookup / Fraud-Rate features ──────────────────────────────
        "category_fraud_rate", "category_txn_count", "merchant_fraud_rate",
        "merchant_txn_count", "state_fraud_rate",
        # ── Categorical (passed as-is to CatBoost) ────────────────────
        "category", "merchant", "state", "gender", "city", "zip", "job",
    ]
    
    # Keep only columns that actually exist in the processed table
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    
    X_train_pd = train_df.select(feature_cols).toPandas()
    y_train_pd = train_df.select("is_fraud").toPandas()["is_fraud"].astype(int)
    X_val_pd   = val_df.select(feature_cols).toPandas()
    y_val_pd   = val_df.select("is_fraud").toPandas()["is_fraud"].astype(int)
    X_test_pd  = test_df.select(feature_cols).toPandas()
    y_test_pd  = test_df.select("is_fraud").toPandas()["is_fraud"].astype(int)
    
    # 4. LOCKED-IN CHAMPION PARAMETERS (Discovered in notebook 03)
    champion_params = {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "AUC",
        "early_stopping_rounds": 100,
        "scale_pos_weight": 5.0,     # Champion setting
        "l2_leaf_reg": 15,           # Champion setting
        "random_seed": 42
    }
    
    # 5. Train & Evaluate
    mlflow.set_experiment("/Shared/fraud_detection")
    with mlflow.start_run(run_name="catboost_weekly_champion") as run:
        model = train_model(X_train_pd, y_train_pd, X_val_pd, y_val_pd, model_params=champion_params)
        
        # Operational cutoff threshold from notebook 03
        production_threshold = 0.9980
        metrics = evaluate_model(model, X_test_pd, y_test_pd, threshold=production_threshold)
        
        # Log to MLflow Registry
        mlflow.log_params(champion_params)
        mlflow.log_param("optimal_threshold", production_threshold)
        
        # Separate scalar metrics from confusion matrix
        scalar_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        mlflow.log_metrics(scalar_metrics)
        
        # Log confusion matrix as a dict artifact
        if "confusion_matrix" in metrics:
            import json
            mlflow.log_dict({
                "confusion_matrix": metrics["confusion_matrix"],
                "labels": ["legit", "fraud"]
            }, "confusion_matrix.json")
        
        # Infer model signature for Unity Catalog
        from mlflow.models import infer_signature
        predictions = model.predict_proba(X_test_pd)[:, 1]
        signature = infer_signature(X_test_pd, predictions)
        
        # Log & Promote Model to Production
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="catboost_model",
            registered_model_name="fraud_detection_catboost",
            signature=signature
        )
        print(f"Model trained and registered to MLflow. Version logged: {model_info.model_uri}")

if __name__ == "__main__":
    run_weekly_retraining()
