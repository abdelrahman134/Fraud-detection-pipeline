# src/scoring/test_detection.py
import sys, os, subprocess

# Install required packages if needed
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost", "shap"])

# Monkey-patch packaging.version.Version to handle Databricks Serverless version strings
import packaging.version
_original_version_init = packaging.version.Version.__init__
def _patched_version_init(self, version):
    # Clean up Databricks Serverless version strings before parsing
    if isinstance(version, str) and 'aarch64' in version or 'photon' in version or 'scala' in version:
        # Extract just the numeric part: "18.x-aarch64-photon-scala2" -> "18.0"
        version = version.split('-')[0].replace('.x', '.0')
    _original_version_init(self, version)
packaging.version.Version.__init__ = _patched_version_init

import mlflow, mlflow.pyfunc
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Ensure src/ directory is in python path (matching weekly_retrain.py)
sys.path.append(os.path.dirname(os.getcwd()))
from features.static_features import build_static_features
from features.window_features import build_window_features
from features.geospatial_features import build_geospatial_features
from features.lookup_features import build_lookup_features


def run_test_detection():
    # =========================================================================
    # 📌 CONFIGURABLE INPUTS & PATHS (MODIFY THESE FOR YOUR DATABRICKS TEST)
    # =========================================================================
    
    # 1. INPUT DATA SOURCE
    # Options: "table" (Databricks table) or "parquet" (File path)
    DATA_SOURCE_TYPE = "parquet"  
    
    # If DATA_SOURCE_TYPE = "table": Specify your Databricks table name
    #INPUT_TABLE_NAME = "fraud_features"  
    
    # If DATA_SOURCE_TYPE = "parquet": Specify your workspace/DBFS parquet path
    INPUT_PARQUET_PATH = "/Workspace/Users/mohamed.c.elshenity@gmail.com/fraud/parquet"
    
    # 2. OUTPUT DESTINATION
    # Scored predictions will be saved to this Databricks table
    OUTPUT_TABLE_NAME = "fraud_predictions"
    
    # 3. MLFLOW MODEL REGISTRY POINTER
    # Matches registered_model_name="fraud_detection_catboost" in weekly_retrain.py
    # Unity Catalog models require specific version numbers, not "latest"
    MODEL_URI = "models:/workspace.default.fraud_detection_catboost/1"
    
    # 4. CHAMPION DECISION THRESHOLD (Locked from notebook 03 & weekly_retrain.py)
    PRODUCTION_THRESHOLD = 0.9980
    
    # =========================================================================

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("FraudDetectionScoringTest") \
        .config("spark.sql.shuffle.partitions", "16") \
        .getOrCreate()
        
    print(f"Loading incoming data for scoring from {DATA_SOURCE_TYPE}...")

    # 1. Load Data
    df_raw = spark.read.parquet(INPUT_PARQUET_PATH)

    print(f"Loaded {df_raw.count():,} records for fraud scoring.")

    # 2. Build All Features (Using imported modular functions)
    df_features = build_static_features(df_raw)
    df_features = build_window_features(df_features)
    df_features = build_geospatial_features(df_features)
    
    # Lookup features require historical statistics (using input dataset as lookup source)
    df_features = build_lookup_features(df_features, df_features)

    # 3. Match the 20 Feature Columns used in weekly_retrain.py
    FEATURE_COLS = [
        # Numeric / Static
        "amt", "city_pop", "hour", "day_of_week", "month",
        # Geospatial
        "distance", "haversine_distance",
        # Window / Velocity
        "txn_count_1h", "txn_count_24h", "avg_amt_24h", "spend_24h",
        "unique_merchants_24h", "time_since_last_txn",
        # Lookup / Fraud Rate
        "category_fraud_rate", "category_txn_count", "merchant_fraud_rate",
        "merchant_txn_count", "state_fraud_rate",
        # Categorical
        "category", "merchant", "state", "gender", "city", "zip", "job",
    ]
    
    # Keep only feature columns that exist in the DataFrame
    feature_cols = [c for c in FEATURE_COLS if c in df_features.columns]
    print(f"Scoring using {len(feature_cols)} features...")

    # 4. Load Model from MLflow Registry on Driver
    # Use batch prediction to avoid worker dependency issues on Serverless
    print(f"Loading MLflow model from: {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    
    # 5. Batch Predict: Convert to Pandas, Score, Join Back
    # Add unique ID for joining predictions back
    from pyspark.sql.functions import monotonically_increasing_id
    df_with_id = df_features.withColumn("_temp_id", monotonically_increasing_id())
    
    # Select only needed columns for prediction
    df_for_scoring = df_with_id.select("_temp_id", *feature_cols)
    
    # Convert to Pandas and predict on driver
    pdf = df_for_scoring.toPandas()
    
    # Convert numeric types to match model signature
    if 'merchant_txn_count' in pdf.columns:
        pdf['merchant_txn_count'] = pdf['merchant_txn_count'].astype('float64')
    
    predictions = model.predict(pdf[feature_cols])
    
    # Add predictions to pandas DataFrame
    pdf["fraud_probability"] = predictions
    pdf["is_fraud_predicted"] = (pdf["fraud_probability"] >= PRODUCTION_THRESHOLD).astype(int)
    
    # Convert predictions back to Spark and join with original data
    df_predictions = spark.createDataFrame(pdf[["_temp_id", "fraud_probability", "is_fraud_predicted"]])
    df_scored = df_with_id.join(df_predictions, on="_temp_id").drop("_temp_id")

    # 6. Display Summary & Save Scored Results
    fraud_detected_count = df_scored.filter(F.col("is_fraud_predicted") == 1).count()
    print("=" * 60)
    print(f"SCORING COMPLETE | Total Scored: {df_scored.count():,}")
    print(f"FRAUD DETECTED   | Flagged Alerts: {fraud_detected_count:,}")
    print(f"THRESHOLD USED   | Cutoff: {PRODUCTION_THRESHOLD}")
    print("=" * 60)

    # Save scored results to Databricks table
    df_scored.write.mode("overwrite").saveAsTable(OUTPUT_TABLE_NAME)
    print(f"Scored output saved to Databricks table: '{OUTPUT_TABLE_NAME}'")


if __name__ == "__main__":
    run_test_detection()

