# src/scoring/realtime_detection.py
import sys, os, mlflow, mlflow.pyfunc
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

sys.path.append(os.path.abspath("src"))
from features.static_features import build_static_features
from features.window_features import build_window_features
from features.geospatial_features import build_geospatial_features

def run_realtime_detection():
    spark = SparkSession.builder \
        .appName("FraudDetectionRealtimeListener") \
        .config("spark.sql.streaming.schemaInference", "true") \
        .getOrCreate()
        
    raw_s3_path = "s3://your-bucket/live_transactions/"
    checkpoint_path = "s3://your-bucket/checkpoints/fraud_stream/"
    output_s3_path = "s3://your-bucket/scored_transactions/"
    
    # 1. Continuous Listener on S3 (PySpark Structured Streaming)
    df_stream = spark.readStream.parquet(raw_s3_path)
    
    # 2. Extract Features on Stream
    df_features = build_static_features(df_stream)
    df_features = build_window_features(df_features)
    df_features = build_geospatial_features(df_features)
    
    # 3. Load Active Production Model from MLflow Registry as Spark UDF
    model_uri = "models:/fraud_detection_catboost/Production" # or latest version
    predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
    
    feature_cols =[
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
    ] # Match 20 feature columns
    
    # 4. Score Transaction Probability & Apply Champion Threshold (0.9980)
    PRODUCTION_THRESHOLD = 0.9980
    
    df_scored = df_features \
        .withColumn("fraud_probability", predict_udf(*feature_cols)) \
        .withColumn("is_fraud_predicted", F.when(F.col("fraud_probability") >= PRODUCTION_THRESHOLD, 1).otherwise(0))
        
    # 5. Write Scored Results back to S3 continuously
    query = df_scored.writeStream \
        .format("parquet") \
        .option("checkpointLocation", checkpoint_path) \
        .option("path", output_s3_path) \
        .start()
        
    query.awaitTermination()

if __name__ == "__main__":
    run_realtime_detection()
