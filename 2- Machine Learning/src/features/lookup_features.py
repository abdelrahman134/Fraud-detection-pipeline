
"""
Lookup table-based feature engineering for fraud detection.
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def build_lookup_features(df: DataFrame, train_df: DataFrame) -> DataFrame:
    """
    Build features using lookup tables from training data.
    
    Args:
        df: Input PySpark DataFrame to add features to
        train_df: Training DataFrame to compute lookup statistics from
        
    Returns:
        DataFrame with lookup features added
    """
    # Fraud rate by category
    category_fraud_rate = train_df.groupBy("category").agg(
        F.avg("is_fraud").alias("category_fraud_rate"),
        F.count("*").alias("category_txn_count")
    )
    df = df.join(category_fraud_rate, on="category", how="left")
    
    # Fraud rate by merchant
    merchant_fraud_rate = train_df.groupBy("merchant").agg(
        F.avg("is_fraud").alias("merchant_fraud_rate"),
        F.count("*").alias("merchant_txn_count")
    )
    df = df.join(merchant_fraud_rate, on="merchant", how="left")
    
    # Fraud rate by state
    state_fraud_rate = train_df.groupBy("state").agg(
        F.avg("is_fraud").alias("state_fraud_rate")
    )
    df = df.join(state_fraud_rate, on="state", how="left")
    
    return df

