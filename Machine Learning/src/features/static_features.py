
"""
Static feature engineering for fraud detection.
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def build_static_features(df: DataFrame) -> DataFrame:
    """
    Build static features from raw transaction data.
    
    Args:
        df: Input PySpark DataFrame
        
    Returns:
        DataFrame with static features added
    """
    # Convert string columns to appropriate types
    df = df.withColumn("amt", F.col("amt").cast("double"))
    df = df.withColumn("is_fraud", F.col("is_fraud").cast("integer"))
    df = df.withColumn("lat", F.col("lat").cast("double"))
    df = df.withColumn("long", F.col("long").cast("double"))
    df = df.withColumn("merch_lat", F.col("merch_lat").cast("double"))
    df = df.withColumn("merch_long", F.col("merch_long").cast("double"))
    df = df.withColumn("city_pop", F.col("city_pop").cast("integer"))
    df = df.withColumn("unix_time", F.col("unix_time").cast("long"))
    
    # Create timestamp column
    df = df.withColumn(
        "timestamp",
        F.to_timestamp(F.concat_ws(" ", F.col("trans_date"), F.col("trans_time")), "yyyy-MM-dd HH:mm:ss")
    )
    
    # Extract time-based features
    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("day_of_week", F.dayofweek("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    
    # Calculate distance between customer and merchant
    df = df.withColumn(
        "distance",
        F.sqrt(
            F.pow(F.col("lat") - F.col("merch_lat"), 2) + 
            F.pow(F.col("long") - F.col("merch_long"), 2)
        )
    )
    
    return df
