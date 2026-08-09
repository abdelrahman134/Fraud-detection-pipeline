
"""
Window-based feature engineering for fraud detection.
"""
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F


def build_window_features(df: DataFrame) -> DataFrame:
    """
    Build time-based window features for fraud detection.
    
    Args:
        df: Input PySpark DataFrame with timestamp and cc_num columns
        
    Returns:
        DataFrame with window features added
    """
    # Define window spec for cardholder
    cc_window = Window.partitionBy("cc_num").orderBy("unix_time")
    
    # Number of transactions in last 1 hour (3600 seconds)
    df = df.withColumn(
        "txn_count_1h",
        F.count("*").over(cc_window.rangeBetween(-3600, 0))
    )
    
    # Number of transactions in last 24 hours (86400 seconds)
    df = df.withColumn(
        "txn_count_24h",
        F.count("*").over(cc_window.rangeBetween(-86400, 0))
    )
    
    # Average amount in last 24 hours
    df = df.withColumn(
        "avg_amt_24h",
        F.avg("amt").over(cc_window.rangeBetween(-86400, 0))
    )
    
    # Total spend in last 24 hours
    df = df.withColumn(
        "spend_24h",
        F.sum("amt").over(cc_window.rangeBetween(-86400, 0))
    )
    
    # Number of unique merchants in last 24 hours
    df = df.withColumn(
        "unique_merchants_24h",
        F.size(F.collect_set("merchant").over(cc_window.rangeBetween(-86400, 0)))
    )
    
    # Time since last transaction for the cardholder
    df = df.withColumn(
        "time_since_last_txn",
        F.col("unix_time") - F.lag("unix_time", 1).over(cc_window)
    )
    
    return df

