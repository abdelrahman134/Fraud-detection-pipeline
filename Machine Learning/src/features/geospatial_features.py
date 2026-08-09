
"""
Geospatial feature engineering for fraud detection.
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def build_geospatial_features(df: DataFrame) -> DataFrame:
    """
    Build geospatial features from transaction data.
    
    Args:
        df: Input PySpark DataFrame
        
    Returns:
        DataFrame with geospatial features added
    """
    # Haversine distance calculation (more accurate for Earth)
    R = 6371.0  # Earth radius in km
    df = df.withColumn(
        "lat_rad", F.radians("lat")
    ).withColumn(
        "merch_lat_rad", F.radians("merch_lat")
    ).withColumn(
        "delta_lat", F.radians(F.col("merch_lat") - F.col("lat"))
    ).withColumn(
        "delta_lon", F.radians(F.col("merch_long") - F.col("long"))
    ).withColumn(
        "a",
        F.sin(F.col("delta_lat") / 2) ** 2 +
        F.cos(F.col("lat_rad")) * F.cos(F.col("merch_lat_rad")) *
        F.sin(F.col("delta_lon") / 2) ** 2
    ).withColumn(
        "haversine_distance",
        2 * R * F.atan2(F.sqrt(F.col("a")), F.sqrt(1 - F.col("a")))
    ).drop("lat_rad", "merch_lat_rad", "delta_lat", "delta_lon", "a")
    
    return df

