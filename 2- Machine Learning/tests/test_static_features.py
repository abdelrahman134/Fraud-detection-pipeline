
"""
Test static feature engineering.
"""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys
import os

sys.path.append(os.path.abspath('../src'))
from features.static_features import build_static_features


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("TestStaticFeatures") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sample_data(spark):
    schema = StructType([
        StructField("ssn", StringType()),
        StructField("cc_num", StringType()),
        StructField("lat", StringType()),
        StructField("long", StringType()),
        StructField("merch_lat", StringType()),
        StructField("merch_long", StringType()),
        StructField("amt", StringType()),
        StructField("is_fraud", StringType()),
        StructField("trans_date", StringType()),
        StructField("trans_time", StringType()),
        StructField("unix_time", StringType()),
        StructField("city_pop", StringType()),
    ])
    
    data = [
        ("123-45-6789", "4111111111111111", "40.7128", "-74.0060", "40.7306", "-73.9352", "100.0", "0", "2024-01-01", "12:00:00", "1704067200", "1000000"),
        ("123-45-6789", "4111111111111111", "40.7128", "-74.0060", "40.7580", "-73.9855", "200.0", "1", "2024-01-01", "13:00:00", "1704070800", "1000000"),
    ]
    
    return spark.createDataFrame(data, schema)


def test_build_static_features(spark, sample_data):
    result_df = build_static_features(sample_data)
    
    # Check new columns exist
    assert "timestamp" in result_df.columns
    assert "hour" in result_df.columns
    assert "day_of_week" in result_df.columns
    assert "month" in result_df.columns
    assert "distance" in result_df.columns
    
    # Check data types
    assert result_df.schema["amt"].dataType == DoubleType()
    assert result_df.schema["is_fraud"].dataType == IntegerType()
    assert result_df.schema["hour"].dataType == IntegerType()
    
    # Check row count preserved
    assert result_df.count() == 2

