
"""
Test window feature engineering.
"""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys
import os

sys.path.append(os.path.abspath('../src'))
from features.window_features import build_window_features


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("TestWindowFeatures") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sample_data(spark):
    schema = StructType([
        StructField("cc_num", StringType()),
        StructField("unix_time", LongType()),
        StructField("amt", DoubleType()),
        StructField("merchant", StringType()),
    ])
    
    data = [
        ("4111111111111111", 1704067200, 100.0, "Merchant A"),
        ("4111111111111111", 1704070800, 200.0, "Merchant B"),
        ("4111111111111111", 1704074400, 300.0, "Merchant A"),
        ("5555555555554444", 1704067200, 50.0, "Merchant C"),
    ]
    
    return spark.createDataFrame(data, schema)


def test_build_window_features(spark, sample_data):
    result_df = build_window_features(sample_data)
    
    # Check new columns exist
    expected_cols = [
        "txn_count_1h", "txn_count_24h", "avg_amt_24h", 
        "spend_24h", "unique_merchants_24h", "time_since_last_txn"
    ]
    for col in expected_cols:
        assert col in result_df.columns
    
    # Check row count preserved
    assert result_df.count() == 4

