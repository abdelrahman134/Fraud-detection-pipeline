from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col

# =========================
# CONFIG
# =========================

KAFKA_BROKER = "localhost:80"

TOPIC_NAME = "fraud-stream-topic"

CHECKPOINT_PATH = (
    "s3a://fraud-streaming-landing-zone/"
    "checkpoints/fraud_stream/"
)

OUTPUT_PATH = (
    "s3a://fraud-streaming-landing-zone/"
    "stream_transactions/"
)

# =========================
# SPARK SESSION
# =========================

spark = (
    SparkSession.builder
    .appName("FraudStreamProcessor")

    # S3 Credentials
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )

    # LOW MEMORY SETTINGS
    .config("spark.driver.memory", "512m")
    .config("spark.executor.memory", "512m")

    # REDUCE CPU + RAM
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.default.parallelism", "1")

    # DISABLE METRICS BUG
    .config("spark.sql.streaming.metricsEnabled", "false")

    # REDUCE UI MEMORY
    .config("spark.ui.enabled", "false")

    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nStarting stream consumer...\n")

# =========================
# SCHEMA
# =========================

schema = StructType([

    StructField("ssn", StringType()),
    StructField("cc_num", StringType()),
    StructField("first", StringType()),
    StructField("last", StringType()),
    StructField("gender", StringType()),
    StructField("street", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("zip", StringType()),
    StructField("lat", StringType()),
    StructField("long", StringType()),
    StructField("city_pop", StringType()),
    StructField("job", StringType()),
    StructField("dob", StringType()),
    StructField("acct_num", StringType()),
    StructField("profile", StringType()),
    StructField("trans_num", StringType()),
    StructField("trans_date", StringType()),
    StructField("trans_time", StringType()),
    StructField("unix_time", StringType()),
    StructField("category", StringType()),
    StructField("amt", StringType()),
    StructField("is_fraud", StringType()),
    StructField("merchant", StringType()),
    StructField("merch_lat", StringType()),
    StructField("merch_long", StringType())

])

# =========================
# READ STREAM
# =========================

df = (
    spark.readStream
    .format("kafka")

    .option(
        "kafka.bootstrap.servers",
        KAFKA_BROKER
    )

    .option(
        "subscribe",
        TOPIC_NAME
    )

    .option("failOnDataLoss", "false")

    # KEEP EARLIEST
    # CHECKPOINTS HANDLE RESUMING
    .option(
        "startingOffsets",
        "earliest"
    )

    # LIMIT DATA PER MICROBATCH
    #.option( "maxOffsetsPerTrigger", "1000")

    .load()
)

# =========================
# PARSE JSON
# =========================

json_df = df.selectExpr(
    "CAST(value AS STRING)"
)

parsed_df = (
    json_df
    .select(
        from_json(
            col("value"),
            schema
        ).alias("data")
    )
    .select("data.*")
)
'''
debug_query = (
    parsed_df.writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", "false")
    .start()
)
'''
# =========================
# WRITE STREAM TO S3
# =========================

query = (
    parsed_df.writeStream

    .format("parquet")

    .outputMode("append")

    .option(
        "checkpointLocation",
        CHECKPOINT_PATH
    )

    .option(
        "path",
        OUTPUT_PATH
    )

    .partitionBy("trans_date")

    # PROCESS EVERY 30 seconds
    .trigger(processingTime="30 seconds")

    .start()
)

print("\nStream consumer running...\n")

# =========================
# KEEP STREAM ALIVE
# =========================

query.awaitTermination()

'''
print("QUERY ACTIVE:", query.isActive)
print("QUERY STATUS:", query.status)
print("LAST PROGRESS:", query.lastProgress)
print("EXCEPTION:", query.exception())
'''
