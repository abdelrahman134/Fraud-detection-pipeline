from kafka import KafkaConsumer
import boto3
import pandas as pd
import json
import os

# =========================
# CONFIG
# =========================

KAFKA_BROKER = "localhost:9094"

TOPIC_NAME = "fraud-historical-topic"

BUCKET_NAME = "fraud-detection-historical"

PREFIX = "historical_transactions_parquet/"

OFFSET_FILE = "consumer_offset.txt"

BATCH_SIZE = 5000

LOCAL_FILE = "batch.parquet"

# =========================
# LOAD OFFSET
# =========================

if os.path.exists(OFFSET_FILE):

    with open(OFFSET_FILE, "r") as f:

        START_OFFSET = int(f.read().strip())

else:

    START_OFFSET = 0

print(f"\nStarting from offset: {START_OFFSET}\n")

# =========================
# KAFKA CONSUMER
# =========================

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    consumer_timeout_ms=10000,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

# =========================
# S3 CLIENT
# =========================

s3 = boto3.client("s3")

# =========================
# CONSUME
# =========================

batch = []

current_offset = START_OFFSET

file_counter = START_OFFSET

for message in consumer:

    if message.offset < START_OFFSET:
        continue

    batch.append(message.value)

    current_offset = message.offset

    if len(batch) >= BATCH_SIZE:

        print(
            f"\nProcessing offsets "
            f"{file_counter} -> {current_offset}"
        )

        # =========================
        # CREATE DATAFRAME
        # =========================

        df = pd.DataFrame(batch)

        # =========================
        # SAVE PARQUET
        # =========================

        df.to_parquet(
            LOCAL_FILE,
            engine="pyarrow",
            index=False
        )

        # =========================
        # UPLOAD TO S3
        # =========================

        s3.upload_file(
            LOCAL_FILE,
            BUCKET_NAME,
            f"{PREFIX}batch_{file_counter}.parquet"
        )

        # =========================
        # SAVE OFFSET
        # =========================

        with open(OFFSET_FILE, "w") as f:

            f.write(str(current_offset + 1))

        print(
            f"Uploaded parquet batch "
            f"up to offset {current_offset}"
        )

        batch = []

        file_counter += BATCH_SIZE

# =========================
# FINAL BATCH
# =========================

if batch:

    df = pd.DataFrame(batch)

    df.to_parquet(
        LOCAL_FILE,
        engine="pyarrow",
        index=False
    )

    s3.upload_file(
        LOCAL_FILE,
        BUCKET_NAME,
        f"{PREFIX}batch_{file_counter}.parquet"
    )

    with open(OFFSET_FILE, "w") as f:

        f.write(str(current_offset + 1))

    print(
        f"Final parquet upload "
        f"up to offset {current_offset}"
    )

consumer.close()

print("\nHistorical ingestion complete.\n")
