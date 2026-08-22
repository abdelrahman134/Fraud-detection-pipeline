from kafka import KafkaProducer
import csv
import json
import time

# =========================
# CONFIGURATION
# =========================

KAFKA_BROKER = "localhost:9094"

TOPIC_NAME = "fraud-historical-topic"

CSV_FILE = "generated_data/historical_transactions.csv"

BATCH_SIZE = 5000

# =========================
# CREATE OPTIMIZED PRODUCER
# =========================

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,

    value_serializer=lambda v: json.dumps(v).encode("utf-8"),

    # Wait for Kafka acknowledgement
    acks="all",

    # Compress messages
    compression_type="gzip",

    # Batch messages internally
    batch_size=16384,

    # Wait a little to fill batches
    linger_ms=5,

    # Memory buffer
    buffer_memory=33554432,

    # Retry failed sends
    retries=5
)

# =========================
# SEND DATA
# =========================

print("\nStarting historical batch ingestion...\n")

messages_sent = 0
start_time = time.time()

with open(CSV_FILE, "r") as file:

    reader = csv.DictReader(
        file,
        delimiter="|"
    )

    for row in reader:

        producer.send(
            TOPIC_NAME,
            value=row
        )

        messages_sent += 1

        # Flush every batch
        if messages_sent % BATCH_SIZE == 0:

            producer.flush()

            elapsed = time.time() - start_time

            print(
                f"{messages_sent} messages sent "
                f"in {elapsed:.2f} seconds"
            )

# =========================
# FINAL FLUSH
# =========================

producer.flush()
producer.close()

elapsed = time.time() - start_time

print("\nHistorical ingestion complete.")
print(f"Total messages sent: {messages_sent}")
print(f"Total time: {elapsed:.2f} seconds\n")
