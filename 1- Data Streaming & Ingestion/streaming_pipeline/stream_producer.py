from kafka import KafkaProducer
import csv
import json
import time
import glob
import os

# =========================
# CONFIGURATION
# =========================

#KAFKA_BROKER = "localhost:9094"

KAFKA_BROKER = "localhost:80"

TOPIC_NAME = "fraud-stream-topic"

STREAM_DIR = "generated_data/stream_batches"

POLL_INTERVAL = 60    #producer checks for new generated data after 1 min

PROCESSED_FILES_STATE = "processed_files.txt"

# =========================
# LOAD PROCESSED FILES
# =========================

if os.path.exists(PROCESSED_FILES_STATE):

    with open(PROCESSED_FILES_STATE, "r") as f:

        processed_files = set(
            line.strip()
            for line in f.readlines()
        )

else:

    processed_files = set()

# =========================
# CREATE PRODUCER
# =========================

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,

    value_serializer=lambda v:
    json.dumps(v).encode("utf-8"),

    acks="all",

    compression_type="gzip",

    retries=5
)

print("\nStreaming producer started...\n")

# =========================
# CONTINUOUS LOOP
# =========================

while True:

    csv_files = sorted(
        glob.glob(
            f"{STREAM_DIR}/*.csv"
        )
    )

    new_files_found = False

    for file_path in csv_files:

        file_name = os.path.basename(
            file_path
        )

        # Skip already processed files
        if file_name in processed_files:

            continue

        new_files_found = True

        print(
            f"\nProcessing file: "
            f"{file_name}"
        )

        messages_sent = 0

        with open(file_path, "r") as file:

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

        producer.flush()

        print(
            f"Sent {messages_sent} messages "
            f"from {file_name}"
        )

        # =========================
        # SAVE FILE AS PROCESSED
        # =========================

        processed_files.add(file_name)

        with open(
            PROCESSED_FILES_STATE,
            "a"
        ) as f:

            f.write(file_name + "\n")

    if not new_files_found:

        print(
            "\nNo new stream batches found..."
        )

    time.sleep(POLL_INTERVAL)
