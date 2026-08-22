import os
import glob
import shutil
import subprocess
from datetime import datetime, timedelta
import random

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = (
    "/home/ubuntu/Fraud-Detection-Project"
)

DATAGEN_DIR = os.path.join(
    PROJECT_ROOT,
    "data_generation"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "generated_data"
)

STREAM_DIR = os.path.join(
    OUTPUT_DIR,
    "stream_batches"
)

CUSTOMERS_FILE = os.path.join(
    OUTPUT_DIR,
    "customers.csv"
)

HISTORICAL_FILE = os.path.join(
    OUTPUT_DIR,
    "historical_transactions.csv"
)

STATE_FILE = os.path.join(
    BASE_DIR,
    "stream_state.txt"
)

# IMPORTANT:
# temp output must be inside data_generation
# because Sparkov uses relative paths
TEMP_OUTPUT_DIR = os.path.join(
    DATAGEN_DIR,
    "temp_output"
)

# =========================
# CONFIGURATION
# =========================

CUSTOMER_COUNT = 2000

STREAM_START_DATE = "07-01-2024"

MAX_STREAM_TRANSACTIONS = 500

# =========================
# CREATE DIRECTORIES
# =========================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    STREAM_DIR,
    exist_ok=True
)

# =========================
# DETERMINE MODE
# =========================

historical_exists = os.path.exists(
    HISTORICAL_FILE
)

# =========================
# HISTORICAL MODE
# =========================

if not historical_exists:

    MODE = "historical"

    START_DATE = "01-01-2024"

    END_DATE = "06-30-2024"

    FINAL_OUTPUT_FILE = HISTORICAL_FILE

# =========================
# STREAM MODE
# =========================

else:

    MODE = "stream"

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    FINAL_OUTPUT_FILE = os.path.join(
        STREAM_DIR,
        f"stream_batch_{timestamp}.csv"
    )

    if not os.path.exists(STATE_FILE):

        current_time = datetime.strptime(
            STREAM_START_DATE,
            "%m-%d-%Y"
        )

    else:

        with open(
            STATE_FILE,
            "r"
        ) as f:

            current_time = datetime.strptime(
                f.read().strip(),
                "%Y-%m-%d %H:%M:%S"
            )

    next_time = current_time + timedelta(
        days=1
    )

    START_DATE = current_time.strftime(
        "%m-%d-%Y"
    )

    END_DATE = current_time.strftime(
        "%m-%d-%Y"
    )

print(
    f"\nRunning in {MODE.upper()} mode...\n"
)

# =========================
# CLEAN TEMP DIRECTORY
# =========================

if os.path.exists(
    TEMP_OUTPUT_DIR
):

    print(
        "Removing old temp directory...\n"
    )

    shutil.rmtree(
        TEMP_OUTPUT_DIR
    )

os.makedirs(
    TEMP_OUTPUT_DIR,
    exist_ok=True
)

# =========================
# GENERATE DATA
# =========================

print(
    "Generating Sparkov transaction data...\n"
)

command = [

    "/home/ubuntu/Fraud-Detection-Project/venv/bin/python",

    "datagen.py",

    "-n",
    str(CUSTOMER_COUNT),

    "-o",
    TEMP_OUTPUT_DIR,

    START_DATE,
    END_DATE
]

# =========================
# REUSE CUSTOMERS
# =========================

if (
    MODE == "stream"
    and os.path.exists(CUSTOMERS_FILE)
):

    print(
        "Reusing existing customers file...\n"
    )

    command.extend([
        "-c",
        CUSTOMERS_FILE
    ])

# =========================
# RUN GENERATOR
# =========================

result = subprocess.run(
    command,
    cwd=DATAGEN_DIR
)

if result.returncode != 0:

    print(
        "\nData generation failed.\n"
    )

    exit(1)

# =========================
# SAVE CUSTOMERS
# =========================

temp_customers = os.path.join(
    TEMP_OUTPUT_DIR,
    "customers.csv"
)

if MODE == "historical":

    if os.path.exists(
        temp_customers
    ):

        shutil.copy(
            temp_customers,
            CUSTOMERS_FILE
        )

        print(
            "Customers file saved.\n"
        )

# =========================
# FIND CSV FILES
# =========================

print(
    "Merging transaction CSV files...\n"
)

csv_files = glob.glob(
    os.path.join(
        TEMP_OUTPUT_DIR,
        "*.csv"
    )
)

transaction_files = []

for file in csv_files:

    if (
        "customers.csv"
        not in file
    ):

        transaction_files.append(
            file
        )

if not transaction_files:

    print(
        "\nNo transaction files found.\n"
    )

    print(
        f"Checked path:\n{TEMP_OUTPUT_DIR}\n"
    )

    exit(1)

transaction_files.sort()

# =========================
# MERGE FILES
# =========================

with open(
    FINAL_OUTPUT_FILE,
    "w"
) as outfile:

    with open(
        transaction_files[0],
        "r"
    ) as first_file:

        header = first_file.readline()

        outfile.write(
            header
        )

    for file in transaction_files:

        with open(
            file,
            "r"
        ) as infile:

            next(
                infile,
                None
            )

            for line in infile:

                if line.count("|") < 25:

                    continue

                outfile.write(
                    line
                )

print(
    "Merge complete.\n"
)


# =========================
# LIMIT STREAM BATCH
# =========================

if MODE == "stream":

    print(
        "Sampling stream batch...\n"
    )

    with open(
        FINAL_OUTPUT_FILE,
        "r"
    ) as f:

        lines = f.readlines()

    if len(lines) > 1:

        header = lines[0]

        header_cols = header.strip().split("|")

        fraud_index = None

        if "is_fraud" in header_cols:

            fraud_index = header_cols.index(
                "is_fraud"
            )

            header_cols.remove(
                "is_fraud"
            )

        header = (
            "|".join(header_cols)
            + "\n"
        )

        data = lines[1:]

        # Randomly sample transactions
        if len(data) > MAX_STREAM_TRANSACTIONS:

            data = random.sample(
                data,
                MAX_STREAM_TRANSACTIONS
            )

        # Remove is_fraud column
        cleaned_data = []

        for row in data:

            cols = row.strip().split("|")

            if (
                fraud_index is not None
                and fraud_index < len(cols)
            ):

                del cols[fraud_index]

            cleaned_data.append(
                "|".join(cols) + "\n"
            )

        data = cleaned_data

        with open(
            FINAL_OUTPUT_FILE,
            "w"
        ) as f:

            f.write(
                header
            )

            f.writelines(
                data
            )

        print(
            f"Randomly selected "
            f"{len(data)} transactions.\n"
        )


# =========================
# CLEAN TEMP FILES
# =========================

print(
    "Cleaning temporary files...\n"
)

if os.path.exists(
    TEMP_OUTPUT_DIR
):

    shutil.rmtree(
        TEMP_OUTPUT_DIR
    )

print(
    "Cleanup complete.\n"
)

# =========================
# SAVE STREAM STATE
# =========================

if MODE == "stream":

    with open(
        STATE_FILE,
        "w"
    ) as f:

        f.write(
            next_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    print(
        f"Next stream state: "
        f"{next_time}"
    )

# =========================
# FINAL OUTPUT
# =========================

print("\nDone.\n")

print("Generated files:")

if os.path.exists(
    CUSTOMERS_FILE
):

    print(
        f"- {CUSTOMERS_FILE}"
    )

print(
    f"- {FINAL_OUTPUT_FILE}"
)
