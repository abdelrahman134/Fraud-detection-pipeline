#!/bin/bash

echo "Stopping Producer..."

pkill -f stream_producer.py

echo "Stopping Spark Consumer..."

pkill -f stream_consumer.py

echo "Stopping Kafka..."

cd /home/ubuntu/Fraud-Detection-Project/docker

docker compose down

echo "Pipeline stopped Successfully"
