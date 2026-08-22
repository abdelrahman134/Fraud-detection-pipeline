#!/bin/bash

PUBLIC_IP=$(curl -s ifconfig.me)

echo "PUBLIC_IP=$PUBLIC_IP" > .env

docker compose up -d
