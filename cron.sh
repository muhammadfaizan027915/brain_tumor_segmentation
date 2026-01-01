#!/bin/sh

while true; do
  echo "Starting the cleanup worker..."
  cd /app
  python /workers/clear-session-content.py
  sleep 900
done