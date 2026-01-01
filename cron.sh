#!/bin/sh

while true; do
  echo "Starting the cleanup worker..."
  python workers/clear-session-content.py
  sleep 60 * 15
done