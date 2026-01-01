#!/bin/bash

# Wait for PostgreSQL to be ready
until pg_isready -h db -p 5432 -U user -d brain_tumor_db; do
  echo "Waiting for database..."
  sleep 2
done

echo "Database is ready!"

# Run Prisma migrations
prisma migrate deploy

# Start the application with Quart
quart run --host 0.0.0.0 --port 5000