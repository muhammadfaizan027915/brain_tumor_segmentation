FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PostgreSQL client
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libatomic1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (WITHOUT torch)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

# Install PyTorch from official index (CPU)
RUN pip install --no-cache-dir --timeout 600 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 600 torchio==0.21.0

COPY . .

# Generate Prisma client
RUN prisma generate

# Entrypoint script to wait for DB and run migrations
RUN chmod +x entrypoint.sh cron.sh

EXPOSE 5000

CMD ["./entrypoint.sh"]