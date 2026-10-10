FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

# Install system packages required for native Python packages like annoy
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements-prod.txt .

# Upgrade packaging tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install production dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements-prod.txt

# Copy only backend application
COPY app/ ./app/

# Render provides PORT automatically
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
