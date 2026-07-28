FROM python:3.11-slim

# System deps for scikit-learn / matplotlib native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create model cache dir (mapped to Fly volume in production)
RUN mkdir -p model_cache

EXPOSE 8080

# workers=1 is intentional: scheduler.start() spawns a background thread on
# import, and each gunicorn worker is a separate process — 2+ workers means
# 2+ independent schedulers (duplicate SMS broadcasts, duplicate auto-sell
# checks). Use --threads for request concurrency instead of more workers.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 300 --keep-alive 5