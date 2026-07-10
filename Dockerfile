FROM python:3.11-slim

# System deps for TensorFlow + MySQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 pkg-config \
    default-libmysqlclient-dev \
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

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", \
     "--timeout", "300", "--keep-alive", "5", "app:app"]
