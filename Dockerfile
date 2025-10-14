FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy service account key
COPY serviceKey.json /app/serviceAccountKey.json

# Create cache directory
RUN mkdir -p /app/model_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.10 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["python3.10", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
