FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install torch==2.11.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r requirements.txt

COPY handler.py .
COPY model.py .
COPY schemas.py .

CMD ["python", "-u", "handler.py"]