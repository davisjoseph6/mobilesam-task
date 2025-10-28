# syntax=docker/dockerfile:1
FROM python:3.10.12-slim

# Headless / clean logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /usr/src/app

# System deps for OpenCV + a font for PIL labels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
# 1) Install CPU-only PyTorch stack (avoids CUDA downloads)
# 2) Install the rest of your requirements, excluding torch/vision/audio if present
COPY requirements.txt .
ARG TORCH_CHANNEL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url ${TORCH_CHANNEL} \
        torch torchvision torchaudio && \
    grep -v -E '^(torch|torchvision|torchaudio)([=<>].*)?$' requirements.txt > /tmp/requirements.notorch.txt || true && \
    pip install --no-cache-dir -r /tmp/requirements.notorch.txt

# App code (incl. mobile_sam.pt)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

