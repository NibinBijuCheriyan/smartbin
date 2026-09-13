# Multi-stage production Dockerfile for SmartBin AI v2
FROM python:3.11-slim-bullseye AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for OpenCV and GUI/camera
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY smartbin-v2/requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source repository
COPY . /app

EXPOSE 8000 8501

# Default CMD runs the FastAPI backend
CMD ["uvicorn", "smartbin_v2.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
