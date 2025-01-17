# Base image
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS base

# Install essential tools and Python
RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential \
    gcc \
    python3-dev \
    python3-pip \
    python3-setuptools && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
RUN pip install -r requirements.txt --no-cache-dir --verbose
COPY src src/
RUN pip install . --no-deps --no-cache-dir --verbose
COPY config.yaml .
COPY .env .

ENTRYPOINT ["python", "-u", "src/segmentationsuim/train.py"]
