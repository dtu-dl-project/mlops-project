# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
RUN pip install -r requirements.txt --no-cache-dir --verbose
COPY src src/
COPY configs configs/
RUN pip install . --no-deps --no-cache-dir --verbose
COPY *.env /

ENTRYPOINT ["python", "-u", "src/segmentationsuim/evaluate.py"]
