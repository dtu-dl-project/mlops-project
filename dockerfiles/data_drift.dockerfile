# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml


RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install -r requirements_dev.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Expose the port (Cloud Run sets this via $PORT)
EXPOSE 8080
# Command to run the application, dynamically using $PORT
CMD ["uvicorn", "src.segmentationsuim.data_drift_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
