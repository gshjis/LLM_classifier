FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/outputs/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install --upgrade pip && pip install . pytest

RUN mkdir -p /app/data/raw /app/data/processed /app/outputs/models /app/outputs/metrics /app/outputs/logs

ENTRYPOINT ["spam-classifier"]
CMD ["--help"]
