# syntax=docker/dockerfile:1
# Мультистейджинг: builder (зависимости + пакет) -> test (pytest) -> runtime (только приложение).
# Зависимости закреплены в requirements*.txt (сгенерированы из poetry.lock, CPU-only torch).

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Закреплённые зависимости из poetry.lock: слой кешируется, пока не изменится requirements.txt.
#    В pip-кеш BuildKit ставим mount, чтобы кеш не попадал в образ и ускорял пересборку.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# 2) Ставим сам проект (без зависимостей — они уже установлены).
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install . --no-deps

# Стадия тестов: pytest + tests/. В runtime-образ не попадает.
FROM builder AS test

COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

COPY tests ./tests

ENTRYPOINT ["pytest"]
CMD ["tests"]

# Финальный образ: только приложение и его зависимости, без pytest и tests/.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/outputs/huggingface

WORKDIR /app

COPY --from=builder /usr/local /usr/local

RUN mkdir -p /app/data/raw /app/data/processed /app/outputs/models /app/outputs/metrics /app/outputs/logs

ENTRYPOINT ["spam-classifier"]
CMD ["--help"]
