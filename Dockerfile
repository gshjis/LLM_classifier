FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/outputs/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# 1) Сначала ставим только зависимости из pyproject.toml (это позволяет лучше кешировать слой при изменениях в коде)
RUN pip install --upgrade pip \
    && python - <<'PY'
import pathlib, tomllib

pyproject = tomllib.loads(pathlib.Path('pyproject.toml').read_text('utf-8'))
deps = pyproject['project']['dependencies']
pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(deps), encoding='utf-8')
PY
    && pip install -r /tmp/requirements.txt \
    && pip install pytest

# 2) Затем копируем исходники и ставим сам проект (без зависимостей, т.к. они уже установлены)
COPY src ./src
RUN pip install . --no-deps

# 3) Для целей тестов контейнеру нужны тесты
COPY tests ./tests

RUN mkdir -p /app/data/raw /app/data/processed /app/outputs/models /app/outputs/metrics /app/outputs/logs

ENTRYPOINT ["spam-classifier"]
CMD ["--help"]
