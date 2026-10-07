# Spam Classifier

Учебный проект для классификации SMS-сообщений на спам и не спам с использованием классических ML-бейзлайнов и файнтюнинга небольшой LLM. Все команды можно запускать локально через Poetry или в Docker.

## Требования

- Для запуска через Docker нужен Docker Engine; Docker Compose не используется.
- Локально: Python 3.12+ и Poetry 2.x.
- GPU/CUDA не требуются. Файнтюнинг LLM запускается на CPU и может быть медленным.

Makefile предоставляет короткие команды `make ...`, которые собирают Docker-образ или запускают приложение внутри контейнера. Например, `make build` выполняет `docker build`, а `make train-baselines` запускает контейнер с командой `spam-classifier train-baselines`. Каталоги `data/` и `outputs/` подключаются из проекта в контейнер, поэтому подготовленные данные, модели и метрики сохраняются на компьютере.

## Данные

В репозитории используется датасет [SMS Spam Collection на Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download), сохранённый в `data/raw/spam.csv`. Загрузчик понимает исходные столбцы `v1,v2` (и возможные лишние пустые столбцы), а также нормализованный формат `text,label`. Для стандартного разбиения нужно не менее 10 примеров каждого класса.

```csv
text,label
"Поздравляем, вы выиграли приз!",spam
"Встреча перенесена на завтра",ham
```

Поддерживаемые метки: `spam` и `ham`/`not_spam`, а также `1`/`0`. При подготовке `ham` преобразуется в каноническую метку `not_spam`. В датасете должны присутствовать оба класса. Не помещайте приватные сообщения в публичный репозиторий.

Подготовка создаст стратифицированные `train.csv`, `validation.csv` и `test.csv` в `data/processed/`. Тестовая выборка используется только для финальной оценки.

## Быстрый старт через Docker

```bash
make build
make prepare-data
make train-baselines
make evaluate-baselines
```

Для инференса бейзлайна:

```bash
make predict-baseline TEXT="Поздравляем, вы выиграли приз!"
```

`make test` пересобирает образ и запускает в контейнере `pytest tests`. Если Docker не используется, тесты можно запустить локально командой `poetry run pytest`.

Файнтюнинг и оценка LLM (на CPU, модель скачивается с Hugging Face при первом запуске):

```bash
make train-llm
make evaluate-llm
make predict-llm TEXT="Проверьте срочно ваш аккаунт"
```

В Docker-монтируются `data/` и `outputs/`, поэтому датасет и модели остаются на хосте. Передавайте длинный текст в Make как `TEXT='...'`.

## Запуск локально

```bash
poetry install
poetry run spam-classifier prepare-data
poetry run spam-classifier train-baselines
poetry run spam-classifier evaluate-baselines
poetry run pytest
```

## Основные команды CLI

```text
spam-classifier prepare-data [--input PATH] [--output-dir PATH]
spam-classifier train-baselines [--config PATH]
spam-classifier evaluate-baselines [--config PATH]
spam-classifier predict-baseline --text TEXT
spam-classifier train-llm [--config PATH]
spam-classifier evaluate-llm [--config PATH]
spam-classifier predict-llm --text TEXT
```

См. `configs/` для параметров. Обученные артефакты и метрики сохраняются в `outputs/`; они исключены из Git. Модель LLM по умолчанию — `cointegrated/rubert-tiny2`; можно заменить её в `configs/llm.yaml`.

## Структура

- `src/llm_classifier/baselines/` — TF-IDF + Multinomial Naive Bayes и Logistic Regression.
- `src/llm_classifier/llm/` — файнтюнинг последовательного классификатора Hugging Face на CPU.
- `tests/` — тесты обработки данных, метрик, бейзлайнов и предсказаний.
- `data/raw/` — исходный SMS-датасет; `data/processed/` — готовые выборки.

## Инструменты и команды Docker

- **Make** запускает команды проекта по коротким именам, например `make build` или `make train-baselines`. Make сам не обучает модели — он вызывает Docker или локальные инструменты.
- **Docker image (образ)** — шаблон окружения приложения. `make build` собирает его из `Dockerfile` и помечает тегом `spam-classifier`.
- **Docker container (контейнер)** — временный запущенный экземпляр образа. Большинство целей Make запускают его через `docker run --rm`: `--rm` удаляет контейнер после выполнения, но не образ.
- **Монтирование папок** — параметр `-v` подключает локальные `data/` и `outputs/` к `/app/data` и `/app/outputs` в контейнере. Поэтому данные и результаты остаются на хосте после удаления контейнера.
- В `Dockerfile` задан `ENTRYPOINT ["spam-classifier"]`, поэтому, например, `docker run spam-classifier prepare-data` фактически вызывает `spam-classifier prepare-data` внутри контейнера.
- Для тестов Make переопределяет entrypoint: `docker run --rm --entrypoint pytest spam-classifier tests`. Это запускает `pytest tests` вместо CLI приложения.

Основные цели Make: `build`, `test`, `prepare-data`, `train-baselines`, `evaluate-baselines`, `predict-baseline`, `train-llm`, `evaluate-llm` и `predict-llm`. Для команд предсказания передавайте текст через `TEXT`, например `make predict-llm TEXT="Проверьте сообщение"`.
