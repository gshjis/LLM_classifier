# Spam Classifier

Учебный проект для знакомства с классификацией текстовых сообщений на спам и не спам, классическими ML-бейзлайнами, файнтюнингом небольшой LLM и CPU-инференсом. Исходный набор — SMS Spam Collection; это SMS, не email. Все команды можно запускать локально через Poetry или в Docker.

## Требования

- Docker и Docker Compose не нужны (достаточно Docker Engine).
- Локально: Python 3.12+ и Poetry 2.x.
- GPU/CUDA не требуются. Файнтюнинг LLM запускается на CPU и может быть медленным.

## Данные

В репозитории используется `data/raw/spam.csv` из SMS Spam Collection. Загрузчик понимает исходные столбцы `v1,v2` (и возможные лишние пустые столбцы), а также нормализованный формат `text,label`. Для стандартного разбиения нужно не менее 10 примеров каждого класса.

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
