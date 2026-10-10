# Spam Classifier

Классификация SMS-сообщений на спам и не-спам: классические бейзлайны (TF-IDF) против файнтюнинга маленькой LLM на CPU.

## Содержание

- [Данные](#данные)
- [Модели](#модели)
- [Запуск](#запуск)
- [Результаты](#результаты)

## Данные

Датасет [SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), сохранённый в `data/raw/spam.csv`. Загрузчик принимает исходный формат `v1,v2` или нормализованный `text,label`; метки `ham`/`not_spam`/`0` приводятся к `not_spam`, `spam`/`1` — к `spam`.

Команда `make prepare-data` создаёт стратифицированные `train.csv`, `validation.csv` и `test.csv` в `data/processed/`. Тест используется только для финальной оценки.

## Модели

- **Бейзлайны** (`configs/baseline.yaml`) — TF-IDF (до 50k признаков, уни- и биграммы) + Multinomial Naive Bayes и Logistic Regression.
- **LLM** (`configs/llm.yaml`) — [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2) с линейной головой классификации, 1 эпоха, обучение строго на CPU.

Обученные модели и метрики сохраняются в `outputs/` (не коммитится).

## Запуск

Через Docker:

```bash
make build
make prepare-data
make train-baselines
make evaluate-baselines
make train-llm
make evaluate-llm
```

Предсказание:

```bash
make predict-baseline TEXT="Поздравляем, вы выиграли приз!"
make predict-llm TEXT="Проверьте срочно ваш аккаунт"
```

Локально (нужны Python 3.12+ и Poetry):

```bash
poetry install
poetry run spam-classifier prepare-data
poetry run spam-classifier train-baselines
poetry run pytest
```

## Результаты

На тестовой выборке (558 сообщений: 483 не-спам, 75 спам):

| Модель | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Logistic Regression | 0.987 | 0.947 | 0.960 | 0.954 |
| rubert-tiny2 | 0.984 | 0.971 | 0.907 | 0.938 |
| Naive Bayes | 0.952 | 1.000 | 0.640 | 0.780 |

Лучший баланс точности и полноты спама у Logistic Regression; Naive Bayes не пропускает ложных спамов (precision 1.0), но пропускает 36% спама; LLM чуть уступает линейной регрессии по F1.
