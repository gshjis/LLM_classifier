# Agent notes

- Python 3.12+ project managed with Poetry; CLI entry point is `spam-classifier` in `src/llm_classifier/cli.py`.
- Main workflows: `make build`, `make prepare-data`, `make train-baselines`, `make evaluate-baselines`; run tests with `poetry run pytest` or `make test` (`make test` builds/runs Docker and needs a working Docker daemon).
- Focused test: `poetry run pytest tests/test_data.py -q`. Validate Poetry metadata/lock with `poetry check --lock`.
- Local dataset defaults to `data/raw/spam.csv` (ignored by Git). Loader accepts SMS Spam Collection columns `v1,v2` with trailing empty fields, or normalized `text,label`; labels are normalized to `spam` and `not_spam` (`ham` means `not_spam`).
- `prepare-data` generates stratified `train.csv`, `validation.csv`, and `test.csv` under `data/processed/`; input needs at least 10 examples per class. Do not use test data for training or tuning.
- Baselines live in `src/llm_classifier/baselines/`; Hugging Face fine-tuning/inference is in `src/llm_classifier/llm/`. Configuration is in `configs/`; generated models and metrics go to ignored `outputs/`.
- LLM training and prediction are explicitly CPU-only (`use_cpu=True`, pipeline `device=-1`); model downloads happen from Hugging Face on first use.
