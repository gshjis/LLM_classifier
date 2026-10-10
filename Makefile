IMAGE ?= spam-classifier
DOCKER_RUN = docker run --rm -v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/outputs:/app/outputs" -v "$(CURDIR)/configs:/app/configs" $(IMAGE)
TEXT ?=

.PHONY: build test requirements download-data prepare-data train-baselines evaluate-baselines predict-baseline train-llm evaluate-llm predict-llm help

help:
	@echo "build                 Build the Docker image"
	@echo "test                  Run tests in Docker"
	@echo "requirements          Regenerate requirements*.txt from poetry.lock (CPU-only torch)"
	@echo "download-data         Download spam.csv into data/raw/spam.csv (via kagglehub)"
	@echo "prepare-data          Split data/raw/spam.csv into train/validation/test"
	@echo "train-baselines       Train Naive Bayes and Logistic Regression"
	@echo "evaluate-baselines    Evaluate baseline models on test set"
	@echo "predict-baseline      Predict with TEXT='...'"
	@echo "train-llm             Fine-tune the configured model on CPU"
	@echo "evaluate-llm          Evaluate LLM on test set"
	@echo "predict-llm           Predict with TEXT='...'"

build:
	docker build --target runtime -t $(IMAGE) .

test:
	docker build --target test -t $(IMAGE)-test .
	docker run --rm $(IMAGE)-test

# Регенерация requirements*.txt из poetry.lock (Docker ставит зависимости из них).
# Нужен плагин: poetry self add poetry-plugin-export
requirements:
	@poetry export --help >/dev/null 2>&1 || { \
		echo "poetry-plugin-export is missing. Install with: poetry self add poetry-plugin-export"; \
		exit 1; \
	}
	poetry lock
	poetry export -f requirements.txt --without-hashes --only main -o requirements.txt.tmp
	sed -i -e 's/^torch==\([0-9.]*\)$$/torch==\1+cpu/' requirements.txt.tmp
	{ \
		echo "# Auto-generated from poetry.lock via 'make requirements' — pinned exact versions for reproducible Docker builds."; \
		echo "# CPU-only torch: install index must include https://download.pytorch.org/whl/cpu"; \
		echo "--extra-index-url https://download.pytorch.org/whl/cpu"; \
		cat requirements.txt.tmp; \
	} > requirements.txt
	rm -f requirements.txt.tmp
	poetry export -f requirements.txt --without-hashes --only dev -o requirements-dev.txt.tmp
	{ \
		echo "# Auto-generated from poetry.lock via 'make requirements' — dev/test dependencies (pytest stack)."; \
		cat requirements-dev.txt.tmp; \
	} > requirements-dev.txt
	rm -f requirements-dev.txt.tmp
	@echo "Regenerated requirements.txt and requirements-dev.txt from poetry.lock"

download-data:
	$(DOCKER_RUN) download-data --output /app/data/raw/spam.csv

prepare-data:
	$(DOCKER_RUN) prepare-data

train-baselines:
	$(DOCKER_RUN) train-baselines

evaluate-baselines:
	$(DOCKER_RUN) evaluate-baselines

predict-baseline:
	@test -n "$(TEXT)" || (echo 'Usage: make predict-baseline TEXT="SMS message"' && exit 1)
	$(DOCKER_RUN) predict-baseline --text "$(TEXT)"

train-llm:
	$(DOCKER_RUN) train-llm

evaluate-llm:
	$(DOCKER_RUN) evaluate-llm

predict-llm:
	@test -n "$(TEXT)" || (echo 'Usage: make predict-llm TEXT="SMS message"' && exit 1)
	$(DOCKER_RUN) predict-llm --text "$(TEXT)"
