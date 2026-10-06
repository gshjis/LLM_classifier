IMAGE ?= spam-classifier
DOCKER_RUN = docker run --rm -v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/outputs:/app/outputs" $(IMAGE)
TEXT ?=

.PHONY: build test prepare-data train-baselines evaluate-baselines predict-baseline train-llm evaluate-llm predict-llm help

help:
	@echo "build                 Build the Docker image"
	@echo "test                  Run tests in Docker"
	@echo "prepare-data          Split data/raw/emails.csv"
	@echo "train-baselines       Train Naive Bayes and Logistic Regression"
	@echo "evaluate-baselines    Evaluate baseline models on test set"
	@echo "predict-baseline      Predict with TEXT='...'"
	@echo "train-llm             Fine-tune the configured model on CPU"
	@echo "evaluate-llm          Evaluate LLM on test set"
	@echo "predict-llm           Predict with TEXT='...'"

build:
	docker build -t $(IMAGE) .

test: build
	docker run --rm --entrypoint pytest $(IMAGE) tests

prepare-data:
	$(DOCKER_RUN) prepare-data

train-baselines:
	$(DOCKER_RUN) train-baselines

evaluate-baselines:
	$(DOCKER_RUN) evaluate-baselines

predict-baseline:
	@test -n "$(TEXT)" || (echo 'Usage: make predict-baseline TEXT="email text"' && exit 1)
	$(DOCKER_RUN) predict-baseline --text "$(TEXT)"

train-llm:
	$(DOCKER_RUN) train-llm

evaluate-llm:
	$(DOCKER_RUN) evaluate-llm

predict-llm:
	@test -n "$(TEXT)" || (echo 'Usage: make predict-llm TEXT="email text"' && exit 1)
	$(DOCKER_RUN) predict-llm --text "$(TEXT)"
