"""Baseline model construction and persistence."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_models(config: dict) -> dict[str, Pipeline]:
    vectorizer_config = config.get("tfidf", {})
    if "ngram_range" in vectorizer_config:
        vectorizer_config["ngram_range"] = tuple(vectorizer_config["ngram_range"])
    models = {}
    for name, classifier in {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.get("random_state", 42)),
    }.items():
        models[name] = Pipeline(
            [("tfidf", TfidfVectorizer(**vectorizer_config)), ("classifier", classifier)]
        )
    return models


def save_model(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
